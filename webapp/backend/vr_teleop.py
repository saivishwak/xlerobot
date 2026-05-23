"""Safe single-arm VR teleop session.

The webapp's only control surface. Drives **one** arm at a time via the
Meta Quest 3 controller stream, with three independent safety guards:

  1. Engagement gate — UI toggle. Motors stay still until the user flips it.
  2. Calibration gate — even while engaged, motors stay still until the user
     issues a Quest-controller RESET. The RESET anchors VR poses to the
     robot's current EE pose, so motion is always *relative* and bounded.
  3. Watchdog — if VR goals stop arriving (controller put down, browser
     closed, network blip), the drive loop stops sending within 0.3 s and
     auto-disengages within 1 s.

Plus per-tick joint clamps and the degree-space calibration bounds already
in motors.SESSION.bounds.

The drive math is the same shape as
XLerobot/software/examples/4_xlerobot_teleop_keyboard.py:SimpleTeleopArm:
  - Use the 2-link analytical IK for (shoulder_lift, elbow_flex) from EE (x, y).
  - Direct delta-mapping for shoulder_pan / wrist_flex / wrist_roll / gripper.
  - P-controlled action: write `present + kp * (target - present)`.

We do NOT use the upstream full URDF-based IK pipeline (RobotKinematics +
EEReferenceAndDelta + …) because the upstream script hard-codes a placeholder
URDF path. The analytical 2-link IK in lerobot/.../SO101Robot.py is what the
keyboard examples already trust on this hardware, so we reuse the same math.
"""
from __future__ import annotations

import asyncio
import http.server
import logging
import math
import os
import pathlib
import socket
import ssl
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from . import motors as _motors
from .motors import SESSION as MOTORS, ArmSide
from . import dataset as _dataset
from . import home as _home
from . import vr_calibration as _vrcal

log = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
XLEVR_DIR = REPO_ROOT / "XLerobot_xuweiwu" / "XLeVR"

# Make the extended XLeVR (with relative_position / relative_rotvec / RESET mode)
# importable. The setup script wired the OLDER XLeRobot/XLeVR onto sys.path; this
# overrides that.
if str(XLEVR_DIR) not in sys.path:
    sys.path.insert(0, str(XLEVR_DIR))

# --- control / safety constants -----------------------------------------------

LOOP_HZ = 30.0
LOOP_PERIOD_S = 1.0 / LOOP_HZ

GOAL_SKIP_AGE_S = 0.30     # skip motor write if VR goal older than this
# Moderate software P-control matches the smoother xuweiwu dual-arm reference.
# It damps residual IK/servo noise after the target filters below.
KP = 0.75

# Per-tick caps. At LOOP_HZ=30 these are effectively doubled in deg/sec vs the
# old 15 Hz tuning, putting max joint speeds in the natural-hand-motion range:
#   shoulder_pan 4°/tick × 30Hz = 120°/s
#   shoulder_lift 2°/tick × 30Hz = 60°/s (gravity-loaded, kept tighter)
#   elbow_flex 3°/tick × 30Hz = 90°/s
#   wrist 4°/tick × 30Hz = 120°/s
PER_TICK_DEG_CAPS: dict[str, float] = {
    # At 30 Hz: 5°/tick = 150°/s, 6°/tick = 180°/s. These are deliberately
    # below the recent 8-12° caps but still responsive enough for hand teleop.
    "shoulder_pan":  5.0,
    "shoulder_lift": 5.0,
    "elbow_flex":    5.0,
    "wrist_flex":    6.0,
    "wrist_roll":    6.0,
    "gripper":       15.0,   # ~0.25s open/close
}

# Per-frame VR delta caps BEFORE the joint clamp. At scale=1.0, 6 mm/tick ×
# 30 Hz = 18 cm/s EE speed, between the reference's 3 mm and the recent 20 mm.
EE_DELTA_LIMIT_M = 0.006
WRIST_RAD_DELTA_LIMIT = math.radians(5)

# Mapping from VR controller frame to robot base frame (upstream's convention,
# upstream/xuweiwu's 8_vr_teleop_with_dataset_recording_dualarm.py lines 327–330):
#     robot_x (forward from base) ← -vr_z      (controller forward = away from operator)
#     robot_y (sideways)          ← -vr_x      (controller right = robot's left side)
#     robot_z (vertical up)       ←  vr_y
# Full 3D EE position is tracked (not just planar). At each tick:
#   shoulder_pan_target = atan2(target_y, target_x)   ← arm yaws toward EE
#   r_horizontal        = hypot(target_x, target_y)   ← forward distance in arm's plane
#   (sl, ef)            = analytical IK on (r_horizontal, target_z)
# This is the standard 5DOF SCARA-style decomposition: position via 3-joint IK,
# orientation via wrist_flex + wrist_roll. EE yaw follows shoulder_pan automatically.

# EE-position safety box in robot base frame (metres). Tuned for the SO101's
# actual reach (L1+L2 = 0.251m). Includes the full hemisphere in front + sides,
# with small clearance behind the base (in case shoulder_pan rotates >90°).
# The IK and motor-calibration clamps downstream are the final safety net.
EE_BOUNDS = {
    # The URDF-IK reach envelope is ~0.45 m in front of the base. These bounds
    # are a sanity box outside which we don't even try IK — the IK itself does
    # the soft saturation against actual joint limits.
    "x": (-0.45, 0.45),
    "y": (-0.45, 0.45),
    "z": (-0.30, 0.45),
}

# Gripper convention. Default 100 = open, 0 = closed. If your SO101 calibration
# is the opposite (calibrated with the gripper open at the "min" tick instead of
# the "max" tick), override `gripper.open_value` / `gripper.closed_value` in
# config/xlerobot.yaml.
DEFAULT_GRIPPER_OPEN = 100.0
DEFAULT_GRIPPER_CLOSED = 0.0

# Quest face-button mapping per controller side. The lower face button on each
# controller (A on right, X on left) toggles engage for THAT controller's arm.
# Only the right controller has a B button — it's the global recording toggle.
ENGAGE_BUTTON_BY_SIDE: dict[str, str] = {"right": "A", "left": "X"}
RECORD_BUTTON_BY_SIDE: dict[str, str] = {"right": "B"}

# Minimum motion magnitude (in metres) required to accept a calibration. Anything
# smaller than this is too noisy to reliably determine "user-forward" direction.
CALIBRATION_MIN_MOTION_M: float = 0.05    # 5 cm
CALIBRATION_TARGET_MOTION_M: float = 0.10  # the wizard says "move ~10 cm"

# Smoothing factors. 1.0 = raw input, 0.0 = frozen.
POS_EMA_ALPHA: float = 0.5
ORI_EMA_ALPHA: float = 0.4
JOINT_EMA_ALPHA: float = 0.2


# Homing: per-joint tolerance (degrees) to declare "arrived". 1.5° on
# Present_Position is OK in theory but the motor's internal PID has a small
# deadband, so the physical position often settles a few degrees from the
# commanded target and never converges to within 1.5° — making the UI hang on
# "HOMING…" forever. We now check SOFTWARE convergence (last_sent_targets
# equals the home target) instead, which is deterministic. The 0.5° threshold
# below is just for the per-tick-clamped value, which converges exactly.
HOMING_TOL_DEG: float = 0.5
HOMING_TIMEOUT_S: float = 15.0   # hard cap; if not converged by then, give up


# --- SO101 analytical kinematics ---------------------------------------------
#
# Inverse kinematics formula is transcribed verbatim from
# `lerobot.model.SO101Robot.SO101Kinematics.inverse_kinematics` (the math used by
# XLerobot keyboard teleop and the upstream VR script). Reproduced here because
# importing that module fails: SO101Robot.py has `from lerobot.robots.so101_follower...`
# which doesn't exist in the current lerobot layout.
#
# Forward kinematics is derived by inverting the IK chain, including the same
# theta1_offset / theta2_offset for the SO101's joint-zero geometry and the final
# 90°-transform. FK ↔ IK round-trip is exact within numerical precision for joint
# angles inside the IK's clamp range; for poses outside the IK envelope the FK
# still returns sensible values but the IK→FK roundtrip won't recover them.

class _SO101Kin:
    """Analytical 2-link kinematics for SO101 with joint-zero offsets baked in."""

    # Constants from lerobot.model.SO101Robot.SO101Kinematics
    THETA1_OFFSET = math.atan2(0.028, 0.11257)                              # ≈ 0.244 rad / 14°
    THETA2_OFFSET = math.atan2(0.0052, 0.1349) + THETA1_OFFSET              # ≈ 0.282 rad / 16°
    # Internal joint pre-clamps. Widened beyond the upstream's [-0.1, 3.45] /
    # [-0.2, π] so the IK can output the full range the SO101's URDF + motor
    # calibration actually allows (shoulder_lift ±100°, elbow_flex ±96.8°).
    # The motor calibration clamp downstream is the actual safety guard.
    JOINT2_PRE_MIN = -0.20    # shoulder_lift output max: 90 - degrees(-0.20) = +101.5°
    JOINT2_PRE_MAX = 3.65     # shoulder_lift output min: 90 - degrees(3.65) = -119.2°
    JOINT3_PRE_MIN = -0.30    # elbow_flex output min: degrees(-0.30) - 90 = -107.2°
    JOINT3_PRE_MAX = 3.55     # elbow_flex output max: degrees(3.55) - 90 = +113.4°

    def __init__(self, l1: float = 0.1159, l2: float = 0.1350):
        self.l1, self.l2 = l1, l2

    def inverse(self, x: float, y: float) -> tuple[float, float]:
        """(x, y) in IK plane (metres, base frame) → (shoulder_lift_deg, elbow_flex_deg).
        Output is in lerobot's degree convention (matches what the motor expects)."""
        l1, l2 = self.l1, self.l2
        # Workspace scaling — if target is beyond reach, scale onto the boundary.
        r = math.hypot(x, y)
        r_max = l1 + l2
        if r > r_max:
            scale_factor = r_max / r
            x *= scale_factor; y *= scale_factor; r = r_max
        r_min = abs(l1 - l2)
        if 0 < r < r_min:
            scale_factor = r_min / r
            x *= scale_factor; y *= scale_factor; r = r_min

        # Law of cosines (note the leading minus — upstream's convention).
        cos_theta2 = -(r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))
        theta2 = math.pi - math.acos(cos_theta2)

        beta = math.atan2(y, x)
        gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
        theta1 = beta + gamma

        joint2 = theta1 + self.THETA1_OFFSET
        joint3 = theta2 + self.THETA2_OFFSET
        # Pre-transform clamp (the upstream's safety net for URDF joint limits).
        joint2 = max(self.JOINT2_PRE_MIN, min(self.JOINT2_PRE_MAX, joint2))
        joint3 = max(self.JOINT3_PRE_MIN, min(self.JOINT3_PRE_MAX, joint3))

        # Final coordinate transform to match SO101 motor convention.
        sl_deg = 90 - math.degrees(joint2)
        ef_deg = math.degrees(joint3) - 90
        return sl_deg, ef_deg

    def forward(self, sl_deg: float, ef_deg: float) -> tuple[float, float]:
        """(shoulder_lift_deg, elbow_flex_deg) → (x, y) in IK plane (metres).
        Exact inverse of the IK formula above (assuming joints inside the clamp range)."""
        # Reverse the final coordinate transform.
        joint2 = math.radians(90 - sl_deg)
        joint3 = math.radians(ef_deg + 90)
        theta1 = joint2 - self.THETA1_OFFSET
        theta2 = joint3 - self.THETA2_OFFSET
        gamma = math.atan2(self.l2 * math.sin(theta2),
                           self.l1 + self.l2 * math.cos(theta2))
        beta = theta1 - gamma
        # r² = l1² + l2² + 2·l1·l2·cos(theta2), derived from the IK's cos_theta2 formula.
        r_sq = self.l1 ** 2 + self.l2 ** 2 + 2 * self.l1 * self.l2 * math.cos(theta2)
        r = math.sqrt(max(0.0, r_sq))
        return r * math.cos(beta), r * math.sin(beta)

    @classmethod
    def sl_deg_in_ik_envelope(cls, sl_deg: float) -> bool:
        """True if `sl_deg` is in a region the IK can actually generate (i.e. its FK
        output round-trips through IK back to (sl_deg, _)). Used to warn at RESET
        when the user's arm is sitting outside the IK envelope."""
        joint2 = math.radians(90 - sl_deg)
        return cls.JOINT2_PRE_MIN <= joint2 <= cls.JOINT2_PRE_MAX


# ─── URDF-based 5-DOF IK ──────────────────────────────────────────────────────
#
# Uses lerobot.model.kinematics.RobotKinematics → wraps the `placo` solver against
# the SO-ARM100/Simulation/SO101/so101_new_calib.urdf model. Returns the best 5-joint
# solution that approximates a 6-DOF target EE pose (position + orientation), weighting
# position higher than orientation since the SO101 can't match arbitrary orientations
# (it has no EE yaw DOF independent of shoulder_pan).
#
# Joint name → array index: ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']

_SO101_URDF = REPO_ROOT / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
_IK_JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# VR controller tracking noise. Sub-mm deltas at rest = controller jitter, not
# intentional motion. Below this threshold (per drive tick, AFTER accumulation),
# treat as zero. At 30 Hz with ~2 VR frames per tick, controller jitter is ~0.6mm,
# so 0.8 mm catches noise without eating real slow motion.
POSITION_DEADBAND_M = 0.0008     # 0.8 mm per tick
ROTVEC_DEADBAND_RAD = math.radians(0.3)  # 0.3° per tick

# SO101 reach. Used to clamp the running EE target inside the actual workspace —
# placo's solver is unstable when target is outside the reach envelope (it returns
# different local minima per tick → jitter).
WORKSPACE_REACH_M = 0.45         # stay inside URDF reach to avoid IK edge flips


def _load_urdf_kinematics():
    """Lazily import + construct RobotKinematics. Returns None on failure so the
    drive loop can fall back to the analytical IK."""
    try:
        from lerobot.model.kinematics import RobotKinematics
        if not _SO101_URDF.is_file():
            log.warning("URDF not found at %s; falling back to analytical IK", _SO101_URDF)
            return None
        return RobotKinematics(
            urdf_path=str(_SO101_URDF),
            target_frame_name="gripper_frame_link",
            joint_names=list(_IK_JOINT_ORDER),
        )
    except Exception as e:
        log.warning("failed to load URDF kinematics: %s; falling back to analytical IK", e)
        return None


# Default VR → robot base frame rotation. Used as the initial value of the
# *session* matrix before the first RESET; assumes the user is standing facing
# the robot at session start (controller forward = -VR.z = +robot.x):
#   vr.x (right)  → robot -y       (controller right = robot's left side)
#   vr.y (up)     → robot  z       (vertical preserved)
#   vr.z (back)   → robot -x       (controller forward = robot forward)
# At every RESET (grip-press), `_compute_session_frame` re-derives this matrix
# from the controller's actual orientation at RESET — so the user's "forward"
# (controller's barrel direction at grip-press) becomes "robot forward" regardless
# of which way they happen to be facing in the room.
import numpy as _np
_VR_TO_ROBOT = _np.array([[0, 0, -1],
                          [-1, 0, 0],
                          [0, 1, 0]], dtype=float)


def _slerp_rotation_matrix(
    previous: _np.ndarray,
    target: _np.ndarray,
    alpha: float,
    max_step_rad: float | None = None,
) -> _np.ndarray:
    """EMA-style smoothing for rotation matrices using shortest-path SLERP."""
    if alpha <= 0.0:
        return previous.copy()
    if alpha >= 1.0 and max_step_rad is None:
        return target.copy()
    from scipy.spatial.transform import Rotation as _R, Slerp as _Slerp

    prev_r = _R.from_matrix(previous)
    target_r = _R.from_matrix(target)
    candidate = _Slerp([0.0, 1.0], _R.concatenate([prev_r, target_r]))([alpha])[0]

    if max_step_rad is not None:
        step_angle = float((candidate * prev_r.inv()).magnitude())
        if step_angle > max_step_rad > 0.0:
            capped_alpha = max_step_rad / step_angle
            candidate = _Slerp([0.0, 1.0], _R.concatenate([prev_r, candidate]))([capped_alpha])[0]
    return candidate.as_matrix()


def _clamp_to_workspace_reach(position: _np.ndarray) -> _np.ndarray:
    """Keep the requested EE target inside the robot-base reach sphere."""
    radius = float(_np.linalg.norm(position))
    if radius <= WORKSPACE_REACH_M or radius <= 1e-9:
        return position
    return position * (WORKSPACE_REACH_M / radius)


def _compute_session_frame_from_two_motions(
    motion_fwd_vr: tuple[float, float, float],
    motion_up_vr: tuple[float, float, float],
) -> tuple[_np.ndarray, str]:
    """Build the full 3D session VR→robot rotation matrix from two USER-MOTION
    vectors. Returns `(matrix, confidence)` where confidence is "good" if the
    vectors are well-separated, "poor" if too parallel (matrix is shaky).

    Cosine threshold: 0.6 (≈ 53° between vectors). Below that, the
    Gram-Schmidt orthogonalization throws away too much information from the
    user's motion intent.
    """
    f = _np.array(motion_fwd_vr, dtype=float)
    u = _np.array(motion_up_vr, dtype=float)
    fn = float(_np.linalg.norm(f))
    if fn < 1e-3:
        log.warning("calibration forward motion too small; using default frame")
        return _VR_TO_ROBOT.copy(), "poor"
    fwd_axis = f / fn

    # Pre-orthogonalize confidence check.
    u_norm = float(_np.linalg.norm(u))
    confidence = "good"
    if u_norm > 1e-3:
        cos_raw = abs(float(_np.dot(u, fwd_axis) / u_norm))
        if cos_raw > 0.6:
            confidence = "poor"
            log.warning(
                "calibration motions are %0.1f° apart (cos=%.2f) — too parallel; "
                "matrix confidence is POOR. Re-run wizard with more orthogonal motions.",
                math.degrees(math.acos(min(1.0, cos_raw))), cos_raw,
            )

    u_orth = u - _np.dot(u, fwd_axis) * fwd_axis
    un = float(_np.linalg.norm(u_orth))
    if un < 1e-3:
        log.warning("calibration up motion parallel to forward; falling back to yaw-only")
        return _compute_session_frame_from_motion(motion_fwd_vr), "poor"
    up_axis = u_orth / un

    right_axis = _np.cross(up_axis, fwd_axis)
    right_axis /= float(_np.linalg.norm(right_axis))

    return _np.stack([fwd_axis, right_axis, up_axis], axis=0), confidence


def _compute_session_frame_from_motion(motion_vr: tuple[float, float, float]) -> _np.ndarray:
    """Build the per-session VR→robot rotation matrix from a USER MOTION vector
    rather than the controller's orientation.

    This is the calibration-wizard path: the user squeezes grip (anchor) and then
    physically moves their hand in the direction they consider "forward" (typically
    toward the robot / workspace). We capture the motion vector in VR world frame,
    project it to horizontal (drop vertical — we only calibrate yaw, never tilt),
    and use it as the new robot-+X direction.

    Far more robust than reading the controller's barrel orientation at grip-press:
    motion direction reflects what the *user's body* considers forward, independent
    of how they happened to be holding the controller.
    """
    horiz = _np.array([motion_vr[0], 0.0, motion_vr[2]])
    norm = float(_np.linalg.norm(horiz))
    if norm < 1e-3:
        log.warning("calibration motion magnitude too small to determine yaw; "
                    "keeping previous session frame")
        return _VR_TO_ROBOT.copy()
    fwd_horiz = horiz / norm
    up_vr = _np.array([0.0, 1.0, 0.0])
    row_x = fwd_horiz
    cross = _np.cross(up_vr, fwd_horiz)
    row_y = cross / float(_np.linalg.norm(cross))
    row_z = up_vr
    return _np.stack([row_x, row_y, row_z], axis=0)


def _compute_session_frame(anchor_quat: tuple[float, float, float, float]) -> _np.ndarray:
    """Given the controller's quaternion at RESET, build a 3×3 matrix M such that
    `v_robot = M @ v_vr` aligns the controller's barrel direction (forward in user
    hand-space) with the robot's +X axis. VR's +Y (up) remains robot's +Z (up) —
    we only calibrate the YAW; vertical is always preserved.

    The controller's local "forward" axis in WebXR/A-Frame is -Z_local. We rotate
    that into VR world frame, project to the horizontal plane (drop the vertical
    component — the user might be holding the controller tilted up/down, but we
    only care about which compass direction they're pointing), and use that as
    the new robot-+X axis in VR coordinates.
    """
    from scipy.spatial.transform import Rotation as _R

    # Controller-local forward in VR world frame.
    R_anchor = _R.from_quat(_np.array(anchor_quat))
    fwd_local = _np.array([0.0, 0.0, -1.0])     # WebXR controller forward is -Z_local
    fwd_vr = R_anchor.as_matrix() @ fwd_local   # 3-vector in VR world frame

    # Project to horizontal (drop Y, the VR up axis) and normalise.
    horiz = _np.array([fwd_vr[0], 0.0, fwd_vr[2]])
    norm = float(_np.linalg.norm(horiz))
    if norm < 1e-3:
        # Controller is pointing straight up or down — can't determine yaw.
        # Fall back to the default fixed transform (user was probably holding the
        # controller normally and got a numerical edge case; safer than NaNs).
        log.warning("session frame: controller pointing near-vertical; using default _VR_TO_ROBOT")
        return _VR_TO_ROBOT.copy()
    fwd_horiz = horiz / norm

    # Build the new VR→robot rotation. Columns of M^T are VR basis vectors in
    # robot frame; equivalently rows of M are robot basis vectors in VR frame.
    up_vr = _np.array([0.0, 1.0, 0.0])
    # robot.+x in VR coordinates = the user's forward direction
    row_x = fwd_horiz
    # robot.+y in VR coordinates = up × forward (right-handed; robot.+y is "robot's left")
    row_y = _np.cross(up_vr, fwd_horiz)
    row_y /= float(_np.linalg.norm(row_y))
    # robot.+z in VR coordinates = VR up
    row_z = up_vr

    return _np.stack([row_x, row_y, row_z], axis=0)


# --- VR data snapshots --------------------------------------------------------

@dataclass
class _LatestGoal:
    """Snapshot of the most recent VR goal — used for status display and trigger/thumb.
    Position/rotation deltas are NOT taken from here; see `_DeltaAccumulator` below.

    `buttons` carries the Quest face-button pressed-state, keyed by Meta's labels:
        right controller → {"A": bool, "B": bool}
        left  controller → {"X": bool, "Y": bool}
    """
    received_at: float = 0.0
    has_data: bool = False
    mode: str = "idle"            # "idle" | "position" | "reset"
    rel_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rel_rotvec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_quat: Optional[tuple[float, float, float, float]] = None
    trigger: bool = False
    thumbstick: tuple[float, float] = (0.0, 0.0)
    buttons: dict[str, bool] = field(default_factory=dict)


@dataclass
class _DeltaAccumulator:
    """Sums VR per-frame deltas (position + rotvec) between drive ticks.

    XLeVR's WSS server already does per-frame differencing: each `relative_position`
    is `current_frame - last_frame`, after which its "last" pose is reassigned to
    current. The same applies to `relative_rotvec`. The WSS pump runs at the VR
    frame rate (~60 Hz); the drive loop runs at LOOP_HZ. Without accumulating, the
    drive loop would only see the most recent ~16 ms of motion and miss the rest.

    Rotvec sum is only an approximation for general rotations, but for the
    ~16 ms-worth deltas the WSS sends each frame it's accurate to small-angle
    order. Per-tick this is what we hand to the IK + wrist mapping.
    """
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotvec: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def add(self, dp: tuple[float, float, float], dr: tuple[float, float, float]) -> None:
        self.pos = (self.pos[0] + dp[0], self.pos[1] + dp[1], self.pos[2] + dp[2])
        self.rotvec = (self.rotvec[0] + dr[0], self.rotvec[1] + dr[1], self.rotvec[2] + dr[2])

    def drain(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Atomically read + zero the accumulator. Caller holds the lock."""
        p, r = self.pos, self.rotvec
        self.pos = (0.0, 0.0, 0.0)
        self.rotvec = (0.0, 0.0, 0.0)
        return p, r


@dataclass
class _AnchorPose:
    """Robot + controller state snapshotted at the most recent RESET.

    Per-tick wrist targets are derived as (anchor_wrist + rotation_delta) where
    rotation_delta is the absolute current-vs-anchor controller quaternion. Position
    is still delta-integrated into `_target_T` (because XLeVR sends per-frame
    position deltas, not absolute positions)."""
    ee_x: float = 0.0
    ee_y: float = 0.0
    pan_deg: float = 0.0
    wrist_flex_deg: float = 0.0
    wrist_roll_deg: float = 0.0
    gripper_pct: float = 50.0
    captured: bool = False
    # Controller orientation in VR world frame at RESET (quaternion x,y,z,w).
    # Used to compute absolute wrist mapping with zero drift across the session.
    ctrl_quat: Optional[tuple[float, float, float, float]] = None


@dataclass
class _LiveTargets:
    """Current commanded joint targets, in degrees. UI reads this."""
    shoulder_pan: float = 0.0
    shoulder_lift: float = 0.0
    elbow_flex: float = 0.0
    wrist_flex: float = 0.0
    wrist_roll: float = 0.0
    gripper: float = 100.0   # open by default; trigger held → closes

    def to_dict_with_prefix(self, side: ArmSide) -> dict[str, float]:
        prefix = f"{side}_arm_"
        return {
            f"{prefix}shoulder_pan":  self.shoulder_pan,
            f"{prefix}shoulder_lift": self.shoulder_lift,
            f"{prefix}elbow_flex":    self.elbow_flex,
            f"{prefix}wrist_flex":    self.wrist_flex,
            f"{prefix}wrist_roll":    self.wrist_roll,
            f"{prefix}gripper":       self.gripper,
        }


# --- minimal cwd-free HTTPS server for the web-ui ------------------------------

class _StaticHTTPSServer:
    """Serves the XLeVR web-ui's static files over HTTPS without cwd hacks.

    The upstream SimpleHTTPSServer at XLerobot_xuweiwu/XLeVR/vr_monitor.py
    calls `context.load_cert_chain('cert.pem', 'key.pem')` with relative paths
    and serves files relative to `os.chdir(XLEVR_PATH)`. Both of those would
    break Flask's working directory. We rebuild a minimal version that takes
    absolute paths.

    Also: the upstream `vr_app.js` has the WebSocket port (8442) hardcoded.
    If the user moves the WSS server to a different port (e.g. to dodge a
    router-level block on 8443/8442), we transparently rewrite the JS at
    serve time so the Quest browser connects to the right port.
    """

    def __init__(self, host: str, port: int, web_root: pathlib.Path,
                 cert: pathlib.Path, key: pathlib.Path,
                 ws_port: int):
        self.host = host
        self.port = port
        self.web_root = web_root.resolve()
        self.cert = cert.resolve()
        self.key = key.resolve()
        self.ws_port = ws_port
        self._httpd: Optional[http.server.HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        web_root = self.web_root
        ws_port = self.ws_port

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_): pass

            def end_headers(self):
                self.send_header("Access-Control-Allow-Origin", "*")
                try: super().end_headers()
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError): pass

            def do_OPTIONS(self):
                self.send_response(200); self.end_headers()

            def _serve(self, relpath: str, content_type: str):
                path = (web_root / relpath).resolve()
                # Disallow escape from web_root.
                try:
                    path.relative_to(web_root)
                except ValueError:
                    self.send_error(403); return
                if not path.is_file():
                    self.send_error(404); return
                try:
                    data = path.read_bytes()
                except OSError:
                    self.send_error(500); return
                # Rewrite the hardcoded WebSocket port in vr_app.js if the user
                # moved the WSS server (e.g. to dodge a router-level port block).
                if relpath.endswith("vr_app.js") and ws_port != 8442:
                    import re
                    text = data.decode("utf-8", errors="replace")
                    text = re.sub(
                        r"(const\s+websocketPort\s*=\s*)\d+\s*;",
                        f"\\g<1>{ws_port};",
                        text,
                    )
                    data = text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try: self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, ssl.SSLError): pass

            def do_GET(self):
                p = self.path.split("?", 1)[0]
                if p in ("/", "/index.html"): return self._serve("index.html", "text/html")
                if p.endswith(".css"):  return self._serve(p.lstrip("/"), "text/css")
                if p.endswith(".js"):   return self._serve(p.lstrip("/"), "application/javascript")
                if p.endswith(".ico"):  return self._serve(p.lstrip("/"), "image/x-icon")
                if p.endswith((".jpg", ".jpeg")): return self._serve(p.lstrip("/"), "image/jpeg")
                if p.endswith(".png"):  return self._serve(p.lstrip("/"), "image/png")
                if p.endswith(".gif"):  return self._serve(p.lstrip("/"), "image/gif")
                self.send_error(404)

        self._httpd = http.server.HTTPServer((self.host, self.port), Handler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(self.cert), str(self.key))
        self._httpd.socket = ctx.wrap_socket(self._httpd.socket, server_side=True)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True, name="vr-https"
        )
        self._thread.start()
        log.info("VR HTTPS server listening on https://%s:%d (web_root=%s)",
                 self.host, self.port, self.web_root)

    def stop(self) -> None:
        if self._httpd is not None:
            try: self._httpd.shutdown()
            except Exception: pass
            try: self._httpd.server_close()
            except Exception: pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3)
        self._httpd = None
        self._thread = None


# --- the session ---------------------------------------------------------------

@dataclass
class _PerArm:
    """All per-arm runtime state for VR teleop. One instance per side, always
    created — populated when the arm is connected/calibrated, otherwise idle.

    Lives inside `VRTeleopSession._arms`. The drive loop iterates over connected
    arms and only acts on the one that is `_active_arm`.
    """
    side: ArmSide
    calibrated: bool = False
    # Clamped IK target (4×4 homogeneous) — passed to the analytical IK each tick.
    # Derived as `anchor_ee_pos + offset_robot`, clamped to EE_BOUNDS + workspace
    # radius. NOT the integrator; see `offset_robot` below.
    target_T: _np.ndarray = field(default_factory=lambda: _np.eye(4))
    # Anchor EE position in robot base frame, captured at RESET via analytical FK.
    anchor_ee_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Unclamped cumulative offset from `anchor_ee_pos` in robot base frame,
    # integrated from VR position deltas. Crucially this is NOT clamped — so the
    # full controller motion is remembered even when the EE target hits a
    # workspace bound. When the hand returns to the anchor pose, `offset_robot`
    # returns to zero and the EE returns to anchor (no hysteresis).
    offset_robot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    anchor: _AnchorPose = field(default_factory=_AnchorPose)
    targets: _LiveTargets = field(default_factory=_LiveTargets)
    last_sent_targets: dict[str, float] = field(default_factory=dict)
    # Per-session VR→robot rotation matrix re-derived at every RESET from the
    # controller's orientation at grip-press. Defaults to the fixed _VR_TO_ROBOT
    # until calibration.
    session_vr_to_robot: _np.ndarray = field(default_factory=lambda: _VR_TO_ROBOT.copy())
    latest: _LatestGoal = field(default_factory=_LatestGoal)
    delta: _DeltaAccumulator = field(default_factory=_DeltaAccumulator)
    reset_pending: bool = False
    # Last-tick button state for edge detection (A/B/X/Y face buttons).
    prev_buttons: dict[str, bool] = field(default_factory=dict)
    # Guided-calibration wizard state. See `_advance_calibration`:
    #   idle → awaiting_anchor_fwd → motioning_fwd → awaiting_anchor_up →
    #   motioning_up → idle (matrix applied)
    cal_state: str = "idle"
    cal_motion_acc: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cal_captured_fwd:  Optional[tuple[float, float, float]] = None
    cal_captured_up:   Optional[tuple[float, float, float]] = None
    cal_captured_left: Optional[tuple[float, float, float]] = None
    # Last completion-time motion magnitudes (m) — for UI to show "calibrated to N cm"
    cal_last_fwd_m:  float = 0.0
    cal_last_up_m:   float = 0.0
    cal_last_left_m: float = 0.0
    # Homing state. While True, the drive loop drives this arm toward
    # `home_target` (a per-joint absolute target, in degrees), instead of the
    # VR-driven target. Cleared automatically when all joints reach their target
    # within HOMING_TOL_DEG.
    homing: bool = False
    home_target: dict[str, float] = field(default_factory=dict)
    home_start_t: float = 0.0   # monotonic seconds when homing began (timeout safety)
    # User-facing knob: when True, mirror the LATERAL axis (left/right). Flips
    # shoulder_pan direction, wrist_roll, and wrist_flex direction all at once
    # (they all derive from the y-axis mapping). Read from config/xlerobot.yaml's
    # `vr:` block per arm.
    invert_lateral: bool = False
    # When True, the YAML setting is EXPLICITLY set by the user (override mode):
    # the calibration wizard's auto-detection at step 3 must not touch
    # `invert_lateral`. Lets users with physically mirror-mounted motors keep
    # their fix in place across recalibrations.
    invert_lateral_override: bool = False
    # Per-arm URDF kinematics + last-good IK solution. Built lazily on first
    # RESET (see _ensure_kinematics). The IK uses `last_q_sol` as the initial
    # guess on every subsequent tick — this is the key trick that kills
    # null-space jitter on the 5-DOF arm (vs using noisy current joints).
    kinematics: Any = None
    last_q_sol: _np.ndarray = field(default_factory=lambda: _np.zeros(5, dtype=float))
    # Per-arm filtered target state. Position uses controller-delta EMA before
    # integration; orientation uses SLERP EMA on the actual IK target; joints use
    # EMA after IK to damp solver noise before motor rate caps.
    pos_ema: tuple[float, float, float] = (0.0, 0.0, 0.0)
    smoothed_R_target: _np.ndarray = field(default_factory=lambda: _np.eye(3))
    last_q_filtered: Optional[_np.ndarray] = None
    # Anchor orientation matrix (3×3) captured at RESET. Combined with the
    # current controller quaternion, gives the absolute desired EE orientation.
    anchor_R_robot: _np.ndarray = field(default_factory=lambda: _np.eye(3))
    # Calibration confidence: "good" if the wizard's captured motion vectors
    # were well-separated, "poor" if too parallel (and the matrix is shaky).
    cal_confidence: str = "good"


class VRTeleopSession:
    def __init__(self):
        self._lock = threading.RLock()

        # Per-arm state — always created for both sides; populated when connected.
        self._arms: dict[ArmSide, _PerArm] = {
            "left":  _PerArm(side="left"),
            "right": _PerArm(side="right"),
        }
        # The arm that VR is currently driving. Only ONE can be active at a time
        # (engage-gated bimanual). None = no arm engaged.
        self._active_arm: Optional[ArmSide] = None

        # VR pipeline (process-global, persists across motor reconnects).
        self._https: Optional[_StaticHTTPSServer] = None
        self._ws_server = None
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
        self._asyncio_thread: Optional[threading.Thread] = None

        # Drive loop (process-global).
        self._drive_thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        # Global teleop state
        self._engaged = False
        # 0.5 = half hand-to-EE mapping (30 cm/s peak). Conservative default
        # because the SO-101's ~45 cm reach is small and 1:1 mapping can feel
        # fast/jerky. Slider goes 0.1..1.0; users bump up for fast tasks.
        self._scale = 0.5
        self._gripper_open = DEFAULT_GRIPPER_OPEN
        self._gripper_closed = DEFAULT_GRIPPER_CLOSED
        self._last_drive_tick: float = 0.0
        self._last_error: Optional[str] = None
        # Recording state. B button on right controller OR UI toggle flip this.
        # `_recorder` is lazily created on first start (so a session that never
        # records pays no dataset cost).
        self._recording: bool = False
        self._recorder: Optional[_dataset.DatasetRecorder] = None
        # Last task string passed via the UI. Cached here so that when the user
        # presses B on the Quest controller (no UI typing possible), the
        # backend uses the most-recently-typed task instead of an empty string.
        self._last_task: str = ""
        # Resolved (absolute, ~-expanded) dataset storage root from most recent
        # recorder init. Shown on the UI's Recording card.
        self._last_dataset_root: str = ""

        # Kinematics — analytical only; URDF IK has null-space jitter on the
        # redundant 5-DOF arm. URDF FK was used at RESET; analytical FK is now
        # used there too for round-trip exactness with the analytical IK.
        self._analytical_kin = _SO101Kin()

        # Restore previously-saved VR calibrations from config/vr_calibration.yaml
        # so the user doesn't have to re-run the wizard every session. New
        # calibrations overwrite the file via `_finalize_calibration`.
        self._load_persisted_calibrations()

    def _load_persisted_calibrations(self) -> None:
        """Restore per-arm session_vr_to_robot from disk. Silent no-op if no
        file exists or the file is malformed. Also reads per-arm invert_lateral
        flags from config/xlerobot.yaml's `vr:` section, both the value AND
        whether it's explicitly set (override mode). Also reads the global
        smoothing and rate-limit factors."""
        import yaml
        global KP, EE_DELTA_LIMIT_M, WRIST_RAD_DELTA_LIMIT
        global POS_EMA_ALPHA, ORI_EMA_ALPHA, JOINT_EMA_ALPHA
        try:
            cfg = yaml.safe_load((REPO_ROOT / "config" / "xlerobot.yaml").read_text()) or {}
            vr_section = cfg.get("vr") or {}
            def _float_key(name: str, default: float, lo: float, hi: float) -> float:
                value = vr_section.get(name)
                if value is None:
                    return default
                return max(lo, min(hi, float(value)))

            KP = _float_key("kp", KP, 0.0, 1.0)
            EE_DELTA_LIMIT_M = _float_key("ee_delta_limit_m", EE_DELTA_LIMIT_M, 0.001, 0.05)
            wrist_deg = _float_key("wrist_delta_limit_deg", math.degrees(WRIST_RAD_DELTA_LIMIT), 1.0, 30.0)
            WRIST_RAD_DELTA_LIMIT = math.radians(wrist_deg)
            POS_EMA_ALPHA = _float_key("pos_ema_alpha", POS_EMA_ALPHA, 0.0, 1.0)
            # Backward-compatible alias: older configs used rotvec_ema_alpha.
            ori_alpha = vr_section.get("ori_ema_alpha", vr_section.get("rotvec_ema_alpha"))
            if ori_alpha is not None:
                ORI_EMA_ALPHA = max(0.0, min(1.0, float(ori_alpha)))
            JOINT_EMA_ALPHA = _float_key("joint_ema_alpha", JOINT_EMA_ALPHA, 0.0, 1.0)
            joint_caps = vr_section.get("joint_deg_caps") or {}
            if isinstance(joint_caps, dict):
                for joint, cap in joint_caps.items():
                    if joint in PER_TICK_DEG_CAPS:
                        PER_TICK_DEG_CAPS[joint] = max(0.1, min(30.0, float(cap)))
            log.info(
                "VR smoothing loaded: kp=%.2f pos_ema=%.2f ori_ema=%.2f joint_ema=%.2f "
                "ee_cap=%.3fm wrist_cap=%.1f°",
                KP, POS_EMA_ALPHA, ORI_EMA_ALPHA, JOINT_EMA_ALPHA,
                EE_DELTA_LIMIT_M, math.degrees(WRIST_RAD_DELTA_LIMIT),
            )
        except Exception as e:
            log.warning("could not read VR smoothing config from YAML: %s", e)
        for side in ("left", "right"):
            self._restore_persisted_arm_config(side)

    def _restore_persisted_arm_config(self, side: ArmSide) -> None:
        """Restore saved calibration and lateral mapping for a freshly reset arm."""
        invert_flags = _vrcal.read_invert_lateral_flags()
        overrides = _vrcal.read_invert_lateral_overrides()
        arm = self._arms[side]
        arm.invert_lateral = invert_flags.get(side, False)
        arm.invert_lateral_override = overrides.get(side, False)
        M = _vrcal.matrix_for_arm(side)
        if M is None:
            return
        arm.session_vr_to_robot = M
        data = _vrcal.read_for_arm(side) or {}
        arm.cal_last_fwd_m = float(data.get("forward_motion_m", 0.0))
        arm.cal_last_up_m = float(data.get("up_motion_m", 0.0))
        log.info(
            "[%s] restored saved VR calibration (invert_lateral=%s, override=%s)",
            side, arm.invert_lateral, arm.invert_lateral_override,
        )

    # ── public API ────────────────────────────────────────────────────────────
    @property
    def any_connected(self) -> bool:
        return MOTORS.any_connected

    @property
    def connected_sides(self) -> list[ArmSide]:
        return MOTORS.connected_sides

    @property
    def active_arm(self) -> Optional[ArmSide]:
        return self._active_arm

    def connect(self, side: ArmSide) -> dict:
        """Connect ONE arm. The other arm (if connected) stays untouched."""
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        with self._lock:
            if MOTORS.is_connected(side):
                return self.status()
            # Reset per-arm state for this side.
            self._arms[side] = _PerArm(side=side)
            self._last_error = None
            self._load_gripper_config()
            try:
                MOTORS.connect(side)
                self._seed_targets_from_present(side)
                # VR pipeline + drive loop are started ONCE and persist across
                # motor connect/disconnect cycles so the Quest browser's WS stays
                # connected when switching arms.
                if self._https is None:
                    self._stop_evt.clear()
                    self._start_vr_pipeline()
                if self._drive_thread is None or not self._drive_thread.is_alive():
                    self._stop_evt.clear()
                    self._start_drive_loop()
            except Exception as e:
                self._last_error = f"{type(e).__name__}: {e}"
                log.exception("VR session connect failed for %s", side)
                try: MOTORS.disconnect(side)
                except Exception: pass
                raise
            return self.status()

    def disconnect(self, side: Optional[ArmSide] = None) -> dict:
        """Disconnect ONE arm, or both if side is None. KEEPS the VR pipeline
        running so the Quest browser stays connected. Use `emergency_stop()` to
        also tear down the VR servers."""
        with self._lock:
            sides = list(MOTORS.connected_sides) if side is None else [side]
            for s in sides:
                # Reset per-arm state on disconnect.
                self._arms[s] = _PerArm(side=s)
                if self._active_arm == s:
                    self._active_arm = None
                    self._engaged = False
                try: MOTORS.disconnect(s)
                except Exception as e:
                    self._last_error = f"disconnect {s}: {e}"
                    log.warning("disconnect %s: %s", s, e)
            return self.status()

    def emergency_stop(self) -> dict:
        """Release torque on every connected arm immediately and tear down the
        VR servers. Flush any in-flight dataset episode. No motion. No homing."""
        with self._lock:
            self._engaged = False
            self._active_arm = None
            was_recording = self._recording
            self._recording = False
            rec = self._recorder
            self._stop_evt.set()
            try:
                MOTORS.emergency_release_torque()
            except Exception as e:
                self._last_error = f"e-stop: {e}"
                log.warning("e-stop: %s", e)
            self._stop_threads_and_servers()
            for s in ("left", "right"):
                self._arms[s] = _PerArm(side=s)
        # Flush the recorder OUTSIDE the lock — finalize may encode video.
        if was_recording and rec is not None:
            try: rec.end_episode()
            except Exception as e: log.warning("e-stop: end_episode: %s", e)
        if rec is not None:
            try: rec.finalize()
            except Exception as e: log.warning("e-stop: finalize: %s", e)
        with self._lock:
            self._recorder = None
        return self.status()

    def _load_gripper_config(self) -> None:
        """Read gripper.open_value / gripper.closed_value from config/xlerobot.yaml.
        Some SO101 calibrations have 0=open and 100=closed; others have the reverse.
        Lets the user flip the convention without touching code."""
        import yaml
        try:
            cfg = yaml.safe_load((REPO_ROOT / "config" / "xlerobot.yaml").read_text()) or {}
            g = cfg.get("gripper") or {}
            self._gripper_open = float(g.get("open_value", DEFAULT_GRIPPER_OPEN))
            self._gripper_closed = float(g.get("closed_value", DEFAULT_GRIPPER_CLOSED))
        except Exception as e:
            log.warning("could not read gripper config: %s", e)
            self._gripper_open = DEFAULT_GRIPPER_OPEN
            self._gripper_closed = DEFAULT_GRIPPER_CLOSED
        log.info("gripper config: open=%s closed=%s", self._gripper_open, self._gripper_closed)

    def engage(self, engaged: bool, scale: Optional[float] = None,
               active_arm: Optional[ArmSide] = None) -> dict:
        """Set the global engage state.

        - If `active_arm` is provided and `engaged=True`, that arm becomes the
          one that VR drives. The arm must be connected first.
        - If `active_arm` is omitted and `engaged=True`, the system picks the
          arm: if exactly one is connected, that one; if both, leaves the
          previous `_active_arm` if still valid; otherwise raises.
        - `engaged=False` clears the active arm.
        """
        with self._lock:
            if scale is not None:
                self._scale = max(0.1, min(1.0, float(scale)))
            if engaged:
                if not MOTORS.any_connected:
                    raise RuntimeError("connect an arm before engaging")
                if active_arm is not None:
                    if active_arm not in ("left", "right"):
                        raise ValueError(
                            f"active_arm must be 'left' or 'right', got {active_arm!r}"
                        )
                    if not MOTORS.is_connected(active_arm):
                        raise RuntimeError(f"{active_arm} arm not connected")
                    self._active_arm = active_arm
                elif self._active_arm is None or not MOTORS.is_connected(self._active_arm):
                    connected = MOTORS.connected_sides
                    if len(connected) == 1:
                        self._active_arm = connected[0]
                    else:
                        raise RuntimeError(
                            "both arms are connected; pass active_arm=left|right "
                            "to choose which arm to engage"
                        )
                arm = self._arms[self._active_arm]
                if not arm.calibrated:
                    log.info(
                        "engage on %s but vr_calibrated=False — motors stay still"
                        " until the Quest controller's RESET (grip) is pressed",
                        self._active_arm,
                    )
            else:
                self._active_arm = None
            self._engaged = bool(engaged)
            return self.status()

    def status(self) -> dict:
        now = time.time()
        with self._lock:
            # Single bus read for both arms (each arm only has its own joints in
            # the result; MOTORS.read_positions(None) merges connected sides).
            joint_present: dict[str, float] = {}
            try:
                joint_present = MOTORS.read_positions()
            except Exception as e:
                self._last_error = f"read: {e}"

            arms_status: dict[str, Any] = {}
            for s in ("left", "right"):
                arm = self._arms[s]
                arms_status[s] = {
                    "connected": MOTORS.is_connected(s),
                    "torque_enabled": MOTORS.is_torque_enabled(s),
                    "calibrated": arm.calibrated,
                    "joint_target": arm.targets.to_dict_with_prefix(s)
                                     if MOTORS.is_connected(s) else {},
                    "controller": {
                        "position": list(arm.latest.rel_position),
                        "rotation": (list(arm.latest.rotation_quat)
                                     if arm.latest.rotation_quat else None),
                        "trigger": arm.latest.trigger,
                        "thumbstick": {"x": arm.latest.thumbstick[0],
                                       "y": arm.latest.thumbstick[1]},
                        "age_ms": (int(1000 * (now - arm.latest.received_at))
                                    if arm.latest.has_data else None),
                        "mode": arm.latest.mode,
                    },
                    # Calibration diagnostics. After a RESET (grip-press), these
                    # let the user see the mapping in action:
                    #   anchor_ee_pos    = robot EE at the moment of grip-press
                    #   offset_robot     = cumulative offset since RESET (unclamped)
                    #   target_ee_pos    = anchor + offset, clamped to workspace
                    #   session_yaw_deg  = the yaw the user's "forward" was at RESET
                    "calibration": {
                        "anchor_ee_pos": list(arm.anchor_ee_pos),
                        "offset_robot": list(arm.offset_robot),
                        "target_ee_pos": [float(arm.target_T[0, 3]),
                                           float(arm.target_T[1, 3]),
                                           float(arm.target_T[2, 3])],
                        # Yaw of the user's "forward" relative to VR-world default
                        # (default = controller pointing -Z). 0° = facing default;
                        # +N° = turned N° to the right; -N° = turned to the left.
                        "session_yaw_deg": float(math.degrees(math.atan2(
                            -arm.session_vr_to_robot[0, 0],
                            -arm.session_vr_to_robot[0, 2],
                        ))),
                        # Guided-calibration wizard state.
                        "wizard_state": arm.cal_state,
                        "wizard_motion_m": math.sqrt(
                            arm.cal_motion_acc[0]**2 +
                            arm.cal_motion_acc[1]**2 +
                            arm.cal_motion_acc[2]**2
                        ),
                        "wizard_target_m": CALIBRATION_TARGET_MOTION_M,
                        "wizard_min_m": CALIBRATION_MIN_MOTION_M,
                        "wizard_last_fwd_m":  arm.cal_last_fwd_m,
                        "wizard_last_up_m":   arm.cal_last_up_m,
                        "wizard_last_left_m": arm.cal_last_left_m,
                        "wizard_fwd_captured":  arm.cal_captured_fwd  is not None,
                        "wizard_up_captured":   arm.cal_captured_up   is not None,
                        "wizard_left_captured": arm.cal_captured_left is not None,
                        "invert_lateral":       arm.invert_lateral,
                        "confidence":           arm.cal_confidence,
                    },
                }

            rec = self._recorder
            # If no recorder yet, compute what root WOULD be used (so the UI's
            # placeholder shows the actual default before first Start).
            if self._last_dataset_root:
                shown_root = self._last_dataset_root
            else:
                try:
                    cfg_now = _dataset.load_dataset_config()
                    shown_root = _dataset.resolve_root(
                        cfg_now.get("root"), str(cfg_now["repo_id"]),
                    )
                except Exception:
                    shown_root = ""
            recording_info = {
                "active": self._recording,
                "episodes_saved": rec.episode_count if rec else 0,
                "frames_in_current_episode": rec.frame_count_in_episode if rec else 0,
                "repo_id": rec.repo_id if rec else None,
                "last_task": self._last_task,
                "root": shown_root,
            }
            # Per-arm home pose status (from YAML + live homing flag).
            try:
                hp_status = _home.home_pose_status()
            except Exception as e:
                log.warning("home_pose_status failed: %s", e)
                hp_status = {"left": {"captured": False, "joints": {}},
                             "right": {"captured": False, "joints": {}}}
            for s in ("left", "right"):
                hp_status[s]["homing"] = self._arms[s].homing
                arms_status[s]["home"] = hp_status[s]

            # Per-arm persisted VR calibration status (config/vr_calibration.yaml).
            try:
                vr_cal_status = _vrcal.status()
            except Exception as e:
                log.warning("vr_calibration.status failed: %s", e)
                vr_cal_status = {"left": {"saved": False}, "right": {"saved": False}}
            for s in ("left", "right"):
                arms_status[s]["calibration"]["persisted"] = vr_cal_status[s]

            out = {
                "arms": arms_status,
                "connected_sides": list(MOTORS.connected_sides),
                "active_arm": self._active_arm,
                "engaged": self._engaged,
                "scale": self._scale,
                "recording": self._recording,
                "recording_info": recording_info,
                "last_tick_age_ms": (int(1000 * (now - self._last_drive_tick))
                                       if self._last_drive_tick else None),
                "last_error": self._last_error,
                "joint_present": joint_present,
                "joint_bounds": {j: list(MOTORS.bounds[j]) for j in MOTORS.bounds},
                "vr_endpoint": self._vr_endpoint_url(),
            }
            # Back-compat: surface the active (or only-connected) arm's data
            # under the old top-level keys so the existing frontend keeps working
            # until it switches to the per-arm `arms` view.
            legacy_side = self._active_arm or (
                MOTORS.connected_sides[0] if MOTORS.connected_sides else None
            )
            if legacy_side is not None:
                a = self._arms[legacy_side]
                out["arm"] = legacy_side
                out["connected"] = MOTORS.is_connected(legacy_side)
                out["vr_calibrated"] = a.calibrated
                out["joint_target"] = a.targets.to_dict_with_prefix(legacy_side)
                out["controller"] = arms_status[legacy_side]["controller"]
                out["last_goal_age_ms"] = (
                    int(1000 * (now - a.latest.received_at))
                    if a.latest.has_data else None
                )
            else:
                out["arm"] = None
                out["connected"] = False
                out["vr_calibrated"] = False
                out["joint_target"] = {}
                out["controller"] = {
                    "position": [0.0, 0.0, 0.0], "rotation": None,
                    "trigger": False, "thumbstick": {"x": 0.0, "y": 0.0},
                    "age_ms": None, "mode": "idle",
                }
                out["last_goal_age_ms"] = None
            return out

    # ── VR pipeline (HTTPS + WSS in an asyncio thread) ──────────────────────
    def _vr_endpoint_url(self) -> Optional[str]:
        if self._https is None:
            return None
        host = self._local_ip() if self._https.host == "0.0.0.0" else self._https.host
        return f"https://{host}:{self._https.port}"

    def _ensure_cert_matches_lan_ip(self, cert: pathlib.Path, key: pathlib.Path) -> None:
        """If `cert` exists and already covers our current LAN IP in subjectAltName,
        leave it alone (preserves the cert fingerprint the user already accepted on
        the Quest). Otherwise (cert missing, or SAN mismatch), generate a fresh pair
        with the current LAN IP baked in."""
        ip = self._local_ip()
        if cert.is_file() and key.is_file() and self._cert_has_ip(cert, ip):
            log.info("VR cert at %s already covers LAN IP %s; reusing", cert, ip)
            return

        log.info("regenerating VR cert with CN=%s (was %s)", ip,
                 "missing" if not cert.is_file() else "stale SAN")
        if shutil_which := getattr(__import__("shutil"), "which"):
            if shutil_which("openssl") is None:
                raise RuntimeError("openssl binary not found in PATH")
        cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-days", "365",
            "-subj", f"/CN={ip}",
            "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost",
            "-keyout", str(key),
            "-out", str(cert),
        ]
        # Write atomically: openssl will overwrite the existing files in place,
        # which is fine because we hold no open handles to them right now.
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        if result.returncode != 0:
            raise RuntimeError(
                f"openssl failed (rc={result.returncode}): {result.stderr.decode(errors='replace')}"
            )
        # Verify the new cert actually contains the IP.
        if not self._cert_has_ip(cert, ip):
            raise RuntimeError(f"new cert at {cert} still doesn't contain IP {ip}")

    @staticmethod
    def _cert_has_ip(cert_path: pathlib.Path, ip: str) -> bool:
        """Return True iff `cert_path` has `ip` listed in its subjectAltName."""
        try:
            out = subprocess.check_output(
                ["openssl", "x509", "-in", str(cert_path), "-noout", "-text"],
                timeout=5,
            ).decode(errors="replace")
        except Exception:
            return False
        # X509v3 SAN section will contain a line like `IP Address:192.168.0.113`.
        return f"IP Address:{ip}" in out

    @staticmethod
    def _local_ip() -> str:
        """Pick the most likely LAN IP for the Quest to connect to.

        Order of preference:
          1. 192.168.x.x  (typical home LAN)
          2. 10.x.x.x     (corporate LAN / second-tier)
          3. 172.16-31.x  (also private; exclude Docker bridges 172.17-19)
          4. Whatever the default-route trick returns

        Filtered out:
          - 127.x.x.x (loopback)
          - 169.254.x.x (link-local — won't be reachable)
          - 100.64-127.x.x (Tailscale CGNAT range)
          - 172.17-19.x.x (default Docker bridges)
        """
        def _classify(ip: str) -> int:
            try:
                a, b, *_ = (int(p) for p in ip.split("."))
            except ValueError:
                return 99
            if ip.startswith("127.") or ip.startswith("169.254."):
                return 99
            if a == 100 and 64 <= b <= 127:                 # Tailscale
                return 90
            if a == 172 and 17 <= b <= 19:                  # Docker bridges
                return 90
            if ip.startswith("192.168."):
                return 0
            if ip.startswith("10."):
                return 1
            if a == 172 and 16 <= b <= 31:                  # other private 172.x
                return 2
            return 50  # public IPs and everything else

        # Collect every IPv4 we can find via hostname -I.
        candidates: list[str] = []
        try:
            out = subprocess.check_output(["hostname", "-I"], timeout=2).decode().strip()
            candidates = [ip for ip in out.split() if "." in ip]
        except Exception:
            pass

        # Add the default-route IP too (in case `hostname -I` is missing on the host).
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                if ip not in candidates:
                    candidates.append(ip)
        except OSError:
            pass

        if not candidates:
            return "localhost"
        candidates.sort(key=_classify)
        return candidates[0] if _classify(candidates[0]) < 90 else candidates[0]

    def _start_vr_pipeline(self) -> None:
        # Late import: xlevr only resolves after sys.path was patched at module load.
        from xlevr.config import XLeVRConfig
        from xlevr.inputs.vr_ws_server import VRWebSocketServer

        cfg = XLeVRConfig()
        cfg.enable_vr = True
        cfg.enable_keyboard = False
        cfg.enable_https = True

        # Honour user overrides from config/xlerobot.yaml's `vr:` section. Some ISP
        # routers block 8443/8442 specifically (alt-HTTPS port heuristic), so the
        # user can move both to e.g. 5443/5442 if 5000 reaches them.
        import yaml
        try:
            yaml_cfg = yaml.safe_load(
                (REPO_ROOT / "config" / "xlerobot.yaml").read_text()
            ) or {}
            vr_section = yaml_cfg.get("vr") or {}
        except Exception:
            vr_section = {}
        cfg.host_ip = str(vr_section.get("host_ip", getattr(cfg, "host_ip", "0.0.0.0")))
        cfg.https_port = int(vr_section.get("https_port", getattr(cfg, "https_port", 8443)))
        cfg.websocket_port = int(vr_section.get("websocket_port",
                                                getattr(cfg, "websocket_port", 8442)))
        log.info("VR using https_port=%s websocket_port=%s host=%s",
                 cfg.https_port, cfg.websocket_port, cfg.host_ip)

        # Resolve our absolute paths (the upstream relies on cwd; we won't touch it).
        # IMPORTANT: the static HTML/JS/CSS the Quest browser fetches live under
        # XLeVR/web-ui/, not under XLeVR/ itself. The upstream's handler prepends
        # "web-ui/" to every request path; our _StaticHTTPSServer treats web_root as
        # the actual static-asset root, so point it at the right subdirectory.
        web_root = XLEVR_DIR / "web-ui"
        cert = XLEVR_DIR / "cert.pem"
        key = XLEVR_DIR / "key.pem"

        # Auto-regenerate the cert if it doesn't list this workstation's current LAN IP
        # in its subjectAltName. Without this, Meta Browser silently refuses the TLS
        # handshake when the URL's IP doesn't appear in the cert (no "Proceed unsafe"
        # button is shown in that case).
        try:
            self._ensure_cert_matches_lan_ip(cert, key)
        except Exception as e:
            raise RuntimeError(
                f"could not prepare HTTPS cert at {cert}: {e}\n"
                f"You can regenerate manually:\n"
                f"  IP=$(hostname -I | awk '{{print $1}}')\n"
                f"  openssl req -x509 -newkey rsa:2048 -nodes -days 365 \\\n"
                f"    -subj \"/CN=$IP\" -addext \"subjectAltName=IP:$IP,IP:127.0.0.1,DNS:localhost\" \\\n"
                f"    -keyout {key} -out {cert}"
            ) from e

        # HTTPS server (serves the web-ui static assets). Pass the configured WSS
        # port so the on-the-fly rewrite of vr_app.js makes the Quest connect to
        # the right WSS endpoint.
        self._https = _StaticHTTPSServer(
            host=cfg.host_ip,
            port=cfg.https_port,
            web_root=web_root, cert=cert, key=key,
            ws_port=cfg.websocket_port,
        )
        self._https.start()

        # WSS server (receives VR controller pose messages) runs in an asyncio loop
        # on its own thread. The xuweiwu VRWebSocketServer needs an asyncio.Queue.
        ready = threading.Event()
        thread_err: dict[str, Any] = {}

        def _run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._asyncio_loop = loop

                queue: asyncio.Queue = asyncio.Queue()
                self._ws_server = VRWebSocketServer(
                    command_queue=queue, config=cfg, print_only=False
                )
                loop.create_task(self._ws_server.start())
                loop.create_task(self._drain_goals(queue))
                ready.set()
                loop.run_forever()
            except Exception as e:
                thread_err["e"] = e
                ready.set()

        self._asyncio_thread = threading.Thread(target=_run, daemon=True, name="vr-wss")
        self._asyncio_thread.start()
        ready.wait(timeout=8)
        if "e" in thread_err:
            self._https.stop(); self._https = None
            raise RuntimeError(f"VR WebSocket server failed to start: {thread_err['e']}")

    async def _drain_goals(self, queue: "asyncio.Queue") -> None:
        """Consume ControlGoals from the WSS server and route each to the matching
        per-arm `_PerArm`. Headset goals are ignored. Goals for an arm that isn't
        currently connected are still accepted into _PerArm state — that way, if
        the user squeezes grip BEFORE connecting that arm in the UI, the latest
        goal is already there when they do connect."""
        try:
            from xlevr.inputs.base import ControlMode  # noqa: F401
        except Exception:
            pass
        while not self._stop_evt.is_set():
            try:
                goal = await asyncio.wait_for(queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            try:
                side = getattr(goal, "arm", None)
                if side not in ("left", "right"):
                    continue  # headset / unknown
                arm = self._arms[side]
                mode_obj = getattr(goal, "mode", None)
                mode = getattr(mode_obj, "value", mode_obj) or "idle"
                rp = getattr(goal, "relative_position", None)
                rr = getattr(goal, "relative_rotvec", None)
                rot = getattr(goal, "vr_ctrl_rotation", None)
                trig = bool(getattr(goal, "trigger", False))
                thumb = getattr(goal, "thumbstick", None) or {}
                btn  = getattr(goal, "buttons", None) or {}
                with self._lock:
                    # Latch reset edges so a fast-following 'position' goal can't
                    # overwrite the calibration trigger before the drive loop ticks.
                    if str(mode) == "reset":
                        arm.reset_pending = True
                    # Accumulate per-frame deltas only on 'position' goals.
                    if str(mode) == "position":
                        rp_t = (float(rp[0]), float(rp[1]), float(rp[2])) if rp is not None else (0.0, 0.0, 0.0)
                        rr_t = (float(rr[0]), float(rr[1]), float(rr[2])) if rr is not None else (0.0, 0.0, 0.0)
                        arm.delta.add(rp_t, rr_t)
                    arm.latest = _LatestGoal(
                        received_at=time.time(),
                        has_data=True,
                        mode=str(mode),
                        rel_position=tuple(float(v) for v in (rp if rp is not None else (0, 0, 0))),
                        rel_rotvec=tuple(float(v) for v in (rr if rr is not None else (0, 0, 0))),
                        rotation_quat=(tuple(float(v) for v in rot.as_quat()) if rot is not None and hasattr(rot, "as_quat") else None),
                        trigger=trig,
                        thumbstick=(float(thumb.get("x", 0)), float(thumb.get("y", 0))),
                        buttons={str(k): bool(v) for k, v in btn.items()},
                    )
                    # Advance the calibration state machine if active for this arm.
                    # Must hold `self._lock` (still held inside this `with` block).
                    if arm.cal_state != "idle":
                        self._advance_calibration(side, arm.latest)
            except Exception as e:
                log.warning("goal-drain: %s", e)

    # ── drive loop ──────────────────────────────────────────────────────────
    def _start_drive_loop(self) -> None:
        self._drive_thread = threading.Thread(
            target=self._drive_loop, daemon=True, name="vr-drive"
        )
        self._drive_thread.start()

    def _seed_targets_from_present(self, side: ArmSide) -> None:
        """Initialise live targets for ONE arm from its present joint positions
        so the first command doesn't try to swing toward zero."""
        if not MOTORS.is_connected(side):
            return
        pres = MOTORS.read_positions(side)
        prefix = f"{side}_arm_"
        get = lambda j: float(pres.get(f"{prefix}{j}", 0.0))
        arm = self._arms[side]
        arm.targets = _LiveTargets(
            shoulder_pan=get("shoulder_pan"),
            shoulder_lift=get("shoulder_lift"),
            elbow_flex=get("elbow_flex"),
            wrist_flex=get("wrist_flex"),
            wrist_roll=get("wrist_roll"),
            gripper=get("gripper"),
        )
        arm.last_sent_targets = {f"{prefix}{j}": getattr(arm.targets, j)
                                  for j in _motors.JOINTS_PER_ARM}

    def _ensure_kinematics(self, arm: _PerArm) -> None:
        """Lazy-build the per-arm URDF kinematics handle. Called from RESET.
        Catches the placo import + URDF load failures with a clear error."""
        if arm.kinematics is not None:
            return
        try:
            from lerobot.model import RobotKinematics
        except ImportError as e:
            raise RuntimeError(
                f"placo/RobotKinematics import failed: {e}. Install with: "
                "uv pip install lerobot[kinematics]"
            ) from e
        urdf = REPO_ROOT / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
        if not urdf.is_file():
            raise RuntimeError(f"SO101 URDF not found at {urdf}")
        arm.kinematics = RobotKinematics(
            urdf_path=str(urdf),
            target_frame_name="gripper_frame_link",
            joint_names=list(_IK_JOINT_ORDER),
        )
        log.info("[%s] URDF kinematics initialized (%s)", arm.side, urdf.name)

    def _capture_anchor(self, side: ArmSide) -> None:
        """RESET handler for ONE arm. Snapshot the EE pose via URDF FK; this
        becomes the anchor for the absolute-pose IK pipeline. Resets the target
        filters + delta accumulator; rebuilds the per-session VR→robot frame.
        """
        if not MOTORS.is_connected(side):
            return
        arm = self._arms[side]
        try:
            self._ensure_kinematics(arm)
        except Exception as e:
            log.exception("[%s] kinematics setup failed: %s", side, e)
            self._last_error = f"[{side}] IK init: {e}"
            return

        present = MOTORS.read_positions(side)
        prefix = f"{side}_arm_"
        get = lambda j: float(present.get(f"{prefix}{j}", 0.0))

        # URDF FK at current joints → 4×4 EE pose. Far more accurate than the
        # planar analytical FK (which assumes wrist is at gripper origin).
        q_now_deg = _np.array([get(j) for j in _IK_JOINT_ORDER], dtype=float)
        T_now = arm.kinematics.forward_kinematics(q_now_deg)

        arm.target_T = T_now.copy()
        arm.anchor_ee_pos = (float(T_now[0, 3]), float(T_now[1, 3]), float(T_now[2, 3]))
        arm.anchor_R_robot = T_now[:3, :3].copy()
        arm.offset_robot = (0.0, 0.0, 0.0)
        arm.pos_ema = (0.0, 0.0, 0.0)
        arm.smoothed_R_target = T_now[:3, :3].copy()
        arm.last_q_sol = q_now_deg.copy()
        arm.last_q_filtered = q_now_deg.copy()
        arm.delta.drain()

        # Build per-session VR→robot frame from controller's anchor quat. XLeVR
        # ships `vr_ctrl_rotation` on RESET goals (and now on POSITION too after
        # our submodule patch).
        ctrl_quat = arm.latest.rotation_quat if arm.latest.has_data else None
        if ctrl_quat is not None:
            arm.session_vr_to_robot = _compute_session_frame(ctrl_quat)
            log.info("[%s] session VR→robot frame calibrated from controller anchor", side)
        else:
            log.warning("[%s] no controller rotation in RESET goal; keeping previous frame",
                        side)

        arm.anchor = _AnchorPose(
            ee_x=float(T_now[0, 3]),
            ee_y=float(T_now[1, 3]),
            pan_deg=get("shoulder_pan"),
            wrist_flex_deg=get("wrist_flex"),
            wrist_roll_deg=get("wrist_roll"),
            gripper_pct=get("gripper"),
            captured=True,
            ctrl_quat=ctrl_quat,
        )
        arm.targets = _LiveTargets(
            shoulder_pan=get("shoulder_pan"),
            shoulder_lift=get("shoulder_lift"),
            elbow_flex=get("elbow_flex"),
            wrist_flex=get("wrist_flex"),
            wrist_roll=get("wrist_roll"),
            gripper=self._gripper_open,
        )
        arm.last_sent_targets = arm.targets.to_dict_with_prefix(side)
        arm.calibrated = True
        log.info("[%s] VR anchor: EE=(%.3f, %.3f, %.3f) m (URDF FK)",
                 side, T_now[0, 3], T_now[1, 3], T_now[2, 3])

    def _drive_loop(self) -> None:
        """Per-tick: process RESETs for ALL connected arms (so each arm's anchor
        is up-to-date), but only command motion on the ACTIVE arm. The non-active
        arm holds its position via its servo PID — no software command needed."""
        next_tick = time.monotonic()
        while not self._stop_evt.is_set():
            now = time.time()
            try:
                with self._lock:
                    engaged = self._engaged
                    active = self._active_arm
                    scale = self._scale
                    connected = list(MOTORS.connected_sides)
                self._last_drive_tick = now

                if not connected:
                    next_tick = self._sleep_until(next_tick)
                    continue

                # Phase 1: handle RESET / IDLE / button edges / drain accumulator
                # for EVERY connected arm. We do this regardless of which one is
                # "active" so that switching active_arm mid-session doesn't pick up
                # stale accumulator data or skip a fresh anchor.
                for side in connected:
                    arm = self._arms[side]
                    with self._lock:
                        goal = arm.latest
                        reset_now = arm.reset_pending
                        if reset_now:
                            arm.reset_pending = False
                        # Button edge detection (drive-loop rate is fast enough
                        # — ~30 Hz — to never miss a press, but slow enough that
                        # holding a button for ~16 ms doesn't toggle twice).
                        cur_btn = dict(goal.buttons or {})
                        prev_btn = arm.prev_buttons
                        arm.prev_buttons = cur_btn

                    # Note: we do NOT clear `arm.calibrated` on IDLE goals.
                    # Releasing grip just stops motion; the anchor stays valid so
                    # the UI keeps showing the captured pose. The next grip-press
                    # sends a fresh RESET goal which re-anchors via `_capture_anchor`
                    # below. Without this, the UI flipped to "not calibrated" every
                    # time the user released grip, which looked like a bug.

                    # RESET captures the anchor for THIS arm — but ONLY if we're
                    # not in the middle of a guided calibration. During calibration,
                    # the grip-press is consumed by the calibration state machine
                    # (via _advance_calibration in _drain_goals); we don't want to
                    # also anchor for teleop until calibration is done.
                    if (reset_now or (goal.has_data and goal.mode == "reset")) \
                            and arm.cal_state == "idle":
                        with self._lock:
                            self._capture_anchor(side)

                    # A/X edge → engage toggle.
                    engage_btn = ENGAGE_BUTTON_BY_SIDE.get(side)
                    if engage_btn and cur_btn.get(engage_btn) and not prev_btn.get(engage_btn):
                        self._handle_engage_button(side)
                    # B edge (right controller only) → recording toggle.
                    record_btn = RECORD_BUTTON_BY_SIDE.get(side)
                    if record_btn and cur_btn.get(record_btn) and not prev_btn.get(record_btn):
                        self._handle_record_button(side)

                    # Drain accumulator every tick — keeps from leaking deltas
                    # while waiting for engage/calibrate.
                    with self._lock:
                        arm.drained_dp_dr = arm.delta.drain()  # tuple of (pos, rotvec)

                # Phase 1.5: drive any HOMING arms toward their home targets.
                # Runs regardless of engage/active state — homing is its own mode.
                # Uses the same per-tick caps + KP smoothing as VR teleop, so
                # motion is slow and bus-safe.
                for side in connected:
                    arm = self._arms[side]
                    if not arm.homing or not arm.home_target:
                        continue
                    if not MOTORS.is_torque_enabled(side):
                        # User released torque mid-homing — abort homing.
                        with self._lock:
                            arm.homing = False
                            arm.home_target = {}
                        continue
                    prefix = f"{side}_arm_"
                    present = MOTORS.read_positions(side)
                    clamped: dict[str, float] = {}
                    # Software convergence = per-tick-clamped value == home target
                    # for every joint. This converges exactly (no PID deadband)
                    # whereas Present_Position can hover a few degrees off due to
                    # the motor's internal PID; the latter caused the UI to never
                    # clear "HOMING…" even after the arm physically arrived.
                    converged = True
                    for pj, target_deg in arm.home_target.items():
                        cap = PER_TICK_DEG_CAPS.get(pj.removeprefix(prefix), 1.0)
                        prev = arm.last_sent_targets.get(pj, present.get(pj, target_deg))
                        delta = max(-cap, min(cap, target_deg - prev))
                        clamped[pj] = prev + delta
                        if abs(clamped[pj] - target_deg) > HOMING_TOL_DEG:
                            converged = False
                    final: dict[str, float] = {}
                    for pj, target in clamped.items():
                        here = present.get(pj, target)
                        final[pj] = here + KP * (target - here)
                    try:
                        MOTORS.send_action(side, final)
                        arm.last_sent_targets = clamped
                    except Exception as e:
                        log.warning("[%s] homing send failed: %s", side, e)
                    elapsed = time.monotonic() - arm.home_start_t
                    if converged or elapsed > HOMING_TIMEOUT_S:
                        with self._lock:
                            arm.homing = False
                            arm.home_target = {}
                        if converged:
                            log.info("[%s] homing complete in %.1fs; arm at saved home_pose",
                                     side, elapsed)
                        else:
                            log.warning("[%s] homing TIMED OUT after %.1fs; "
                                        "arm may not have reached the saved pose",
                                        side, elapsed)

                # Dataset capture: one frame per drive tick when recording is on,
                # even if the active arm is stationary, grip is released, or the
                # VR watchdog skips motor writes. Passive arms still contribute
                # their observation.state.
                self._record_frame_if_active()

                # Phase 2: command the active arm if engaged + calibrated.
                if not engaged or active is None or active not in connected:
                    next_tick = self._sleep_until(next_tick)
                    continue
                # Don't VR-drive an arm that's currently homing — homing already
                # owns send_action above.
                if self._arms[active].homing:
                    next_tick = self._sleep_until(next_tick)
                    continue
                # Don't VR-drive an arm whose torque is released (user is
                # hand-posing it).
                if not MOTORS.is_torque_enabled(active):
                    next_tick = self._sleep_until(next_tick)
                    continue

                arm = self._arms[active]
                with self._lock:
                    goal = arm.latest
                    drained_dp, drained_dr = getattr(arm, "drained_dp_dr",
                                                      ((0,0,0),(0,0,0)))

                # Watchdog: skip if last goal too stale (controller put down).
                goal_age = now - goal.received_at if goal.has_data else 1e9
                if not goal.has_data or goal_age > GOAL_SKIP_AGE_S:
                    next_tick = self._sleep_until(next_tick)
                    continue
                if goal.mode != "position":
                    # Grip release sends IDLE. Do not let EMA filter tails keep
                    # moving the arm after control is intentionally released.
                    arm.pos_ema = (0.0, 0.0, 0.0)
                    next_tick = self._sleep_until(next_tick)
                    continue
                if not arm.calibrated:
                    next_tick = self._sleep_until(next_tick)
                    continue

                # Build joint targets from VR deltas.
                self._compute_targets_from_vr(active, goal, scale,
                                              drained_dp, drained_dr)

                # Per-tick joint clamp vs last sent (caps max joint velocity).
                prefix = f"{active}_arm_"
                raw = arm.targets.to_dict_with_prefix(active)
                clamped: dict[str, float] = {}
                for pj, val in raw.items():
                    cap = PER_TICK_DEG_CAPS.get(pj.removeprefix(prefix), 1.0)
                    prev = arm.last_sent_targets.get(pj, val)
                    delta = max(-cap, min(cap, val - prev))
                    clamped[pj] = prev + delta

                # P-controller blend — but only if KP < 1.0. KP=1.0 collapses to
                # `final = target`, so the present-position bus read (~10 ms) is
                # wasted work that would otherwise eat into our 33 ms tick budget.
                if KP >= 0.999:
                    final = clamped
                    present_full: dict[str, float] = {}     # for the debug log
                else:
                    present_full = MOTORS.read_positions(active)
                    final = {}
                    for pj, target in clamped.items():
                        here = present_full.get(pj, target)
                        final[pj] = here + KP * (target - here)

                MOTORS.send_action(active, final)
                arm.last_sent_targets = clamped

                # Debug: per-arm gripper trigger/target/sent/present log (1Hz).
                self._debug_log_gripper(active, goal, arm.targets,
                                         final, present_full, now)

            except Exception as e:
                log.exception("drive loop error: %s", e)
                with self._lock:
                    self._engaged = False
                    self._last_error = f"drive: {e}"

            next_tick = self._sleep_until(next_tick)

        log.info("drive loop exited")

    # ── guided calibration wizard ──────────────────────────────────────────
    def start_calibration(self, side: ArmSide) -> dict:
        """Begin a 3-vector motion-based calibration for one arm.

        State machine:
          idle → awaiting_anchor_fwd  → motioning_fwd
               → awaiting_anchor_up   → motioning_up
               → awaiting_anchor_left → motioning_left
               → idle (matrix applied + invert_lateral verified)

        Steps:
          1 (forward): user moves hand in their forward direction → captures
            user-forward axis in VR world frame.
          2 (up):      user moves hand up → captures user-up axis.
            After steps 1+2, the 3×3 session matrix is built via Gram-Schmidt.
          3 (left):    user moves hand to THEIR left → captures a verification
            vector. We transform it through M; if the resulting robot-frame y
            is NEGATIVE (i.e., motion ended up on robot's right despite user
            moving left), `invert_lateral` gets set to True. Catches motor
            sign-convention mismatches that the forward+up math alone misses.

        While calibration is active, the arm is force-unengaged so the robot
        doesn't drive during motion capture.
        """
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        with self._lock:
            if not MOTORS.is_connected(side):
                raise RuntimeError(f"{side} arm not connected")
            if self._engaged and self._active_arm == side:
                self._engaged = False
                self._active_arm = None
            arm = self._arms[side]
            arm.cal_state = "awaiting_anchor_fwd"
            arm.cal_motion_acc = (0.0, 0.0, 0.0)
            arm.cal_captured_fwd  = None
            arm.cal_captured_up   = None
            arm.cal_captured_left = None
            arm.cal_last_fwd_m  = 0.0
            arm.cal_last_up_m   = 0.0
            arm.cal_last_left_m = 0.0
            arm.calibrated = False
            log.info("[%s] calibration started; awaiting grip-press for forward axis", side)
            return self.status()

    def cancel_calibration(self, side: ArmSide) -> dict:
        with self._lock:
            arm = self._arms[side]
            if arm.cal_state == "idle":
                return self.status()
            log.info("[%s] calibration cancelled", side)
            arm.cal_state = "idle"
            arm.cal_motion_acc = (0.0, 0.0, 0.0)
            arm.cal_captured_fwd = None
            arm.cal_captured_up = None
            arm.cal_captured_left = None
            return self.status()

    # ── home pose capture + go-to-home ─────────────────────────────────────
    def capture_home(self, side: Optional[ArmSide] = None) -> dict:
        """Read present joint positions for the connected arm(s) and write them
        to `config/xlerobot.yaml`'s `robot.home_pose:` block.

        - `side="left"` or `"right"`: only that arm's joints are written.
        - `side=None`: writes for every connected arm (existing values for
          disconnected arms in the YAML are preserved).
        """
        with self._lock:
            sides = [side] if side else list(MOTORS.connected_sides)
            if not sides:
                raise RuntimeError("connect an arm before capturing home")
            for s in sides:
                if not MOTORS.is_connected(s):
                    raise RuntimeError(f"{s} arm not connected")
            pose: dict[str, float] = {}
            for s in sides:
                pres = MOTORS.read_positions(s)
                prefix = f"{s}_arm_"
                for j in _motors.JOINTS_PER_ARM:
                    key = f"{prefix}{j}"
                    if key in pres:
                        pose[key] = float(pres[key])
            try:
                _home.write_home_pose(pose)
            except Exception as e:
                self._last_error = f"capture_home: {e}"
                log.exception("capture_home failed")
                raise
            log.info("home pose captured for sides=%s: %d joints written",
                     sides, len(pose))
        return self.status()

    def go_home(self, side: Optional[ArmSide] = None) -> dict:
        """Begin a slow, per-tick-clamped interpolation from the current pose to
        the saved home pose. Uses the same drive loop as VR teleop (same KP,
        same per-tick caps, same bus.send_action path), so it's protected by
        all the existing safety guards. Forces the arm out of engage so the
        homing motion and VR teleop don't fight each other.
        """
        with self._lock:
            sides = [side] if side else list(MOTORS.connected_sides)
            if not sides:
                raise RuntimeError("connect an arm before homing")
            full_home = _home.read_home_pose()
            if not full_home:
                raise RuntimeError(
                    "no home pose saved — click 'Capture home' first while the "
                    "arm is in the desired starting pose."
                )
            for s in sides:
                if not MOTORS.is_connected(s):
                    raise RuntimeError(f"{s} arm not connected")
                target = {k: v for k, v in full_home.items()
                          if k.startswith(f"{s}_arm_")}
                if not target:
                    raise RuntimeError(
                        f"no home pose saved for {s} arm — capture one first"
                    )
                arm = self._arms[s]
                arm.home_target = target
                arm.homing = True
                arm.home_start_t = time.monotonic()
                # While homing, don't accept VR drive on this arm.
                if self._active_arm == s:
                    self._engaged = False
                    self._active_arm = None
                log.info("[%s] go_home started; %d joint targets queued", s, len(target))
        return self.status()

    def release_torque_for_posing(self, side: ArmSide) -> dict:
        """Disable torque on one arm so the user can hand-pose it. Forces the
        arm out of engage so VR drive won't fight the user. The drive loop
        skips arms with `torque_enabled=False`."""
        with self._lock:
            if not MOTORS.is_connected(side):
                raise RuntimeError(f"{side} arm not connected")
            if self._active_arm == side:
                self._engaged = False
                self._active_arm = None
            arm = self._arms[side]
            if arm.homing:
                arm.homing = False
                arm.home_target = {}
            MOTORS.release_torque_for_posing(side)
            # Invalidate the anchor — joint pose just changed unpredictably.
            arm.calibrated = False
            return self.status()

    def lock_torque(self, side: ArmSide) -> dict:
        """Re-enable torque on one arm at its current position (no snap-back).
        Caller should typically pair this with `capture_home(side)` if the
        intent was to pose-then-capture, but they're independent operations."""
        with self._lock:
            if not MOTORS.is_connected(side):
                raise RuntimeError(f"{side} arm not connected")
            MOTORS.lock_at_current(side)
            # Seed targets from the new pose so VR drive starts cleanly.
            self._seed_targets_from_present(side)
            return self.status()

    def cancel_homing(self, side: Optional[ArmSide] = None) -> dict:
        """Abort an in-progress homing motion. The arm freezes at its current
        pose (motor PID holds it)."""
        with self._lock:
            sides = [side] if side else ("left", "right")
            for s in sides:
                arm = self._arms[s]
                if arm.homing:
                    arm.homing = False
                    arm.home_target = {}
                    log.info("[%s] homing cancelled", s)
        return self.status()

    def wait_for_homing(self, sides: list[ArmSide], timeout_s: float = 10.0) -> bool:
        """Block until all `sides` finish homing (or timeout). Returns True if
        all finished, False if timeout hit. Caller MUST NOT hold `self._lock`."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if all(not self._arms[s].homing for s in sides):
                    return True
            time.sleep(0.05)
        log.warning("wait_for_homing timeout after %.1fs; arms=%s", timeout_s, sides)
        return False

    def _advance_calibration(self, side: ArmSide, goal: _LatestGoal) -> None:
        """Three-vector wizard state machine. Called WITH `self._lock` held."""
        arm = self._arms[side]
        state = arm.cal_state
        mode = str(goal.mode)
        if state == "idle":
            return
        # Grip-press transitions (RESET goal)
        if mode == "reset":
            if state == "awaiting_anchor_fwd":
                arm.cal_motion_acc = (0.0, 0.0, 0.0)
                arm.cal_state = "motioning_fwd"
                log.info("[%s] cal: anchor for forward captured; "
                         "move hand FORWARD ~10 cm, then release grip", side)
            elif state == "awaiting_anchor_up":
                arm.cal_motion_acc = (0.0, 0.0, 0.0)
                arm.cal_state = "motioning_up"
                log.info("[%s] cal: anchor for up captured; "
                         "move hand UP ~10 cm, then release grip", side)
            elif state == "awaiting_anchor_left":
                arm.cal_motion_acc = (0.0, 0.0, 0.0)
                arm.cal_state = "motioning_left"
                log.info("[%s] cal: anchor for left captured; "
                         "move hand LEFT ~10 cm, then release grip", side)
            return
        # Accumulate per-frame position deltas while moving.
        if mode == "position" and state in (
                "motioning_fwd", "motioning_up", "motioning_left"):
            dp = goal.rel_position
            arm.cal_motion_acc = (
                arm.cal_motion_acc[0] + float(dp[0]),
                arm.cal_motion_acc[1] + float(dp[1]),
                arm.cal_motion_acc[2] + float(dp[2]),
            )
            return
        # Grip-release transitions (IDLE goal)
        if mode == "idle":
            if state == "motioning_fwd":
                mag = math.sqrt(sum(v * v for v in arm.cal_motion_acc))
                if mag < CALIBRATION_MIN_MOTION_M:
                    log.warning("[%s] cal: forward motion too small (%.1f cm); "
                                "still awaiting forward — re-grip and move further",
                                side, mag * 100)
                    arm.cal_state = "awaiting_anchor_fwd"
                    return
                arm.cal_captured_fwd = arm.cal_motion_acc
                arm.cal_last_fwd_m = mag
                arm.cal_state = "awaiting_anchor_up"
                log.info("[%s] cal: forward axis captured (%.1f cm); "
                         "press grip and move hand UP ~10 cm to capture vertical axis",
                         side, mag * 100)
            elif state == "motioning_up":
                mag = math.sqrt(sum(v * v for v in arm.cal_motion_acc))
                if mag < CALIBRATION_MIN_MOTION_M:
                    log.warning("[%s] cal: up motion too small (%.1f cm); "
                                "re-grip and move further", side, mag * 100)
                    arm.cal_state = "awaiting_anchor_up"
                    return
                arm.cal_captured_up = arm.cal_motion_acc
                arm.cal_last_up_m = mag
                # Build the matrix NOW so step 3 (left-motion check) can use it.
                f = arm.cal_captured_fwd
                u = arm.cal_captured_up
                if f is not None and u is not None:
                    arm.session_vr_to_robot, arm.cal_confidence = (
                        _compute_session_frame_from_two_motions(f, u)
                    )
                arm.cal_state = "awaiting_anchor_left"
                log.info("[%s] cal: up axis captured (%.1f cm); matrix built. "
                         "Now press grip and move hand LEFT ~10 cm — we'll use "
                         "this to verify the lateral axis sign.", side, mag * 100)
            elif state == "motioning_left":
                mag = math.sqrt(sum(v * v for v in arm.cal_motion_acc))
                if mag < CALIBRATION_MIN_MOTION_M:
                    log.warning("[%s] cal: left motion too small (%.1f cm); "
                                "re-grip and move further", side, mag * 100)
                    arm.cal_state = "awaiting_anchor_left"
                    return
                arm.cal_captured_left = arm.cal_motion_acc
                arm.cal_last_left_m = mag
                self._finalize_calibration(side)

    def _finalize_calibration(self, side: ArmSide) -> None:
        """Verify the lateral axis sign using the third captured motion, then
        finish calibration. Called WITH `self._lock` held."""
        arm = self._arms[side]
        f = arm.cal_captured_fwd
        u = arm.cal_captured_up
        l = arm.cal_captured_left
        if f is None or u is None or l is None:
            log.warning("[%s] cal: finalize called without all three vectors", side)
            arm.cal_state = "idle"
            return
        # Additional orthogonality check between forward and left motions.
        # If too parallel (cos > 0.6), downgrade confidence — even if fwd/up
        # were well-separated, a poor left-motion can give a brittle sign.
        f_arr = _np.array(f, dtype=float); l_arr = _np.array(l, dtype=float)
        f_n = float(_np.linalg.norm(f_arr)); l_n = float(_np.linalg.norm(l_arr))
        if f_n > 1e-3 and l_n > 1e-3:
            cos_fl = abs(float(_np.dot(f_arr, l_arr) / (f_n * l_n)))
            if cos_fl > 0.6:
                arm.cal_confidence = "poor"
                log.warning("[%s] forward and left motions too parallel "
                            "(cos=%.2f); confidence POOR", side, cos_fl)

        # Verify lateral: transform the captured left-motion through M to robot
        # frame. y > 0 = "user-left → robot-left" (correct, no invert).
        # y < 0 = mirrored (set invert_lateral). BUT if the user has manually
        # set `invert_lateral_<side>` in config/xlerobot.yaml, that's an
        # OVERRIDE — typically for physically mirror-mounted motors that the
        # math can't see — and we skip the auto-decision.
        l_vec = _np.array(l, dtype=float)
        l_robot = arm.session_vr_to_robot @ l_vec
        if arm.invert_lateral_override:
            verdict = (f"OVERRIDDEN by YAML (invert_lateral_{side} explicitly set "
                       f"to {arm.invert_lateral}) — wizard's auto-decision skipped")
        else:
            arm.invert_lateral = bool(l_robot[1] < 0)
            verdict = ("INVERTED (set invert_lateral=True)" if arm.invert_lateral
                       else "OK (invert_lateral=False)")
        log.info(
            "[%s] calibration COMPLETE — forward=%s up=%s left=%s; "
            "left-check → robot delta=%s → lateral %s\n"
            "session matrix:\n%s\nSqueeze grip again to anchor for teleop.",
            side,
            tuple(f"{v:.3f}" for v in f),
            tuple(f"{v:.3f}" for v in u),
            tuple(f"{v:.3f}" for v in l),
            tuple(f"{v:.3f}" for v in l_robot),
            verdict,
            arm.session_vr_to_robot.round(3),
        )
        # Persist to disk so subsequent sessions don't need to re-run the wizard.
        try:
            _vrcal.write_for_arm(
                side, arm.session_vr_to_robot,
                forward_motion_m=arm.cal_last_fwd_m,
                up_motion_m=arm.cal_last_up_m,
            )
        except Exception as e:
            log.warning("[%s] could not persist VR calibration: %s", side, e)
        arm.cal_state = "idle"
        arm.cal_motion_acc = (0.0, 0.0, 0.0)
        # Note: arm.calibrated stays False — user must grip-press once more
        # to anchor for real teleop. The new session matrix will be applied
        # to subsequent VR deltas via `_compute_targets_from_vr`.

    def _handle_engage_button(self, side: ArmSide) -> None:
        """A button on right controller (or X on left) was just pressed.

        Toggle the engage state with this controller's arm as active:
          - Not engaged → engage on this side.
          - Engaged on this side → disengage.
          - Engaged on the OTHER side → switch to this side (keep engaged).
        Equivalent to clicking the UI Engage switch + picking active_arm.
        """
        with self._lock:
            if not MOTORS.is_connected(side):
                log.warning("[%s] engage button pressed but arm not connected", side)
                return
            if self._engaged and self._active_arm == side:
                self._engaged = False
                self._active_arm = None
                log.info("[%s] engage button → DISENGAGED", side)
            elif self._engaged and self._active_arm != side:
                self._active_arm = side
                log.info("[%s] engage button → SWITCHED active arm to %s", side, side)
            else:
                self._engaged = True
                self._active_arm = side
                log.info("[%s] engage button → ENGAGED on %s arm", side, side)

    def _handle_record_button(self, side: ArmSide) -> None:
        """B button on right controller was just pressed → toggle dataset recording."""
        log.info("[%s] B button → toggle recording", side)
        self.set_recording(not self._recording)

    def set_recording(self, enabled: bool, task: str = "",
                       home_first: Optional[bool] = None,
                       root: Optional[str] = None) -> bool:
        """Idempotent recording toggle. Lazy-creates the LeRobotDataset on first
        start, opens a new episode each ON transition, saves the episode on the
        OFF transition. Returns the new recording state.

        `home_first`: if True (or None and `dataset.home_before_episode: true`
        in config/xlerobot.yaml), move every connected arm to its saved home
        pose before opening the new episode. Ensures consistent training data.
        """
        # Resolve home_first from config if not explicitly set.
        if home_first is None:
            try:
                cfg = _dataset.load_dataset_config()
                home_first = bool(cfg.get("home_before_episode", False))
            except Exception:
                home_first = False

        # If starting recording AND home_first AND have home pose AND arms
        # connected: home them BEFORE opening the episode. Block until done.
        if enabled and home_first:
            with self._lock:
                sides_to_home = list(MOTORS.connected_sides)
                have_home = bool(_home.read_home_pose()) if sides_to_home else False
            if sides_to_home and have_home:
                log.info("recording start: homing %s before opening episode", sides_to_home)
                try:
                    self.go_home(side=None)  # all connected
                except Exception as e:
                    log.warning("auto-home before recording failed: %s", e)
                # Wait outside the lock; the drive loop runs the homing.
                self.wait_for_homing(sides_to_home, timeout_s=15.0)

        with self._lock:
            if bool(enabled) == self._recording:
                return self._recording
            # Resolve task: use the just-passed value if non-empty, else fall
            # back to whatever was last typed in the UI. Stash for next B-press.
            effective_task = (task or "").strip() or self._last_task
            if effective_task and enabled:
                self._last_task = effective_task
            if enabled:
                # Lazy-create the recorder on first start. Persists across
                # multiple episodes within the session.
                if self._recorder is None:
                    try:
                        cfg = _dataset.load_dataset_config()
                        roles, shape = _dataset.role_camera_list()
                        if not roles:
                            self._last_error = (
                                "no cameras have a role assigned in config/xlerobot.yaml — "
                                "go to the Cameras page and assign head/left_wrist/right_wrist"
                            )
                            log.warning(self._last_error)
                            return self._recording
                        # Resolve the storage root: explicit arg > YAML setting >
                        # HF default. Stashed for status display.
                        effective_root = (root or "").strip() or cfg.get("root") or None
                        self._last_dataset_root = _dataset.resolve_root(
                            effective_root, str(cfg["repo_id"]),
                        )
                        self._recorder = _dataset.DatasetRecorder(
                            repo_id=str(cfg["repo_id"]),
                            fps=int(cfg["fps"]),
                            camera_roles=roles,
                            camera_shape=shape,
                            root=effective_root,
                            push_to_hub=bool(cfg["push_to_hub"]),
                        )
                    except Exception as e:
                        self._last_error = f"recorder init: {e}"
                        log.exception("could not start dataset recorder")
                        return self._recording
                self._recorder.start_episode(task=effective_task)
                self._recording = True
            else:
                # End the in-flight episode. Capture writes finish on the
                # recorder's internal lock; we don't hold ours during the actual
                # save (which can take seconds for video encoding).
                self._recording = False
                rec = self._recorder
        # Save the episode OUTSIDE the session lock — `end_episode` flushes
        # frames + may invoke video encoding which can take a while.
        if not enabled and rec is not None:
            saved = rec.end_episode()
            # LeRobot buffers episode metadata until finalize(); without this,
            # the viewer sees data/video files but no meta/episodes parquet.
            if saved:
                rec.finalize()
                with self._lock:
                    if self._recorder is rec:
                        self._recorder = None
        return self._recording

    def _record_frame_if_active(self) -> None:
        """If recording is on and an episode is active, append one frame.

        Action = last commanded joint positions (per-arm, prefixed keys), merged
        across both arms. Observation.state = present joint positions (both arms).
        Observation.images.<role> = latest snapshot from each configured camera.
        Missing arm or camera data is filled with zeros by the recorder.
        """
        with self._lock:
            rec = self._recorder
            if not (self._recording and rec is not None and rec.in_episode):
                return
            # Snapshot dictionaries while holding the lock; release before doing
            # camera capture (which is slow).
            action_dict: dict[str, float] = {}
            for s in ("left", "right"):
                if MOTORS.is_connected(s):
                    action_dict.update(self._arms[s].last_sent_targets)
        # Outside lock: read present positions (bus I/O) + camera snapshots.
        try:
            present_dict = MOTORS.read_positions()
        except Exception as e:
            log.warning("record: read_positions failed: %s", e)
            present_dict = {}
        try:
            cam_frames = _dataset.grab_camera_frames()
        except Exception as e:
            log.warning("record: grab_camera_frames failed: %s", e)
            cam_frames = {}
        rec.add_frame(action_dict, present_dict, cam_frames)

    def _debug_log_gripper(self, side: ArmSide, goal: _LatestGoal,
                            targets: _LiveTargets, final: dict[str, float],
                            present: dict[str, float], now: float) -> None:
        """1Hz per-arm log showing trigger value vs gripper target vs sent vs present.
        Lets you bisect 'gripper not moving' between VR/IK/motor sides at a glance."""
        if not hasattr(self, "_dbg_gripper_state"):
            self._dbg_gripper_state: dict[ArmSide, dict[str, Any]] = {}
        state = self._dbg_gripper_state.setdefault(side, {"t": 0.0, "trig": None})
        trigger_now = bool(goal.trigger)
        if trigger_now != state["trig"] or (now - state["t"]) > 1.0:
            prefix = f"{side}_arm_"
            log.info(
                "[%s] gripper: trigger=%s target=%.1f sent=%.1f present=%.1f "
                "(open=%.1f closed=%.1f)",
                side, trigger_now, targets.gripper,
                final.get(f"{prefix}gripper", float("nan")),
                present.get(f"{prefix}gripper", float("nan")),
                self._gripper_open, self._gripper_closed,
            )
            state["t"] = now
            state["trig"] = trigger_now

    def _compute_targets_from_vr(self, side: ArmSide, goal: _LatestGoal,
                                  scale: float,
                                  drained_dp: tuple[float, float, float],
                                  drained_dr: tuple[float, float, float]) -> None:
        """Convert pre-drained VR deltas → joint targets via URDF IK.

        Pipeline (per-tick):
          1. Transform VR deltas through `arm.session_vr_to_robot` to robot frame.
          2. invert_lateral (if set): mirror lateral components.
          3. Deadband: zero out sub-noise-threshold deltas.
          4. Position-delta EMA: kills high-frequency controller translation noise.
          5. Scale + per-tick EE caps.
          6. Integrate position offset → target position = anchor + offset.
          7. Compute absolute desired orientation and smooth it with quaternion SLERP.
          8. Soft-saturate target to workspace (`EE_BOUNDS` box + reach sphere).
          9. URDF/placo IK with `position_weight=1.0, orientation_weight=0.1`
             and `last_q_sol` as initial guess (kills null-space jitter).
          10. EMA-filter joint targets and build `_LiveTargets` + gripper.
        """
        from scipy.spatial.transform import Rotation as _R
        arm = self._arms[side]
        M = arm.session_vr_to_robot

        # 1. Transform VR-frame deltas to robot-frame.
        dp_robot = M @ _np.array(drained_dp, dtype=float)
        dr_robot = M @ _np.array(drained_dr, dtype=float)

        # 2. invert_lateral: flip y position + x/y rotation components.
        if arm.invert_lateral:
            dp_robot = dp_robot * _np.array([1.0, -1.0, 1.0])
            dr_robot = dr_robot * _np.array([-1.0, -1.0, 1.0])

        # 3. Deadband.
        if float(_np.linalg.norm(dp_robot)) < POSITION_DEADBAND_M:
            dp_robot = _np.zeros(3)
        if float(_np.linalg.norm(dr_robot)) < ROTVEC_DEADBAND_RAD:
            dr_robot = _np.zeros(3)
        orientation_moved = float(_np.linalg.norm(dr_robot)) > 0.0

        # 4. Position-delta EMA. The filter has unity DC gain, so a finite hand
        # motion still integrates to the same final offset after the tail settles.
        a_pos = POS_EMA_ALPHA
        arm.pos_ema = (
            a_pos * float(dp_robot[0]) + (1.0 - a_pos) * arm.pos_ema[0],
            a_pos * float(dp_robot[1]) + (1.0 - a_pos) * arm.pos_ema[1],
            a_pos * float(dp_robot[2]) + (1.0 - a_pos) * arm.pos_ema[2],
        )
        dp_smoothed = _np.array(arm.pos_ema, dtype=float)

        # 5. Scale + per-tick caps.
        ee_cap = EE_DELTA_LIMIT_M * scale
        wrist_cap = WRIST_RAD_DELTA_LIMIT * scale
        dp = dp_smoothed * scale
        dp_norm = float(_np.linalg.norm(dp))
        if dp_norm > ee_cap:
            dp = dp * (ee_cap / dp_norm)

        # 6. Position offset: integrate UNCLAMPED so user hand-motion always
        # returns to anchor (no hysteresis when target hits a bound).
        arm.offset_robot = (
            arm.offset_robot[0] + float(dp[0]),
            arm.offset_robot[1] + float(dp[1]),
            arm.offset_robot[2] + float(dp[2]),
        )

        # 7. Absolute desired orientation: use the controller's current quat vs
        #    the anchor quat (both in VR world frame) to compute the absolute
        #    rotation since RESET, then transform to robot frame and apply to
        #    the anchor orientation. Drift-free — if you hold the controller
        #    still, the wrist target stays put exactly.
        anchor_q = arm.anchor.ctrl_quat
        current_q = goal.rotation_quat
        if anchor_q is not None and current_q is not None and orientation_moved:
            try:
                R_anchor_vr  = _R.from_quat(_np.array(anchor_q))
                R_current_vr = _R.from_quat(_np.array(current_q))
                R_delta_vr   = R_current_vr * R_anchor_vr.inv()
                # Similarity transform: rotation in VR frame → rotation in robot frame.
                R_delta_robot = M @ R_delta_vr.as_matrix() @ M.T
                # Apply to the anchor orientation (snapshotted at RESET), then
                # smooth the actual orientation fed to IK. The old rotvec EMA was
                # bypassed here, so raw Quest quaternion jitter reached the wrist.
                R_raw = R_delta_robot @ arm.anchor_R_robot
                R_target = _slerp_rotation_matrix(
                    arm.smoothed_R_target,
                    R_raw,
                    ORI_EMA_ALPHA,
                    max_step_rad=wrist_cap,
                )
                arm.smoothed_R_target = R_target.copy()
            except Exception as e:
                log.warning("[%s] orientation tracking failed (%s); freezing", side, e)
                R_target = arm.smoothed_R_target
        else:
            # Pre-RESET or quat missing: hold previous target orientation.
            R_target = arm.smoothed_R_target

        # 8. Target position from anchor + offset, axis-clamped to EE_BOUNDS
        # (sanity box), then radially clamped before IK to avoid placo hopping
        # between local minima near the edge of the SO101 reach envelope.
        tx = arm.anchor_ee_pos[0] + arm.offset_robot[0]
        ty = arm.anchor_ee_pos[1] + arm.offset_robot[1]
        tz = arm.anchor_ee_pos[2] + arm.offset_robot[2]
        tx = max(EE_BOUNDS["x"][0], min(EE_BOUNDS["x"][1], tx))
        ty = max(EE_BOUNDS["y"][0], min(EE_BOUNDS["y"][1], ty))
        tz = max(EE_BOUNDS["z"][0], min(EE_BOUNDS["z"][1], tz))
        target_pos = _clamp_to_workspace_reach(_np.array([tx, ty, tz], dtype=float))
        tx, ty, tz = (float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
        arm.target_T[:3, 3] = (tx, ty, tz)
        arm.target_T[:3, :3] = R_target

        # 9. URDF IK with regularization (orientation_weight=0.1) and
        # `last_q_sol` as initial guess. The last-good-solution seed (not raw
        # joint reads) is what makes placo NOT flip between null-space minima.
        try:
            q_sol = arm.kinematics.inverse_kinematics(
                arm.last_q_sol,
                arm.target_T,
                position_weight=1.0,
                orientation_weight=0.1,
            )
            # Reject only NaN/Inf — true IK divergence. Large per-tick joint
            # changes are normal for fast user motion and get clipped by the
            # PER_TICK_DEG_CAPS in the drive loop downstream; we don't need a
            # second cap here.
            if not _np.all(_np.isfinite(q_sol)):
                log.warning("[%s] IK output NaN/Inf; reusing previous q_sol", side)
                q_sol = arm.last_q_sol
            else:
                if arm.last_q_filtered is None:
                    arm.last_q_filtered = q_sol.copy()
                a_joint = JOINT_EMA_ALPHA
                q_sol = (1.0 - a_joint) * arm.last_q_filtered + a_joint * q_sol
                arm.last_q_filtered = q_sol.copy()
                arm.last_q_sol = q_sol.copy()
        except Exception as e:
            log.warning("[%s] IK failed (%s); reusing previous q_sol", side, e)
            q_sol = arm.last_q_sol

        # 10. Build the live joint targets. Order: shoulder_pan, shoulder_lift,
        #     elbow_flex, wrist_flex, wrist_roll (matches _IK_JOINT_ORDER).
        gripper_target = self._gripper_closed if goal.trigger else self._gripper_open
        arm.targets = _LiveTargets(
            shoulder_pan=float(q_sol[0]),
            shoulder_lift=float(q_sol[1]),
            elbow_flex=float(q_sol[2]),
            wrist_flex=float(q_sol[3]),
            wrist_roll=float(q_sol[4]),
            gripper=gripper_target,
        )

    @staticmethod
    def _sleep_until(next_tick: float) -> float:
        next_tick += LOOP_PERIOD_S
        wait = next_tick - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        elif wait < -0.2:
            # We're behind by >200 ms — likely the bus stalled; resync rather than spin.
            next_tick = time.monotonic()
        return next_tick

    # ── teardown ────────────────────────────────────────────────────────────
    def _stop_threads_and_servers(self) -> None:
        self._stop_evt.set()
        # Stop drive thread
        if self._drive_thread is not None and self._drive_thread.is_alive():
            self._drive_thread.join(timeout=2)
        self._drive_thread = None
        # Stop asyncio loop + WSS server
        if self._asyncio_loop is not None:
            try:
                async def _shutdown():
                    if self._ws_server is not None:
                        try: await self._ws_server.stop()
                        except Exception: pass
                fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._asyncio_loop)
                try: fut.result(timeout=3)
                except Exception: pass
                self._asyncio_loop.call_soon_threadsafe(self._asyncio_loop.stop)
            except Exception as e:
                log.warning("asyncio teardown: %s", e)
        if self._asyncio_thread is not None and self._asyncio_thread.is_alive():
            self._asyncio_thread.join(timeout=3)
        self._asyncio_loop = None
        self._asyncio_thread = None
        self._ws_server = None
        # Stop HTTPS server
        if self._https is not None:
            try: self._https.stop()
            except Exception as e: log.warning("https stop: %s", e)
            self._https = None


SESSION = VRTeleopSession()
