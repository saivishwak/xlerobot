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
KP = 1.0

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

# Per-frame wrist rotation cap. The cartesian offset has its own LERP-based
# smoothing in `_compute_targets_from_vr`; this is the equivalent step cap on
# orientation, scaled per-tick by `scale` (user-adjustable in the UI).
WRIST_RAD_DELTA_LIMIT = math.radians(5)
SAFE_REAR_X_M = 0.035
IK_JUMP_REJECT_DEG = 25.0

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
DUAL_MODE_BUTTON_BY_SIDE: dict[str, str] = {"left": "Y"}

# Minimum motion magnitude (in metres) required to accept a calibration. Anything
# smaller than this is too noisy to reliably determine "user-forward" direction.
CALIBRATION_MIN_MOTION_M: float = 0.05    # 5 cm
CALIBRATION_TARGET_MOTION_M: float = 0.10  # the wizard says "move ~10 cm"
WRIST_VERIFY_MIN_DEG: float = 8.0
WRIST_VERIFY_TARGET_DEG: float = 20.0

# Smoothing factors. 1.0 = raw input, 0.0 = frozen.
POS_EMA_ALPHA: float = 0.5
ORI_EMA_ALPHA: float = 0.4
# Hardware polarity of the wrist motors, per arm. Loaded from
# `config/xlerobot.yaml` (`vr.wrist_motor_polarity`) at startup.
#
# These are NOT the runtime `wrist_flex_sign` / `wrist_roll_sign` — those are
# derived per session from the polarity combined with the session_vr_to_robot
# matrix (see `_derive_wrist_signs_from_session`). The polarity is the only
# truly invariant piece: it depends on motor mounting and wiring, not on where
# the user stood when calibrating. Defaults below match the standard XLeRobot
# SO101 build; override in yaml only when rewiring/remounting actually changes
# a motor's direction.
_WRIST_MOTOR_POLARITY: dict[str, dict[str, float]] = {
    "left":  {"flex": -1.0, "roll": -1.0},
    "right": {"flex": -1.0, "roll": -1.0},
}


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

# Rotation noise deadband. Sub-degree wrist twitches at rest are controller
# jitter, not intentional motion — skip the SLERP step for changes below this
# threshold so the smoothed orientation target doesn't crawl.
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
from scipy.spatial.transform import Rotation as _R

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


def _project_to_rotation_matrix(matrix: _np.ndarray) -> _np.ndarray:
    """Project a near-rotation matrix to SO(3), matching BEAVR's safety step."""
    try:
        u, _, vt = _np.linalg.svd(_np.array(matrix, dtype=float))
        rot = u @ vt
        if _np.linalg.det(rot) < 0:
            vt[-1, :] *= -1.0
            rot = u @ vt
        return rot
    except Exception:
        return _np.eye(3)


def _positive_quat_xyzw(quat: _np.ndarray) -> _np.ndarray:
    q = _np.array(quat, dtype=float)
    norm = float(_np.linalg.norm(q))
    if norm < 1e-9:
        return _np.array([0.0, 0.0, 0.0, 1.0])
    q = q / norm
    if q[3] < 0:
        q = -q
    return q


def _pose_matrix_from_vr(position: tuple[float, float, float],
                         quat_xyzw: tuple[float, float, float, float]) -> _np.ndarray:
    from scipy.spatial.transform import Rotation as _R

    T = _np.eye(4)
    T[:3, 3] = _np.array(position, dtype=float)
    T[:3, :3] = _R.from_quat(_positive_quat_xyzw(_np.array(quat_xyzw, dtype=float))).as_matrix()
    return T


def _controller_rotation_delta_for_side(side: ArmSide, rotation_delta_vr: _np.ndarray) -> _np.ndarray:
    """Normalize controller rotation handedness before VR->robot mapping.

    WebXR controller positions share the same room frame, but the left and right
    controller local orientation frames are mirrored. The right-hand controller
    rotation matches the robot mapping; the left-hand controller rotation must be
    inverted so wrist/tool rotation follows the same user intent.
    """
    rot = _project_to_rotation_matrix(rotation_delta_vr)
    if side == "left":
        return rot.T
    return rot


def _wrist_rotation_deg_since_anchor(
    anchor_q: tuple[float, float, float, float],
    release_q: tuple[float, float, float, float],
) -> float:
    """Total rotation (degrees) from anchor quaternion to release quaternion."""
    R_anchor = _R.from_quat(anchor_q)
    R_release = _R.from_quat(release_q)
    R_rel = R_anchor.inv() * R_release
    return math.degrees(float(_np.linalg.norm(_np.asarray(R_rel.as_rotvec(), dtype=float))))


def _derive_wrist_signs_from_session(
    side: ArmSide,
    session_vr_to_robot: _np.ndarray,
    polarity: dict[str, float],
    pitch_canonical: Optional[tuple[float, float, float]] = None,
) -> tuple[float, float]:
    """Compute the per-session `wrist_flex_sign` / `wrist_roll_sign` analytically
    from the session_vr_to_robot matrix and the hardware motor polarity.

    Background
    ----------
    The runtime wrist mapping does

        wrist_flex_delta_deg = wrist_flex_sign * (M @ rotvec_vr_anchor_local)[1]
        wrist_roll_delta_deg = wrist_roll_sign * (M @ rotvec_vr_anchor_local)[0]

    where rotvec_vr_anchor_local is the controller's rotation since engage,
    expressed in its own anchor-local frame (post `_controller_rotation_delta_for_side`),
    and M = session_vr_to_robot.

    For the SAME physical user motion ("pitch wrist up"), the canonical
    rotvec is constant in form. Multiplying by M maps that into robot frame,
    and the SIGN of the resulting y-component depends on M — which legitimately
    flips when the user calibrates from a different orientation.

    To keep the wrist tracking the user invariant across calibrations, we
    cannot pin `wrist_flex_sign` as a constant; we must combine the hardware
    motor polarity (a true invariant — depends on wiring and mounting) with
    the M-dependent observed sign of the canonical motion.

    Canonical motions
    -----------------
    - "Pitch up": default WebXR controller anchor-local rotvec = +x. The
      left-side handedness flip in `_controller_rotation_delta_for_side`
      (transpose) negates this, so the *effective* canonical for left is -x.
    - "Roll right": controller anchor-local rotvec = -z. Negated for left side.

    Parameters
    ----------
    pitch_canonical : optional override (3-vector)
        Empirical anchor-local unit rotvec captured by the wrist-verify
        wizard step. SIDE HANDEDNESS ALREADY APPLIED, so it's used as-is.
        Pass None to fall back to the WebXR analytical canonical above —
        works for standard Quest/WebXR controllers, may be off for
        non-standard ones.

    Returns
    -------
    `(wrist_flex_sign, wrist_roll_sign)` — each ±1.0, ready to be written to
    `arm.wrist_flex_sign` / `arm.wrist_roll_sign`.
    """
    M = _np.asarray(session_vr_to_robot, dtype=float)
    if pitch_canonical is not None:
        pitch_canon_local = _np.asarray(pitch_canonical, dtype=float)
        norm = float(_np.linalg.norm(pitch_canon_local))
        if norm > 1e-6:
            pitch_canon_local = pitch_canon_local / norm
        else:
            pitch_canon_local = _np.array([1.0, 0.0, 0.0])
            if side == "left":
                pitch_canon_local = -pitch_canon_local
    else:
        pitch_canon_local = _np.array([1.0, 0.0, 0.0])
        if side == "left":
            pitch_canon_local = -pitch_canon_local

    # Roll canonical: no per-arm capture step, so always analytical default.
    roll_canon_local = _np.array([0.0, 0.0, -1.0])
    if side == "left":
        roll_canon_local = -roll_canon_local

    pitch_robot = M @ pitch_canon_local
    roll_robot  = M @ roll_canon_local
    pitch_obs_sign = 1.0 if float(pitch_robot[1]) >= 0.0 else -1.0
    roll_obs_sign  = 1.0 if float(roll_robot[0])  >= 0.0 else -1.0
    # Choose runtime sign so wrist_flex_delta = polarity['flex'] * |…| (i.e., the
    # motor delta's sign matches polarity for the canonical motion).
    flex_pol = 1.0 if float(polarity.get("flex", -1.0)) >= 0.0 else -1.0
    roll_pol = 1.0 if float(polarity.get("roll", -1.0)) >= 0.0 else -1.0
    wrist_flex_sign = flex_pol / pitch_obs_sign
    wrist_roll_sign = roll_pol / roll_obs_sign
    return (1.0 if wrist_flex_sign >= 0.0 else -1.0,
            1.0 if wrist_roll_sign >= 0.0 else -1.0)


def _clamp_to_workspace_reach(position: _np.ndarray) -> _np.ndarray:
    """Keep the requested EE target inside the robot-base reach sphere."""
    radius = float(_np.linalg.norm(position))
    if radius <= WORKSPACE_REACH_M or radius <= 1e-9:
        return position
    return position * (WORKSPACE_REACH_M / radius)


def _clamp_target_position(position: _np.ndarray) -> _np.ndarray:
    """Clamp EE target before IK, including a rear guard for base singularities."""
    target = _np.array(position, dtype=float).copy()
    target[0] = max(EE_BOUNDS["x"][0], min(EE_BOUNDS["x"][1], float(target[0])))
    target[1] = max(EE_BOUNDS["y"][0], min(EE_BOUNDS["y"][1], float(target[1])))
    target[2] = max(EE_BOUNDS["z"][0], min(EE_BOUNDS["z"][1], float(target[2])))
    target[0] = max(SAFE_REAR_X_M, float(target[0]))
    return _clamp_to_workspace_reach(target)


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

    Kept around as a fallback for the rare case where the wizard's 3rd ("left")
    motion is missing/degenerate — `_compute_session_frame_from_three_motions`
    is the preferred path.
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


def _compute_session_frame_from_three_motions(
    motion_fwd_vr: tuple[float, float, float],
    motion_up_vr: tuple[float, float, float],
    motion_left_vr: tuple[float, float, float],
) -> tuple[_np.ndarray, str]:
    """Build the session VR→robot rotation matrix from all three USER-MOTION
    vectors via constrained least-squares (Kabsch / Procrustes).

    The user moved their hand FORWARD, UP, and to-their-LEFT (in their body
    frame). We want a rotation matrix M ∈ SO(3) such that:

        M @ motion_fwd_vr  ≈ +robot_x  (forward away from arm)
        M @ motion_up_vr   ≈ +robot_z  (vertical up)
        M @ motion_left_vr ≈ +robot_y  (robot's left)

    Three motion vectors give 9 equations for SO(3)'s 3 DoFs — overdetermined.
    The Kabsch solution finds the closest rotation matrix in least-squares
    sense, averaging out any single-motion noise (off-axis drift, jitter in a
    hand-held motion). Much more robust than the 2-motion Gram-Schmidt path,
    which can be perturbed by any small noise on the up-motion vector.

    Confidence:
      - "good"  if all three motions are well separated AND the residual
                fit error is small.
      - "poor"  if any pair is too parallel (cos > 0.6) OR the residual is large.

    Falls back to the 2-motion path if the left motion is degenerate (too
    small or perfectly aligned with the existing axes).
    """
    f = _np.array(motion_fwd_vr,  dtype=float)
    u = _np.array(motion_up_vr,   dtype=float)
    l = _np.array(motion_left_vr, dtype=float)
    fn = float(_np.linalg.norm(f))
    un = float(_np.linalg.norm(u))
    ln = float(_np.linalg.norm(l))
    if fn < 1e-3 or un < 1e-3 or ln < 1e-3:
        log.warning(
            "3-motion calibration has a degenerate vector (|fwd|=%.3f, |up|=%.3f, "
            "|left|=%.3f); falling back to 2-motion build",
            fn, un, ln,
        )
        return _compute_session_frame_from_two_motions(motion_fwd_vr, motion_up_vr)

    f_hat = f / fn
    u_hat = u / un
    l_hat = l / ln

    # Pairwise separation check (degrees apart).
    cos_fu = abs(float(_np.dot(f_hat, u_hat)))
    cos_fl = abs(float(_np.dot(f_hat, l_hat)))
    cos_ul = abs(float(_np.dot(u_hat, l_hat)))
    cos_max = max(cos_fu, cos_fl, cos_ul)
    confidence = "good"
    if cos_max > 0.6:
        confidence = "poor"
        log.warning(
            "3-motion calibration: motions too parallel (max cos=%.2f, ≈%.1f° apart); "
            "matrix confidence POOR. Re-run wizard with more orthogonal motions.",
            cos_max, math.degrees(math.acos(min(1.0, cos_max))),
        )

    # Procrustes: minimize ||M A - B||_F over M ∈ SO(3).
    #   A's columns are the normalized VR motions.
    #   B's columns are the target robot-frame unit axes.
    A = _np.stack([f_hat, u_hat, l_hat], axis=1)
    B = _np.stack([
        _np.array([1.0, 0.0, 0.0]),  # +x_robot ← fwd
        _np.array([0.0, 0.0, 1.0]),  # +z_robot ← up
        _np.array([0.0, 1.0, 0.0]),  # +y_robot ← left
    ], axis=1)

    H = B @ A.T
    U_, _S, Vt = _np.linalg.svd(H)
    D = _np.eye(3)
    if _np.linalg.det(U_ @ Vt) < 0:
        D[-1, -1] = -1.0
    M = U_ @ D @ Vt

    # Residual fit error: how far is M from a perfect mapping of all three?
    residual = float(_np.linalg.norm(M @ A - B, ord="fro"))
    if residual > 0.5 and confidence == "good":
        confidence = "poor"
        log.warning(
            "3-motion calibration: large residual fit error (%.3f); "
            "user motions disagree about the frame. Confidence POOR.",
            residual,
        )
    else:
        log.info(
            "3-motion calibration: residual fit error %.3f (max cos %.2f); confidence %s",
            residual, cos_max, confidence,
        )

    return M, confidence


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
    """Snapshot of the most recent VR goal — used for status display, the
    calibration wizard (which reads per-frame `rel_position`/`rel_rotvec`), and
    the SE(3) controller-pose mapping (which reads `controller_position` +
    `rotation_quat`).

    `buttons` carries the Quest face-button pressed-state, keyed by Meta's labels:
        right controller → {"A": bool, "B": bool}
        left  controller → {"X": bool, "Y": bool}
    """
    received_at: float = 0.0
    has_data: bool = False
    mode: str = "idle"            # "idle" | "position" | "reset"
    rel_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rel_rotvec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    controller_position: Optional[tuple[float, float, float]] = None
    rotation_quat: Optional[tuple[float, float, float, float]] = None
    trigger: bool = False
    thumbstick: tuple[float, float] = (0.0, 0.0)
    buttons: dict[str, bool] = field(default_factory=dict)


@dataclass
class _AnchorPose:
    """Robot + controller state snapshotted at the most recent RESET.

    Per-tick wrist targets are derived as (anchor_wrist + rotation_delta) where
    rotation_delta is the absolute current-vs-anchor controller quaternion.
    Position uses the absolute controller pose carried in every goal too: the
    VR-world displacement from the controller anchor maps through
    `session_vr_to_robot` directly to the robot-frame cartesian offset, with no
    per-frame delta integration."""
    ee_x: float = 0.0
    ee_y: float = 0.0
    pan_deg: float = 0.0
    shoulder_lift_deg: float = 0.0
    elbow_flex_deg: float = 0.0
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
    # Cumulative offset from `anchor_ee_pos` in robot base frame, integrated from
    # VR position deltas and reconciled to the reachable target each tick. Do not
    # allow hidden offset to grow past workspace limits; that stored "debt" can
    # release later as a sudden jump.
    offset_robot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    anchor: _AnchorPose = field(default_factory=_AnchorPose)
    targets: _LiveTargets = field(default_factory=_LiveTargets)
    last_sent_targets: dict[str, float] = field(default_factory=dict)
    # Per-session VR→robot rotation matrix re-derived at every RESET from the
    # controller's orientation at grip-press. Defaults to the fixed _VR_TO_ROBOT
    # until calibration.
    session_vr_to_robot: _np.ndarray = field(default_factory=lambda: _VR_TO_ROBOT.copy())
    latest: _LatestGoal = field(default_factory=_LatestGoal)
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
    # Wrist-verify step 4: controller quaternion captured at grip-press, used
    # to compute the delta rotation at grip-release.
    cal_anchor_quat_for_wrist: Optional[tuple[float, float, float, float]] = None
    # Latest release quaternion latched from the last POSITION goal while in
    # motioning_wrist_verify (fallback when IDLE goal omits rotation).
    cal_wrist_release_quat: Optional[tuple[float, float, float, float]] = None
    # Live wrist rotation since the step-4 anchor (degrees) — for wizard UI.
    cal_wrist_verify_deg: float = 0.0
    # Per-arm anchor-local rotvec (unit vector) the user's wrist rotates around
    # when pitching UP. SIDE HANDEDNESS ALREADY APPLIED — fed directly to M in
    # `_derive_wrist_signs_from_session`. None = use the WebXR analytical
    # default (±x_anchor_local depending on side). Captured by the wrist-verify
    # wizard step; persisted in `vr_calibration.yaml` as `wrist_pitch_anchor_local`.
    wrist_pitch_canonical: Optional[tuple[float, float, float]] = None
    # Last completion-time motion magnitudes (m) — for UI to show "calibrated to N cm"
    cal_last_fwd_m:  float = 0.0
    cal_last_up_m:   float = 0.0
    cal_last_left_m: float = 0.0
    cal_validation: dict[str, Any] = field(default_factory=dict)
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
    wrist_flex_sign: float = 1.0
    wrist_roll_sign: float = -1.0
    # Per-arm URDF kinematics + last-good IK solution. Built lazily on first
    # RESET (see _ensure_kinematics). The IK uses `last_q_sol` as the initial
    # guess on every subsequent tick — this is the key trick that kills
    # null-space jitter on the 5-DOF arm (vs using noisy current joints).
    kinematics: Any = None
    last_q_sol: _np.ndarray = field(default_factory=lambda: _np.zeros(5, dtype=float))
    using_analytical_fallback: bool = False
    # Per-arm filtered target state. The cartesian offset is LERP-smoothed in
    # `_compute_targets_from_vr` (no separate field); orientation uses SLERP EMA
    # on the actual IK target. There is no joint-level EMA — relying on the
    # per-tick joint cap as the only motor-rate limiter (matches wrist behaviour
    # and avoids double smoothing on pan/lift/elbow).
    smoothed_R_target: _np.ndarray = field(default_factory=lambda: _np.eye(3))
    last_q_filtered: Optional[_np.ndarray] = None
    # Anchor orientation matrix (3×3) captured at RESET. Combined with the
    # current controller quaternion, gives the absolute desired EE orientation.
    anchor_R_robot: _np.ndarray = field(default_factory=lambda: _np.eye(3))
    controller_anchor_T: Optional[_np.ndarray] = None
    robot_anchor_T: _np.ndarray = field(default_factory=lambda: _np.eye(4))
    target_R_robot: _np.ndarray = field(default_factory=lambda: _np.eye(3))
    # Calibration confidence: "good" if the wizard's captured motion vectors
    # were well-separated, "poor" if too parallel (and the matrix is shaky).
    cal_confidence: str = "good"
    stale_since: Optional[float] = None
    last_commanded_targets: dict[str, float] = field(default_factory=dict)
    last_diag: dict[str, Any] = field(default_factory=dict)


class VRTeleopSession:
    def __init__(self):
        self._lock = threading.RLock()

        # Per-arm state — always created for both sides; populated when connected.
        self._arms: dict[ArmSide, _PerArm] = {
            "left":  _PerArm(side="left"),
            "right": _PerArm(side="right"),
        }
        # The arm that VR is currently driving in single-arm mode. In dual mode,
        # both connected arms are driven and this remains the preferred fallback
        # when dual mode is turned off.
        self._active_arm: Optional[ArmSide] = None
        self._dual_mode: bool = False

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
        self._recording_transition_lock = threading.Lock()
        self._episodes_saved: int = 0
        self._last_saved_episode_index: Optional[int] = None
        self._last_saved_episode_frames: int = 0
        # Last task string synced from the UI. Cached here so the Quest B button
        # can start an episode with the task the user typed on the web page.
        # Empty text clears the cache and recording start is rejected.
        self._last_task: str = ""
        # Resolved (absolute, ~-expanded) dataset storage root from most recent
        # recorder init. Shown on the UI's Recording card.
        self._last_dataset_root: str = ""
        self._recording_repo_id: Optional[str] = None

        # Kinematics — analytical only; URDF IK has null-space jitter on the
        # redundant 5-DOF arm. URDF FK was used at RESET; analytical FK is now
        # used there too for round-trip exactness with the analytical IK.
        self._analytical_kin = _SO101Kin()

        # Restore previously-saved VR calibrations from config/vr_calibration.yaml
        # so the user doesn't have to re-run the wizard every session. New
        # calibrations overwrite the file via `_finalize_calibration`.
        self._load_persisted_calibrations()
        # Populate UI recording counters from existing dataset (if present)
        # before the first "Start recording" click.
        self._bootstrap_recording_status_from_disk()

    def _bootstrap_recording_status_from_disk(self) -> None:
        """Load existing dataset episode summary for UI status on startup."""
        try:
            cfg = _dataset.load_dataset_config()
            repo_id = str(cfg["repo_id"])
            configured_root = cfg.get("root")
            resolved_root = _dataset.resolve_root(configured_root, repo_id)
            idx, frames = _dataset.last_episode_summary(repo_id=repo_id, root=configured_root)
            self._recording_repo_id = repo_id
            self._last_dataset_root = resolved_root
            self._last_saved_episode_index = idx
            self._last_saved_episode_frames = frames
            self._episodes_saved = (idx + 1) if idx is not None else 0
        except Exception as e:
            # Keep status operational even if dataset metadata is absent/corrupt.
            log.info("recording bootstrap skipped: %s", e)

    def _load_persisted_calibrations(self) -> None:
        """Restore per-arm session_vr_to_robot from disk. Silent no-op if no
        file exists or the file is malformed. Also reads per-arm invert_lateral
        flags from config/xlerobot.yaml's `vr:` section, both the value AND
        whether it's explicitly set (override mode). Also reads the global
        smoothing and rate-limit factors, plus the hardware
        `vr.wrist_motor_polarity` block (per-arm motor polarity used to derive
        runtime wrist signs)."""
        import yaml
        global KP, WRIST_RAD_DELTA_LIMIT
        global POS_EMA_ALPHA, ORI_EMA_ALPHA
        try:
            cfg = yaml.safe_load((REPO_ROOT / "config" / "xlerobot.yaml").read_text()) or {}
            vr_section = cfg.get("vr") or {}
            def _float_key(name: str, default: float, lo: float, hi: float) -> float:
                value = vr_section.get(name)
                if value is None:
                    return default
                return max(lo, min(hi, float(value)))

            KP = _float_key("kp", KP, 0.0, 1.0)
            wrist_deg = _float_key("wrist_delta_limit_deg", math.degrees(WRIST_RAD_DELTA_LIMIT), 1.0, 30.0)
            WRIST_RAD_DELTA_LIMIT = math.radians(wrist_deg)
            POS_EMA_ALPHA = _float_key("pos_ema_alpha", POS_EMA_ALPHA, 0.0, 1.0)
            # Backward-compatible alias: older configs used rotvec_ema_alpha.
            ori_alpha = vr_section.get("ori_ema_alpha", vr_section.get("rotvec_ema_alpha"))
            if ori_alpha is not None:
                ORI_EMA_ALPHA = max(0.0, min(1.0, float(ori_alpha)))
            joint_caps = vr_section.get("joint_deg_caps") or {}
            if isinstance(joint_caps, dict):
                for joint, cap in joint_caps.items():
                    if joint in PER_TICK_DEG_CAPS:
                        PER_TICK_DEG_CAPS[joint] = max(0.1, min(30.0, float(cap)))
            # Per-arm hardware motor polarity (NOT the runtime signs — those are
            # derived from polarity × session matrix at restore time). Anything
            # missing falls back to the defaults in _WRIST_MOTOR_POLARITY.
            polarity_block = vr_section.get("wrist_motor_polarity") or {}
            if isinstance(polarity_block, dict):
                for side in ("left", "right"):
                    arm_block = polarity_block.get(side) or {}
                    if not isinstance(arm_block, dict):
                        continue
                    for axis in ("flex", "roll"):
                        if axis not in arm_block:
                            continue
                        try:
                            raw = float(arm_block[axis])
                        except (TypeError, ValueError):
                            log.warning(
                                "vr.wrist_motor_polarity.%s.%s is not numeric (%r); "
                                "keeping default %+.0f",
                                side, axis, arm_block[axis],
                                _WRIST_MOTOR_POLARITY[side][axis],
                            )
                            continue
                        _WRIST_MOTOR_POLARITY[side][axis] = 1.0 if raw >= 0 else -1.0
            log.info(
                "VR smoothing loaded: kp=%.2f pos_ema=%.2f ori_ema=%.2f "
                "wrist_cap=%.1f° wrist_motor_polarity(left)=(flex %+.0f, roll %+.0f) "
                "wrist_motor_polarity(right)=(flex %+.0f, roll %+.0f)",
                KP, POS_EMA_ALPHA, ORI_EMA_ALPHA,
                math.degrees(WRIST_RAD_DELTA_LIMIT),
                _WRIST_MOTOR_POLARITY["left"]["flex"], _WRIST_MOTOR_POLARITY["left"]["roll"],
                _WRIST_MOTOR_POLARITY["right"]["flex"], _WRIST_MOTOR_POLARITY["right"]["roll"],
            )
        except Exception as e:
            log.warning("could not read VR smoothing config from YAML: %s", e)
        for side in ("left", "right"):
            self._restore_persisted_arm_config(side)

    def _restore_persisted_arm_config(self, side: ArmSide) -> None:
        """Restore saved calibration and lateral mapping for a freshly reset arm.

        Runtime `wrist_flex_sign` / `wrist_roll_sign` are derived from the
        saved session_vr_to_robot matrix combined with the hardware motor
        polarity (`vr.wrist_motor_polarity`). They are intentionally NOT read
        from `vr_calibration.yaml` — the wizard never writes them, and pinning
        them as YAML constants doesn't survive recalibrating from a different
        orientation (the M matrix's first column flips, and the canonical
        controller motion lands on the opposite robot-frame axis). Deriving
        them per-session is what keeps the wrist tracking the user invariant.
        """
        invert_flags = _vrcal.read_invert_lateral_flags()
        overrides = _vrcal.read_invert_lateral_overrides()
        arm = self._arms[side]
        arm.invert_lateral_override = overrides.get(side, False)
        data = _vrcal.read_for_arm(side) or {}
        if arm.invert_lateral_override:
            arm.invert_lateral = invert_flags.get(side, False)
        else:
            arm.invert_lateral = bool(data.get("invert_lateral", invert_flags.get(side, False)))
        arm.cal_confidence = str(data.get("confidence", arm.cal_confidence) or "good")
        arm.cal_last_fwd_m = float(data.get("forward_motion_m", 0.0))
        arm.cal_last_up_m = float(data.get("up_motion_m", 0.0))
        arm.cal_last_left_m = float(data.get("left_motion_m", 0.0))
        polarity = _WRIST_MOTOR_POLARITY.get(side, {"flex": -1.0, "roll": -1.0})
        # Empirical wrist pitch canonical (anchor-local rotvec, side-handedness
        # already applied), captured by the optional wrist-verify wizard step.
        # Absent → use the WebXR analytical default.
        saved_canonical = data.get("wrist_pitch_anchor_local")
        if (isinstance(saved_canonical, (list, tuple))
                and len(saved_canonical) == 3
                and all(isinstance(v, (int, float)) for v in saved_canonical)):
            arm.wrist_pitch_canonical = (float(saved_canonical[0]),
                                          float(saved_canonical[1]),
                                          float(saved_canonical[2]))
        else:
            arm.wrist_pitch_canonical = None
        M = _vrcal.matrix_for_arm(side)
        if M is None:
            # No calibration on disk yet — fall back to the identity-mapping
            # default session matrix (already set on `arm`) so the derived
            # signs are at least self-consistent until the user calibrates.
            arm.wrist_flex_sign, arm.wrist_roll_sign = _derive_wrist_signs_from_session(
                side, arm.session_vr_to_robot, polarity,
                pitch_canonical=arm.wrist_pitch_canonical,
            )
            return
        arm.session_vr_to_robot = M
        arm.wrist_flex_sign, arm.wrist_roll_sign = _derive_wrist_signs_from_session(
            side, M, polarity, pitch_canonical=arm.wrist_pitch_canonical,
        )
        canon_label = "empirical" if arm.wrist_pitch_canonical is not None else "analytical"
        log.info(
            "[%s] restored saved VR calibration (invert_lateral=%s, override=%s, "
            "confidence=%s, derived wrist_signs=(flex %+.0f, roll %+.0f) "
            "from polarity=(flex %+.0f, roll %+.0f), pitch canonical=%s)",
            side, arm.invert_lateral, arm.invert_lateral_override,
            arm.cal_confidence, arm.wrist_flex_sign, arm.wrist_roll_sign,
            polarity["flex"], polarity["roll"], canon_label,
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
            self._restore_persisted_arm_config(side)
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
                if self._dual_mode:
                    self._dual_mode = False
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
            self._dual_mode = False
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
            else:
                with self._lock:
                    self._episodes_saved = rec.episode_count
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
        - `engaged=False` clears the active arm and dual mode.
        - This API is single-arm selection; the left-controller Y button toggles
          dual mode without changing the target/IK/control code.
        """
        with self._lock:
            old_active = self._active_arm
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
                if arm.cal_confidence in ("poor", "legacy"):
                    raise RuntimeError(
                        f"{self._active_arm} VR calibration confidence is {arm.cal_confidence}; rerun calibration before engaging"
                    )
                if not arm.calibrated:
                    log.info(
                        "engage on %s but vr_calibrated=False — motors stay still"
                        " until the Quest controller's RESET (grip) is pressed",
                        self._active_arm,
                    )
            else:
                self._active_arm = None
                self._dual_mode = False
            self._engaged = bool(engaged)
            if self._engaged and self._active_arm is not None:
                self._dual_mode = False
            if self._active_arm is not None and self._active_arm != old_active:
                arm = self._arms[self._active_arm]
                arm.stale_since = None
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
                        "position": (list(arm.latest.controller_position)
                                     if arm.latest.controller_position else None),
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
                        "validation":          dict(arm.cal_validation),
                        "wizard_fwd_captured":  arm.cal_captured_fwd  is not None,
                        "wizard_up_captured":   arm.cal_captured_up   is not None,
                        "wizard_left_captured": arm.cal_captured_left is not None,
                        "wizard_wrist_verify_deg": arm.cal_wrist_verify_deg,
                        "wizard_wrist_verify_target_deg": WRIST_VERIFY_TARGET_DEG,
                        "wizard_wrist_verify_min_deg": WRIST_VERIFY_MIN_DEG,
                        "wizard_wrist_captured": arm.wrist_pitch_canonical is not None,
                        "wrist_pitch_canonical": (
                            list(arm.wrist_pitch_canonical)
                            if arm.wrist_pitch_canonical is not None else None
                        ),
                        "invert_lateral":       arm.invert_lateral,
                        "confidence":           arm.cal_confidence,
                        "wrist_flex_sign":      arm.wrist_flex_sign,
                        "wrist_roll_sign":      arm.wrist_roll_sign,
                        "controller_anchor_T": (
                            arm.controller_anchor_T.round(4).tolist()
                            if arm.controller_anchor_T is not None else None
                        ),
                        "robot_anchor_T": arm.robot_anchor_T.round(4).tolist(),
                        "mapping_dry_run": self._mapping_dry_run_status(arm),
                        "diagnostics":          dict(arm.last_diag),
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
                "episodes_saved": (rec.episode_count if rec else self._episodes_saved),
                "frames_in_current_episode": rec.frame_count_in_episode if rec else 0,
                "last_episode_index": (
                    rec.last_saved_episode_index if rec else self._last_saved_episode_index
                ),
                "last_episode_frames": (
                    rec.last_saved_episode_frames if rec else self._last_saved_episode_frames
                ),
                "repo_id": rec.repo_id if rec else self._recording_repo_id,
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
                "dual_mode": self._dual_mode,
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
            return out

    def _mapping_dry_run_status(self, arm: _PerArm) -> dict[str, Any]:
        """Synthetic, non-driving mapping checks shown in status for calibration review."""
        M = arm.session_vr_to_robot

        def map_vec(vec: tuple[float, float, float]) -> list[float]:
            out = M @ _np.array(vec, dtype=float)
            if arm.invert_lateral:
                out = out * _np.array([1.0, -1.0, 1.0])
            return [float(v) for v in out]

        pitch_rot = M @ _np.array([0.0, 0.1, 0.0], dtype=float)
        roll_rot = M @ _np.array([0.1, 0.0, 0.0], dtype=float)
        if arm.invert_lateral:
            pitch_rot = pitch_rot * _np.array([-1.0, -1.0, 1.0])
            roll_rot = roll_rot * _np.array([-1.0, -1.0, 1.0])
        backward_target = _clamp_target_position(
            _np.array(arm.anchor_ee_pos) + _np.array(map_vec((0.0, 0.0, 0.10)))
        )
        return {
            "vr_forward_to_robot": map_vec((0.0, 0.0, -0.10)),
            "vr_backward_target_clamped": [float(v) for v in backward_target],
            "vr_up_to_robot": map_vec((0.0, 0.10, 0.0)),
            "vr_left_to_robot": map_vec((-0.10, 0.0, 0.0)),
            "wrist_pitch_sign_probe_deg": float(arm.wrist_flex_sign * math.degrees(float(pitch_rot[1]))),
            "wrist_roll_sign_probe_deg": float(arm.wrist_roll_sign * math.degrees(float(roll_rot[0]))),
        }

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
                cp = getattr(goal, "vr_ctrl_position", None)
                rot = getattr(goal, "vr_ctrl_rotation", None)
                trig = bool(getattr(goal, "trigger", False))
                thumb = getattr(goal, "thumbstick", None) or {}
                btn  = getattr(goal, "buttons", None) or {}
                with self._lock:
                    cur_buttons = {str(k): bool(v) for k, v in btn.items()}
                    prev_buttons = arm.prev_buttons
                    arm.prev_buttons = dict(cur_buttons)
                    # Latch reset edges so a fast-following 'position' goal can't
                    # overwrite the calibration trigger before the drive loop ticks.
                    if str(mode) == "reset":
                        arm.reset_pending = True
                    arm.latest = _LatestGoal(
                        received_at=time.time(),
                        has_data=True,
                        mode=str(mode),
                        rel_position=tuple(float(v) for v in (rp if rp is not None else (0, 0, 0))),
                        rel_rotvec=tuple(float(v) for v in (rr if rr is not None else (0, 0, 0))),
                        controller_position=(tuple(float(v) for v in cp) if cp is not None else None),
                        rotation_quat=(tuple(float(v) for v in rot.as_quat()) if rot is not None and hasattr(rot, "as_quat") else None),
                        trigger=trig,
                        thumbstick=(float(thumb.get("x", 0)), float(thumb.get("y", 0))),
                        buttons=cur_buttons,
                    )
                    # Advance the calibration state machine if active for this arm.
                    # Must hold `self._lock` (still held inside this `with` block).
                    if arm.cal_state != "idle":
                        self._advance_calibration(side, arm.latest)
                    self._handle_button_edges(side, cur_buttons, prev_buttons)
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
        arm.last_commanded_targets = dict(arm.last_sent_targets)

    def _ensure_kinematics(self, arm: _PerArm) -> bool:
        """Lazy-build the per-arm URDF kinematics handle. Called from RESET.
        Returns False when the analytical fallback should be used."""
        if arm.kinematics is not None:
            return True
        if arm.using_analytical_fallback:
            return False
        try:
            from lerobot.model import RobotKinematics
        except ImportError as e:
            log.warning("[%s] RobotKinematics unavailable; using analytical fallback: %s", arm.side, e)
            arm.using_analytical_fallback = True
            return False
        urdf = REPO_ROOT / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
        if not urdf.is_file():
            log.warning("[%s] SO101 URDF not found at %s; using analytical fallback", arm.side, urdf)
            arm.using_analytical_fallback = True
            return False
        arm.kinematics = RobotKinematics(
            urdf_path=str(urdf),
            target_frame_name="gripper_frame_link",
            joint_names=list(_IK_JOINT_ORDER),
        )
        log.info("[%s] URDF kinematics initialized (%s)", arm.side, urdf.name)
        return True

    def _capture_anchor(self, side: ArmSide) -> None:
        """RESET handler for ONE arm. Snapshot the EE pose via URDF FK; this
        becomes the anchor for the absolute-pose IK pipeline. Resets the target
        filters + delta accumulator; rebuilds the per-session VR→robot frame.
        """
        if not MOTORS.is_connected(side):
            return
        arm = self._arms[side]
        use_urdf = self._ensure_kinematics(arm)

        present = MOTORS.read_positions(side)
        prefix = f"{side}_arm_"
        get = lambda j: float(present.get(f"{prefix}{j}", 0.0))

        q_now_deg = _np.array([get(j) for j in _IK_JOINT_ORDER], dtype=float)
        if use_urdf:
            # URDF FK at current joints → 4×4 EE pose. Far more accurate than the
            # planar analytical FK (which assumes wrist is at gripper origin).
            T_now = arm.kinematics.forward_kinematics(q_now_deg)
        else:
            from scipy.spatial.transform import Rotation as _R
            reach, z = self._analytical_kin.forward(get("shoulder_lift"), get("elbow_flex"))
            pan_rad = math.radians(get("shoulder_pan"))
            T_now = _np.eye(4)
            T_now[:3, 3] = (
                reach * math.cos(pan_rad),
                reach * math.sin(pan_rad),
                z,
            )
            pitch = get("shoulder_lift") + get("elbow_flex") + get("wrist_flex")
            T_now[:3, :3] = _R.from_euler(
                "zyx",
                [get("shoulder_pan"), pitch, get("wrist_roll")],
                degrees=True,
            ).as_matrix()

        arm.target_T = T_now.copy()
        arm.robot_anchor_T = T_now.copy()
        arm.anchor_ee_pos = (float(T_now[0, 3]), float(T_now[1, 3]), float(T_now[2, 3]))
        arm.anchor_R_robot = T_now[:3, :3].copy()
        arm.target_R_robot = T_now[:3, :3].copy()
        arm.offset_robot = (0.0, 0.0, 0.0)
        arm.smoothed_R_target = T_now[:3, :3].copy()
        arm.last_q_sol = q_now_deg.copy()
        arm.last_q_filtered = q_now_deg.copy()

        # Keep the translation frame stable. Rebuilding `session_vr_to_robot`
        # from the controller quaternion on every RESET makes hand translation
        # depend on wrist pose at anchor time, so arm motion feels random. Use the
        # saved/wizard/default frame for position; the quaternion is still stored
        # below as the wrist-orientation anchor.
        ctrl_quat = arm.latest.rotation_quat if arm.latest.has_data else None
        ctrl_pos = arm.latest.controller_position if arm.latest.has_data else None
        if ctrl_quat is None or ctrl_pos is None:
            arm.controller_anchor_T = None
            log.warning(
                "[%s] missing controller pose in RESET goal; SE(3) mapping disabled "
                "until next grip-press with full controller state",
                side,
            )
        else:
            arm.controller_anchor_T = _pose_matrix_from_vr(ctrl_pos, ctrl_quat)
        log.info("[%s] keeping VR→robot translation frame:\n%s", side, arm.session_vr_to_robot.round(3))

        arm.anchor = _AnchorPose(
            ee_x=float(T_now[0, 3]),
            ee_y=float(T_now[1, 3]),
            pan_deg=get("shoulder_pan"),
            shoulder_lift_deg=get("shoulder_lift"),
            elbow_flex_deg=get("elbow_flex"),
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
            gripper=get("gripper"),
        )
        arm.last_sent_targets = arm.targets.to_dict_with_prefix(side)
        arm.last_commanded_targets = dict(arm.last_sent_targets)
        arm.calibrated = True
        log.info("[%s] VR anchor: EE=(%.3f, %.3f, %.3f) m (%s FK)",
                 side, T_now[0, 3], T_now[1, 3], T_now[2, 3],
                 "URDF" if use_urdf else "analytical")

    def _drive_loop(self) -> None:
        """Per-tick: process RESETs for ALL connected arms (so each arm's anchor
        is up-to-date). Command motion on the active arm, or both arms when
        dual mode is enabled. The target/IK/control math remains per-arm."""
        next_tick = time.monotonic()
        while not self._stop_evt.is_set():
            now = time.time()
            try:
                with self._lock:
                    engaged = self._engaged
                    active = self._active_arm
                    dual_mode = self._dual_mode
                    scale = self._scale
                    connected = list(MOTORS.connected_sides)
                self._last_drive_tick = now

                if not connected:
                    next_tick = self._sleep_until(next_tick)
                    continue

                # Phase 1: handle RESET / IDLE / drain accumulator
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
                        sent = MOTORS.send_action(side, final)
                        arm.last_sent_targets = dict(sent)
                        arm.last_commanded_targets = dict(sent)
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

                # Commands sent during this tick become the dataset action for
                # the same tick. Arms not commanded fall back to present state.
                commanded_this_tick: dict[ArmSide, dict[str, float]] = {}

                # Phase 2: command the active arm if engaged + calibrated. In
                # dual mode, run the same per-arm path for both connected arms.
                if dual_mode:
                    drive_sides = [s for s in ("left", "right") if s in connected]
                elif active is not None and active in connected:
                    drive_sides = [active]
                else:
                    drive_sides = []
                if not engaged or not drive_sides:
                    # Still record idle/passive ticks; they become hold/no-op
                    # frames with action equal to present state.
                    self._record_frame_if_active(commanded_this_tick=commanded_this_tick)
                    next_tick = self._sleep_until(next_tick)
                    continue
                for drive_side in drive_sides:
                    # Don't VR-drive an arm that's currently homing — homing already
                    # owns send_action above.
                    if self._arms[drive_side].homing:
                        continue
                    # Don't VR-drive an arm whose torque is released (user is
                    # hand-posing it).
                    if not MOTORS.is_torque_enabled(drive_side):
                        continue

                    arm = self._arms[drive_side]
                    with self._lock:
                        goal = arm.latest

                    # Watchdog: skip if last goal too stale (controller put down).
                    goal_age = now - goal.received_at if goal.has_data else 1e9
                    if not goal.has_data or goal_age > GOAL_SKIP_AGE_S:
                        with self._lock:
                            if arm.stale_since is None:
                                arm.stale_since = now
                            if now - arm.stale_since > 1.0:
                                self._engaged = False
                                self._active_arm = None
                                self._dual_mode = False
                                log.warning("[%s] VR goals stale for >1s; auto-disengaged", drive_side)
                        continue
                    arm.stale_since = None
                    if goal.mode != "position":
                        # Grip release sends IDLE: hold last commanded targets; the
                        # per-tick joint cap below already prevents drift.
                        continue
                    if not arm.calibrated:
                        continue

                    # Build joint targets from the latest VR controller pose.
                    self._compute_targets_from_vr(drive_side, goal, scale)

                    # Per-tick joint clamp vs last sent (caps max joint velocity).
                    prefix = f"{drive_side}_arm_"
                    raw = arm.targets.to_dict_with_prefix(drive_side)
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
                        present_full = MOTORS.read_positions(drive_side)
                        final = {}
                        for pj, target in clamped.items():
                            here = present_full.get(pj, target)
                            final[pj] = here + KP * (target - here)

                    sent = MOTORS.send_action(drive_side, final)
                    arm.last_sent_targets = dict(sent)
                    arm.last_commanded_targets = dict(sent)
                    commanded_this_tick[drive_side] = dict(sent)

                    # Debug: per-arm gripper trigger/target/sent/present log (1Hz).
                    self._debug_log_gripper(drive_side, goal, arm.targets,
                                             final, present_full, now)

                # Dataset capture: one frame per drive tick when recording is on.
                # Capture after motor writes so `action` is the command from this
                # tick, not the previous tick.
                self._record_frame_if_active(commanded_this_tick=commanded_this_tick)

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
               → idle (matrix + lateral verified)

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

        Runtime wrist signs (wrist_flex_sign / wrist_roll_sign) are NOT part of
        this wizard — they are derived per session from the session matrix
        combined with the hardware motor polarity in
        `config/xlerobot.yaml` under `vr.wrist_motor_polarity.{left,right}.{flex,roll}`.
        Flip polarity there if you rewire/remount a wrist motor; recalibrating
        translation here is enough to keep the wrist tracking the user even
        when you face a different direction.

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
            arm.cal_anchor_quat_for_wrist = None
            arm.cal_wrist_release_quat = None
            arm.cal_wrist_verify_deg = 0.0
            # Discard any prior empirical wrist pitch canonical — the user
            # is re-running the wizard. They can either re-capture in step 4
            # or press 'Skip wrist verify' to fall back to the WebXR default.
            arm.wrist_pitch_canonical = None
            arm.cal_validation = {}
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
            arm.cal_anchor_quat_for_wrist = None
            arm.cal_wrist_release_quat = None
            arm.cal_wrist_verify_deg = 0.0
            arm.cal_validation = {}
            # Restore everything (matrix, lateral, polarity-derived signs, and
            # any persisted empirical wrist canonical) from disk.
            self._restore_persisted_arm_config(side)
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
            elif state == "awaiting_anchor_wrist_verify":
                anchor_q = goal.rotation_quat
                if anchor_q is None:
                    log.warning(
                        "[%s] cal: wrist-verify reset has no controller quaternion; "
                        "release grip and try again (or press 'Skip wrist verify')",
                        side,
                    )
                    return
                arm.cal_anchor_quat_for_wrist = anchor_q
                arm.cal_wrist_release_quat = None
                arm.cal_wrist_verify_deg = 0.0
                arm.cal_state = "motioning_wrist_verify"
                log.info(
                    "[%s] cal: wrist-verify anchor captured. KEEP GRIP HELD and pitch "
                    "your wrist clearly UP (rotate palm toward ceiling) ~20-45°, "
                    "then release. Or press 'Skip wrist verify' to use the WebXR default.",
                    side,
                )
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
        if mode == "position" and state == "motioning_wrist_verify":
            release_q = goal.rotation_quat
            anchor_q = arm.cal_anchor_quat_for_wrist
            if release_q is not None:
                arm.cal_wrist_release_quat = release_q
            if anchor_q is not None and release_q is not None:
                arm.cal_wrist_verify_deg = _wrist_rotation_deg_since_anchor(
                    anchor_q, release_q,
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
                # Build a preliminary 2-motion matrix so the UI's diagnostics
                # (and the upcoming lateral-check arrow if anyone looks at it)
                # have something sensible to show. The FINAL matrix is rebuilt
                # in _finalize_translation_calibration from all three motions
                # via Procrustes for better noise rejection.
                f = arm.cal_captured_fwd
                u = arm.cal_captured_up
                if f is not None and u is not None:
                    arm.session_vr_to_robot, arm.cal_confidence = (
                        _compute_session_frame_from_two_motions(f, u)
                    )
                arm.cal_state = "awaiting_anchor_left"
                log.info("[%s] cal: up axis captured (%.1f cm); preliminary matrix built. "
                         "Now press grip and move hand LEFT ~10 cm — the final matrix "
                         "is built from all three motions when this completes.",
                         side, mag * 100)
            elif state == "motioning_left":
                mag = math.sqrt(sum(v * v for v in arm.cal_motion_acc))
                if mag < CALIBRATION_MIN_MOTION_M:
                    log.warning("[%s] cal: left motion too small (%.1f cm); "
                                "re-grip and move further", side, mag * 100)
                    arm.cal_state = "awaiting_anchor_left"
                    return
                arm.cal_captured_left = arm.cal_motion_acc
                arm.cal_last_left_m = mag
                self._finalize_translation_calibration(side)
            elif state == "motioning_wrist_verify":
                self._finalize_wrist_verify(side, goal)

    def _finalize_translation_calibration(self, side: ArmSide) -> None:
        """Rebuild the session matrix from all three captured motions via
        Procrustes/Kabsch, verify the lateral axis sign, then transition to the
        wrist-verify step. Called WITH `self._lock` held."""
        arm = self._arms[side]
        f = arm.cal_captured_fwd
        u = arm.cal_captured_up
        l = arm.cal_captured_left
        if f is None or u is None or l is None:
            log.warning("[%s] cal: finalize called without all three vectors", side)
            arm.cal_state = "idle"
            return

        # Rebuild M using ALL three captured motions. This averages out noise
        # in any single motion (small off-axis drift, hand jitter) far better
        # than the 2-motion Gram-Schmidt path. Falls back to the 2-motion
        # build internally if one motion is degenerate.
        arm.session_vr_to_robot, arm.cal_confidence = (
            _compute_session_frame_from_three_motions(f, u, l)
        )

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
        f_robot = arm.session_vr_to_robot @ _np.array(f, dtype=float)
        u_robot = arm.session_vr_to_robot @ _np.array(u, dtype=float)
        arm.cal_validation = {
            "forward_robot_delta": [float(v) for v in f_robot],
            "up_robot_delta": [float(v) for v in u_robot],
            "left_robot_delta": [float(v) for v in l_robot],
            "lateral_verdict": verdict,
        }
        log.info(
            "[%s] translation calibration finalized (Procrustes M, conf=%s) — "
            "forward=%s up=%s left=%s → robot deltas fwd=%s up=%s left=%s → lateral %s\n"
            "session matrix:\n%s",
            side, arm.cal_confidence,
            tuple(f"{v:.3f}" for v in f),
            tuple(f"{v:.3f}" for v in u),
            tuple(f"{v:.3f}" for v in l),
            tuple(f"{v:.3f}" for v in f_robot),
            tuple(f"{v:.3f}" for v in u_robot),
            tuple(f"{v:.3f}" for v in l_robot),
            verdict,
            arm.session_vr_to_robot.round(3),
        )

        # Hand off to wrist-verify step. The user can either do one clean
        # pitch-up motion to capture the empirical canonical (most robust),
        # or skip via the UI button (analytical canonical from WebXR is then
        # used; works for standard Quest controllers).
        arm.cal_anchor_quat_for_wrist = None
        arm.cal_state = "awaiting_anchor_wrist_verify"
        log.info(
            "[%s] Translation done. Next (OPTIONAL): squeeze grip, pitch wrist UP "
            "(rotate forearm so palm rotates from level → up), release. Or press "
            "'Skip wrist verify' to finish with the WebXR analytical canonical.",
            side,
        )

    def _finalize_wrist_verify(self, side: ArmSide, goal: _LatestGoal) -> None:
        """Step 4 of the wizard: empirically capture the user's "pitch up" axis
        in controller anchor-local frame, then bake it into the wrist sign
        derivation. Called WITH `self._lock` held.

        The user squeezed grip (snapshot of `arm.cal_anchor_quat_for_wrist`),
        pitched their wrist UP, and released. We compute the relative rotation
        of the controller from anchor to release — that vector, in anchor-local
        frame, IS the empirical "pitch up canonical". Stored side-handedness-
        applied, ready for direct use by `_derive_wrist_signs_from_session`.
        """
        arm = self._arms[side]
        anchor_q = arm.cal_anchor_quat_for_wrist
        release_q = goal.rotation_quat or arm.cal_wrist_release_quat
        if anchor_q is None or release_q is None:
            log.warning(
                "[%s] cal: wrist-verify missing controller quaternion (anchor=%s, "
                "release=%s); re-grip and pitch wrist again, or press 'Skip'.",
                side, anchor_q is not None, release_q is not None,
            )
            arm.cal_state = "awaiting_anchor_wrist_verify"
            return

        R_anchor = _R.from_quat(anchor_q)
        R_release = _R.from_quat(release_q)
        # Relative rotation expressed in the anchor's local frame.
        R_rel = R_anchor.inv() * R_release
        rotvec = _np.asarray(R_rel.as_rotvec(), dtype=float)
        mag = float(_np.linalg.norm(rotvec))
        arm.cal_wrist_verify_deg = math.degrees(mag)
        if mag < math.radians(WRIST_VERIFY_MIN_DEG):
            log.warning(
                "[%s] cal: wrist-verify motion too small (%.1f°, need ≥%.0f°). "
                "Re-grip and pitch wrist further, or press 'Skip wrist verify' "
                "to use the WebXR analytical default.",
                side, math.degrees(mag), WRIST_VERIFY_MIN_DEG,
            )
            arm.cal_state = "awaiting_anchor_wrist_verify"
            arm.cal_anchor_quat_for_wrist = None
            arm.cal_wrist_release_quat = None
            arm.cal_wrist_verify_deg = 0.0
            return

        # Apply side handedness to match `_controller_rotation_delta_for_side`
        # at runtime (left = transpose ≈ negate-rotvec for unit-axis motions).
        canonical = rotvec / mag
        if side == "left":
            canonical = -canonical

        arm.wrist_pitch_canonical = tuple(float(v) for v in canonical)
        arm.cal_validation["wrist_pitch_anchor_local"] = list(arm.wrist_pitch_canonical)
        arm.cal_validation["wrist_verify_magnitude_deg"] = math.degrees(mag)

        # Compare with the WebXR analytical default to surface any mismatch.
        analytical = _np.array([1.0, 0.0, 0.0])  # both sides, BEFORE side flip
        # canonical above is post-side-flip; un-apply for comparison
        measured_pre_flip = -_np.asarray(canonical) if side == "left" else _np.asarray(canonical)
        cos = float(_np.dot(measured_pre_flip, analytical))
        cos_clamped = max(-1.0, min(1.0, cos))
        off_deg = math.degrees(math.acos(cos_clamped))
        arm.cal_validation["wrist_verify_off_from_webxr_default_deg"] = off_deg
        if cos >= 0.85:
            log.info(
                "[%s] wrist-verify: empirical pitch canonical %s (anchor-local, "
                "post-side-flip). Matches WebXR analytical default within %.0f° "
                "→ wrist mapping confirmed.",
                side, tuple(f"{v:+.3f}" for v in canonical), off_deg,
            )
        else:
            log.warning(
                "[%s] wrist-verify: empirical pitch canonical %s is %.0f° off from "
                "the WebXR analytical default ([+1, 0, 0]). This means your "
                "controller's anchor-local pitch axis is NOT what WebXR docs "
                "suggest — the empirical value will be used at runtime and "
                "wrist mapping will track your actual motion correctly.",
                side, tuple(f"{v:+.3f}" for v in canonical), off_deg,
            )
        arm.cal_anchor_quat_for_wrist = None
        arm.cal_wrist_release_quat = None
        self._persist_final_calibration(side)

    def skip_wrist_verify(self, side: ArmSide) -> dict:
        """Skip the optional step-4 wrist-verify motion and finish calibration
        using the WebXR analytical canonical for the wrist pitch axis.

        Safe to call any time the wizard is in the wrist-verify substep; no-op
        otherwise. The user can always re-run the full wizard later to capture
        an empirical canonical if wrist tracking turns out to be off."""
        with self._lock:
            arm = self._arms[side]
            if arm.cal_state not in ("awaiting_anchor_wrist_verify",
                                     "motioning_wrist_verify"):
                return self.status()
            log.info(
                "[%s] wrist-verify skipped — falling back to WebXR analytical "
                "canonical for wrist sign derivation.", side,
            )
            arm.wrist_pitch_canonical = None
            arm.cal_anchor_quat_for_wrist = None
            arm.cal_wrist_release_quat = None
            arm.cal_wrist_verify_deg = 0.0
            self._persist_final_calibration(side)
        return self.status()

    def _persist_final_calibration(self, side: ArmSide) -> None:
        arm = self._arms[side]
        # Persist to disk so subsequent sessions don't need to re-run the wizard.
        # Runtime `wrist_flex_sign` / `wrist_roll_sign` are NOT persisted — they
        # are derived per session from `session_vr_to_robot` + the hardware
        # `vr.wrist_motor_polarity` block in xlerobot.yaml. The empirical
        # pitch canonical IS persisted when captured (overrides the WebXR
        # default in `_derive_wrist_signs_from_session`).
        try:
            _vrcal.write_for_arm(
                side, arm.session_vr_to_robot,
                forward_motion_m=arm.cal_last_fwd_m,
                up_motion_m=arm.cal_last_up_m,
                left_motion_m=arm.cal_last_left_m,
                invert_lateral=arm.invert_lateral,
                confidence=arm.cal_confidence,
                wrist_pitch_anchor_local=arm.wrist_pitch_canonical,
            )
        except Exception as e:
            log.warning("[%s] could not persist VR calibration: %s", side, e)
        # The session matrix (and possibly the empirical canonical) just
        # changed — re-derive the runtime wrist signs so the very next teleop
        # frame uses the updated mapping.
        polarity = _WRIST_MOTOR_POLARITY.get(side, {"flex": -1.0, "roll": -1.0})
        arm.wrist_flex_sign, arm.wrist_roll_sign = _derive_wrist_signs_from_session(
            side, arm.session_vr_to_robot, polarity,
            pitch_canonical=arm.wrist_pitch_canonical,
        )
        arm.cal_state = "idle"
        arm.cal_motion_acc = (0.0, 0.0, 0.0)
        canon_label = "empirical" if arm.wrist_pitch_canonical is not None else "analytical"
        log.info(
            "[%s] calibration COMPLETE — confidence=%s invert_lateral=%s "
            "derived wrist_signs=(flex %+.0f, roll %+.0f) from polarity=(flex "
            "%+.0f, roll %+.0f) using %s pitch canonical. "
            "Squeeze grip again to anchor for teleop.",
            side, arm.cal_confidence, arm.invert_lateral,
            arm.wrist_flex_sign, arm.wrist_roll_sign,
            polarity["flex"], polarity["roll"], canon_label,
        )
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
                if self._arms[side].cal_confidence in ("poor", "legacy"):
                    log.warning("[%s] engage switch ignored: VR calibration confidence is %s; rerun calibration",
                                side, self._arms[side].cal_confidence)
                    return
                self._active_arm = side
                arm = self._arms[side]
                arm.stale_since = None
                log.info("[%s] engage button → SWITCHED active arm to %s", side, side)
            else:
                if self._arms[side].cal_confidence in ("poor", "legacy"):
                    log.warning("[%s] engage button ignored: VR calibration confidence is %s; rerun calibration",
                                side, self._arms[side].cal_confidence)
                    return
                self._engaged = True
                self._active_arm = side
                arm = self._arms[side]
                arm.stale_since = None
                log.info("[%s] engage button → ENGAGED on %s arm", side, side)

    def _handle_button_edges(
        self,
        side: ArmSide,
        cur_btn: dict[str, bool],
        prev_btn: dict[str, bool],
    ) -> None:
        """Handle Quest face-button edges immediately as goals arrive.

        This avoids sampling races from the 30 Hz drive loop where a quick tap
        could be overwritten by a later goal before the drive loop saw it. Y is
        intentionally a clean, single-button left-controller press: X+Y chords
        are ignored so dual mode cannot accidentally fight single-arm engage.
        """
        pressed_edges = {
            name for name, pressed in cur_btn.items()
            if pressed and not prev_btn.get(name, False)
        }
        if not pressed_edges:
            return

        dual_btn = DUAL_MODE_BUTTON_BY_SIDE.get(side)
        if dual_btn and dual_btn in pressed_edges:
            other_face_held = any(
                cur_btn.get(name, False)
                for name in ("X", "Y")
                if name != dual_btn
            )
            if other_face_held:
                log.info("[%s] %s ignored for dual mode because another face button is held", side, dual_btn)
                return
            if self._arms[side].cal_state != "idle":
                log.info("[%s] %s ignored for dual mode during calibration", side, dual_btn)
                return
            self._handle_dual_mode_button(side)
            return

        # B edge (right controller only) → recording toggle.
        record_btn = RECORD_BUTTON_BY_SIDE.get(side)
        if record_btn and record_btn in pressed_edges:
            self._handle_record_button(side)

        # A/X edge → single-arm engage toggle.
        engage_btn = ENGAGE_BUTTON_BY_SIDE.get(side)
        if engage_btn and engage_btn in pressed_edges:
            if self._arms[side].cal_state != "idle":
                log.info("[%s] %s ignored for engage during calibration", side, engage_btn)
                return
            self._handle_engage_button(side)

    def _handle_dual_mode_button(self, side: ArmSide) -> None:
        """Y button on the left controller toggles dual-arm drive mode.

        This only changes enablement/selection: when dual mode is on, the drive
        loop runs the existing per-arm target/IK/control path once for left and
        once for right. It does not alter the IK, smoothing, wrist mapping, or
        per-tick motor control.
        """
        with self._lock:
            if side != "left":
                return
            connected = set(MOTORS.connected_sides)
            if {"left", "right"} - connected:
                log.warning("[left] Y dual-mode toggle ignored: connect both arms first")
                return
            if self._dual_mode:
                self._dual_mode = False
                self._engaged = False
                log.info("[left] Y button → DUAL MODE OFF (disengaged)")
                return
            for s in ("left", "right"):
                arm = self._arms[s]
                if arm.cal_confidence in ("poor", "legacy"):
                    log.warning(
                        "[left] Y dual-mode toggle ignored: %s calibration confidence is %s; rerun calibration",
                        s, arm.cal_confidence,
                    )
                    return
                arm.stale_since = None
            self._dual_mode = True
            self._engaged = True
            self._active_arm = None
            log.info("[left] Y button → DUAL MODE ON (left + right arms)")

    def _handle_record_button(self, side: ArmSide) -> None:
        """B button on right controller was just pressed → toggle dataset recording."""
        log.info("[%s] B button → toggle recording", side)
        self.set_recording(not self._recording)

    def set_recording_task(self, task: str) -> dict:
        """Cache the UI task text for future B-button recording starts."""
        with self._lock:
            self._last_task = (task or "").strip()
        return self.status()

    def delete_last_recorded_episode(self) -> dict:
        """Delete the most recently saved episode so operators can retry."""
        with self._recording_transition_lock:
            with self._lock:
                if self._recording:
                    self._last_error = "stop recording before deleting the last episode"
                    return self.status()
                rec = self._recorder
                self._recorder = None
            if rec is not None:
                try:
                    rec.finalize()
                except Exception as e:
                    log.warning("delete-last: finalize before delete failed: %s", e)
            try:
                cfg = _dataset.load_dataset_config()
                effective_root = self._last_dataset_root or _dataset.resolve_root(
                    cfg.get("root"),
                    str(cfg["repo_id"]),
                )
                new_total, resolved_root = _dataset.delete_last_episode(
                    repo_id=str(cfg["repo_id"]),
                    root=effective_root,
                )
                with self._lock:
                    self._episodes_saved = new_total
                    self._last_dataset_root = resolved_root
                    idx, frames = _dataset.last_episode_summary(
                        repo_id=str(cfg["repo_id"]),
                        root=resolved_root,
                    )
                    self._last_saved_episode_index = idx
                    self._last_saved_episode_frames = frames
                    self._last_error = None
            except Exception as e:
                with self._lock:
                    self._last_error = f"delete last episode failed: {e}"
            return self.status()

    def set_recording(self, enabled: bool, task: str = "",
                       home_first: Optional[bool] = None,
                       root: Optional[str] = None) -> bool:
        with self._recording_transition_lock:
            return self._set_recording_locked(enabled, task=task, home_first=home_first, root=root)

    def _set_recording_locked(self, enabled: bool, task: str = "",
                              home_first: Optional[bool] = None,
                              root: Optional[str] = None) -> bool:
        """Idempotent recording toggle. Lazy-creates the LeRobotDataset on first
        start, opens a new episode each ON transition, saves the episode on the
        OFF transition. Returns the new recording state.

        `home_first`: if True (or None and `dataset.home_before_episode: true`
        in config/xlerobot.yaml), move every connected arm to its saved home
        pose before opening the new episode. Ensures consistent training data.
        """
        effective_task = (task or "").strip() or self._last_task
        if enabled and not effective_task:
            self._last_error = "task description required before starting an episode"
            log.warning(self._last_error)
            return self._recording

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
                        self._recording_repo_id = str(cfg["repo_id"])
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
                self._episodes_saved = self._recorder.episode_count
                self._last_saved_episode_index = getattr(self._recorder, "last_saved_episode_index", None)
                self._last_saved_episode_frames = int(getattr(self._recorder, "last_saved_episode_frames", 0))
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
                with self._lock:
                    self._episodes_saved = rec.episode_count
                    self._last_saved_episode_index = getattr(rec, "last_saved_episode_index", None)
                    self._last_saved_episode_frames = int(getattr(rec, "last_saved_episode_frames", 0))
                rec.finalize()
                with self._lock:
                    if self._recorder is rec:
                        self._recorder = None
        return self._recording

    def _record_frame_if_active(
        self,
        commanded_this_tick: Optional[dict[ArmSide, dict[str, float]]] = None,
    ) -> None:
        """If recording is on and an episode is active, append one frame.

        Action = same-tick commanded joint positions for arms moved this tick,
        with passive arms falling back to present joint positions.
        Observation.state = present joint positions (both arms).
        Observation.images.<role> = latest snapshot from each configured camera.
        Missing arm or camera data is filled with zeros by the recorder.
        """
        with self._lock:
            rec = self._recorder
            if not (self._recording and rec is not None and rec.in_episode):
                return
            # Snapshot dictionaries while holding the lock; release before doing
            # camera capture (which is slow).
            connected = list(MOTORS.connected_sides)
            commanded_by_side = {
                s: dict(commanded)
                for s, commanded in (commanded_this_tick or {}).items()
            }
        # Outside lock: read present positions (bus I/O) + camera snapshots.
        try:
            present_dict = MOTORS.read_positions()
        except Exception as e:
            log.warning("record: read_positions failed: %s", e)
            present_dict = {}
        action_dict: dict[str, float] = {}
        for s in connected:
            prefix = f"{s}_arm_"
            if commanded_by_side.get(s):
                action_dict.update(commanded_by_side[s])
            else:
                for j in _motors.JOINTS_PER_ARM:
                    key = f"{prefix}{j}"
                    if key in present_dict:
                        action_dict[key] = float(present_dict[key])
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
                                  scale: float) -> None:
        """Convert the latest VR controller pose → joint targets via URDF IK.

        Pipeline (per-tick):
          1. Build the reset-relative SE(3) controller transform from the
             absolute controller pose carried in the goal.
          2. Map the VR-world hand displacement (current − anchor) through
             `arm.session_vr_to_robot` into robot-frame translation; LERP-smooth
             this into `arm.offset_robot` so the cartesian target chases the
             hand exponentially without a hard rate cap.
          3. Map the controller-relative rotation through the same calibration,
             apply per-hand handedness correction, and SLERP-smooth toward the
             robot-frame anchor-times-relative-rotation target.
          4. Clamp the resulting EE position to the workspace box + reach sphere
             + rear-singularity guard, then reconcile `arm.offset_robot` so a
             clamped step doesn't accumulate hidden motion debt.
          5. URDF/placo IK (position-only) for pan/lift/elbow; large jumps in
             the IK solution are rejected and the previous solution reused.
          6. Wrist flex/roll come directly from the smoothed orientation delta
             (no IK, no EMA) so they stay snappy.
          7. The per-tick joint cap in the drive loop is the only motor-rate
             limiter — there is no joint-level EMA.

        SE(3) mapping requires `arm.controller_anchor_T` (captured on RESET) and
        the goal to carry both `controller_position` and `rotation_quat`. If
        either is missing we leave the previous tick's targets in place; the
        per-tick joint cap holds the robot still until valid input returns.
        """
        from scipy.spatial.transform import Rotation as _R
        arm = self._arms[side]
        M = arm.session_vr_to_robot
        current_pos = goal.controller_position
        current_q = goal.rotation_quat
        wrist_cap = WRIST_RAD_DELTA_LIMIT * scale

        if arm.controller_anchor_T is None or current_pos is None or current_q is None:
            log.warning(
                "[%s] SE(3) inputs missing (anchor=%s current_pos=%s current_q=%s); holding targets",
                side,
                arm.controller_anchor_T is not None,
                current_pos is not None,
                current_q is not None,
            )
            return

        try:
            controller_current_T = _pose_matrix_from_vr(current_pos, current_q)
            controller_rel_T = _np.linalg.solve(arm.controller_anchor_T, controller_current_T)
        except Exception as e:
            log.warning("[%s] SE(3) controller mapping failed (%s); holding targets", side, e)
            return

        # Translation calibration is learned from VR-world hand motions
        # (forward/up/left), so use reset-relative world displacement for
        # position. The homogeneous relative transform is still used for
        # rotation, where controller-local handedness matters.
        raw_vr_delta = _np.array(current_pos, dtype=float) - arm.controller_anchor_T[:3, 3]
        dp_robot = M @ raw_vr_delta
        if arm.invert_lateral:
            dp_robot = dp_robot * _np.array([1.0, -1.0, 1.0])
        desired_offset = dp_robot * scale
        current_offset = _np.array(arm.offset_robot, dtype=float)
        # BEAVR-style LERP smoothing: cartesian offset tracks the desired offset
        # exponentially. POS_EMA_ALPHA controls how aggressively the robot
        # target chases the hand. The per-joint cap downstream is the real
        # motor-rate safety; we only keep a glitch cap here so a controller
        # tracking jump cannot teleport the IK seed.
        pos_alpha = max(0.05, min(1.0, POS_EMA_ALPHA))
        new_offset = (1.0 - pos_alpha) * current_offset + pos_alpha * desired_offset
        step = new_offset - current_offset
        step_norm = float(_np.linalg.norm(step))
        GLITCH_CAP_M = 0.05
        if step_norm > GLITCH_CAP_M:
            new_offset = current_offset + step * (GLITCH_CAP_M / step_norm)
        arm.offset_robot = (float(new_offset[0]), float(new_offset[1]), float(new_offset[2]))

        R_delta_vr = _controller_rotation_delta_for_side(side, controller_rel_T[:3, :3])
        R_delta_robot = _project_to_rotation_matrix(M @ R_delta_vr @ M.T)
        if arm.invert_lateral:
            D = _np.diag([1.0, -1.0, 1.0])
            R_delta_robot = _project_to_rotation_matrix(D @ R_delta_robot @ D)

        # BEAVR-style reset-relative target: robot anchor pose post-multiplied by
        # the mapped controller-relative rotation.
        R_raw = _project_to_rotation_matrix(arm.anchor_R_robot @ R_delta_robot)
        raw_step = (
            _R.from_matrix(R_raw)
            * _R.from_matrix(arm.smoothed_R_target).inv()
        ).magnitude()
        if raw_step >= ROTVEC_DEADBAND_RAD:
            arm.smoothed_R_target = _slerp_rotation_matrix(
                arm.smoothed_R_target,
                R_raw,
                ORI_EMA_ALPHA,
                max_step_rad=wrist_cap,
            )
        arm.target_R_robot = _project_to_rotation_matrix(arm.smoothed_R_target)

        R_wrist_delta = _project_to_rotation_matrix(arm.anchor_R_robot.T @ arm.target_R_robot)

        # Target position from anchor + offset, axis-clamped to EE_BOUNDS (sanity
        # box), then radially clamped before IK to avoid placo hopping between
        # local minima near the edge of the SO101 reach envelope. We reconcile
        # `arm.offset_robot` to the clamped position so a clamp doesn't leave
        # the LERP integrator chasing an unreachable target.
        target_pos = _clamp_target_position(_np.array([
            arm.anchor_ee_pos[0] + arm.offset_robot[0],
            arm.anchor_ee_pos[1] + arm.offset_robot[1],
            arm.anchor_ee_pos[2] + arm.offset_robot[2],
        ], dtype=float))
        tx, ty, tz = (float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
        arm.offset_robot = (
            tx - arm.anchor_ee_pos[0],
            ty - arm.anchor_ee_pos[1],
            tz - arm.anchor_ee_pos[2],
        )
        arm.target_T[:3, 3] = (tx, ty, tz)
        arm.target_T[:3, :3] = arm.target_R_robot

        # Position IK in URDF gripper-frame coordinates. We intentionally keep
        # orientation_weight=0 and only use the arm joints from the solution. The
        # URDF target frame matches the RESET anchor; the analytical planar IK
        # does not, because it models the arm linkage before the gripper offset.
        ik_mode = "urdf" if arm.kinematics is not None else "analytical"
        try:
            if arm.kinematics is not None:
                q_ik = arm.kinematics.inverse_kinematics(
                    arm.last_q_sol,
                    arm.target_T,
                    position_weight=1.0,
                    orientation_weight=0.0,
                )
            else:
                pan_deg = math.degrees(math.atan2(ty, tx))
                lift_deg, elbow_deg = self._analytical_kin.inverse(math.hypot(tx, ty), tz)
                q_ik = arm.last_q_sol.copy()
                q_ik[0] = pan_deg
                q_ik[1] = lift_deg
                q_ik[2] = elbow_deg
            q_sol = arm.last_q_sol.copy()
            q_sol[:3] = q_ik[:3]
            if not _np.all(_np.isfinite(q_sol)):
                log.warning("[%s] position IK output NaN/Inf; reusing previous q_sol", side)
                q_sol = arm.last_q_sol
            else:
                raw_jump = _np.abs(q_sol[:3] - arm.last_q_sol[:3])
                if _np.any(raw_jump > IK_JUMP_REJECT_DEG):
                    log.warning(
                        "[%s] position IK jump rejected: target_pos=(%.3f, %.3f, %.3f), dq=%s",
                        side, tx, ty, tz, tuple(f"{v:.1f}" for v in raw_jump),
                    )
                    q_sol = arm.last_q_sol.copy()
                if arm.last_q_filtered is None:
                    arm.last_q_filtered = q_sol.copy()
                # No joint-level EMA: the cartesian offset is already
                # LERP-smoothed, and the per-tick joint cap in the drive loop is
                # the motor-rate safety. Wrist joints bypass any joint smoothing
                # entirely, and arm joints now match for consistent response.
                bounds = MOTORS.bounds
                for idx, joint in enumerate(_IK_JOINT_ORDER):
                    lo, hi = bounds.get(f"{side}_arm_{joint}", (-180.0, 180.0))
                    q_sol[idx] = max(lo, min(hi, float(q_sol[idx])))
                arm.last_q_filtered = q_sol.copy()
                arm.last_q_sol = q_sol.copy()
        except Exception as e:
            log.warning("[%s] position IK failed (%s); reusing previous q_sol", side, e)
            q_sol = arm.last_q_sol

        # Build the live joint targets. Order: shoulder_pan, shoulder_lift,
        # elbow_flex, wrist_flex, wrist_roll (matches _IK_JOINT_ORDER).
        wrist_delta = _R.from_matrix(R_wrist_delta).as_rotvec()
        wrist_flex_delta_deg = arm.wrist_flex_sign * math.degrees(float(wrist_delta[1]))
        wrist_roll_delta_deg = arm.wrist_roll_sign * math.degrees(float(wrist_delta[0]))
        anchor_pitch_deg = (
            arm.anchor.shoulder_lift_deg
            + arm.anchor.elbow_flex_deg
            + arm.anchor.wrist_flex_deg
        )
        wrist_flex = (
            anchor_pitch_deg
            - float(q_sol[1])
            - float(q_sol[2])
            + wrist_flex_delta_deg
        )
        wrist_roll = arm.anchor.wrist_roll_deg + wrist_roll_delta_deg
        bounds = MOTORS.bounds
        wrist_flex_lo, wrist_flex_hi = bounds.get(f"{side}_arm_wrist_flex", (-180.0, 180.0))
        wrist_roll_lo, wrist_roll_hi = bounds.get(f"{side}_arm_wrist_roll", (-180.0, 180.0))
        wrist_flex = max(wrist_flex_lo, min(wrist_flex_hi, float(wrist_flex)))
        wrist_roll = max(wrist_roll_lo, min(wrist_roll_hi, float(wrist_roll)))
        q_seed = q_sol.copy()
        q_seed[3] = wrist_flex
        q_seed[4] = wrist_roll
        arm.last_q_sol = q_seed.copy()
        arm.last_q_filtered = q_seed.copy()
        gripper_target = self._gripper_closed if goal.trigger else self._gripper_open
        arm.targets = _LiveTargets(
            shoulder_pan=float(q_sol[0]),
            shoulder_lift=float(q_sol[1]),
            elbow_flex=float(q_sol[2]),
            wrist_flex=float(wrist_flex),
            wrist_roll=float(wrist_roll),
            gripper=gripper_target,
        )
        arm.last_diag = {
            "controller_rotation_handedness": "inverted_for_left" if side == "left" else "normal",
            "controller_position": [float(v) for v in current_pos],
            "controller_rel_translation": [float(v) for v in controller_rel_T[:3, 3]],
            "controller_world_delta": [
                float(v) for v in (_np.array(current_pos, dtype=float) - arm.controller_anchor_T[:3, 3])
            ],
            "dp_robot": [float(v) for v in dp_robot],
            "offset_robot": [float(v) for v in arm.offset_robot],
            "target_ee_pos": [tx, ty, tz],
            "target_quat_xyzw": [
                float(v) for v in _positive_quat_xyzw(_R.from_matrix(arm.target_R_robot).as_quat())
            ],
            "ik_mode": ik_mode,
            "q_arm": [float(v) for v in q_sol[:3]],
            "wrist_delta_deg": [wrist_flex_delta_deg, wrist_roll_delta_deg],
            "wrist_signs": [arm.wrist_flex_sign, arm.wrist_roll_sign],
            "using_analytical_fallback": bool(arm.using_analytical_fallback),
        }

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
