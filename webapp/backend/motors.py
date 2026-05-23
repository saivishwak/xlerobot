"""Bimanual SOFollower sessions.

Holds **two** independent SOFollower connections — one per arm. Each `_ArmSession`
owns its own bus, calibration, bounds, and torque state. The `_BimanualMotors`
wrapper exposes per-side methods (`MOTORS.connect(side)`, `MOTORS.read_positions(side)`,
`MOTORS.send_action(side, action)`) and provides aggregate views (`connected_sides`,
`status()`).

Safety model: each arm is connected explicitly. Disconnect is per-side too — the
other arm is untouched. The webapp can have:
    - 0 arms connected (nothing torqued)
    - 1 arm connected (single-arm mode, equivalent to the old behavior)
    - 2 arms connected (both arms torqued, both holding their pose)

VR drive decides which arm is "active" — even with both arms torqued, only the
active arm receives motion commands; the other arm just holds via its motor PID.
No autonomous motion anywhere: connect = torque on (arm holds current pose);
disconnect = torque off (arm goes limp where it is).

Used by `webapp/backend/vr_teleop.py` as the motor-bus backing store.
"""
from __future__ import annotations

import atexit
import logging
import pathlib
import signal
import sys
import threading
import time
from typing import Any, Literal

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Re-use the same XLerobot config builder the CLI runners use (it patches the
# right calibration_dir + ports from config/xlerobot.yaml).
_SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

log = logging.getLogger(__name__)

ArmSide = Literal["left", "right"]

JOINTS_PER_ARM = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _all_joint_names() -> tuple[str, ...]:
    return tuple(
        f"{side}_arm_{j}"
        for side in ("left", "right")
        for j in JOINTS_PER_ARM
    )


JOINTS = _all_joint_names()
DEFAULT_BOUNDS = {j: (-180.0, 180.0) for j in JOINTS}


class _ArmSession:
    """One SOFollower instance, owning the bus and calibration for one arm.

    Created once at module load (one per side) and reused across connect/disconnect
    cycles — the SOFollower itself is recreated each connect, but the wrapper
    keeps the cached `bounds`, `connected_at`, `last_error` state for status.
    """

    def __init__(self, side: ArmSide) -> None:
        self._side: ArmSide = side
        self._arm: Any = None                              # SOFollower instance
        self._lock = threading.RLock()
        self.bounds: dict[str, tuple[float, float]] = dict(DEFAULT_BOUNDS)
        self.last_error: str | None = None
        self.connected_at: float | None = None
        # Cached torque state — set True on connect() / lock_at_current(),
        # False on release_torque(). The drive loop checks this every tick
        # to skip arms that the user has released for hand-posing.
        self.torque_enabled: bool = False

    # ── properties ────────────────────────────────────────────────────────
    @property
    def connected(self) -> bool:
        return self._arm is not None

    @property
    def side(self) -> ArmSide:
        return self._side

    @property
    def arm(self):
        return self._arm

    # ── lifecycle ─────────────────────────────────────────────────────────
    def connect(self) -> dict:
        with self._lock:
            if self.connected:
                return self.status()
            try:
                self._arm = self._open_arm()
                self._sync_calibration(self._arm)
                self._refresh_bounds()
                self._loosen_gripper_protections(self._arm)
                self._stiffen_motor_pid(self._arm)
                self.connected_at = time.time()
                self.torque_enabled = True   # SOFollower.connect() torques on
                self.last_error = None
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                self._arm = None
                log.exception("connect failed for %s", self._side)
                raise
            return self.status()

    def disconnect(self) -> dict:
        """Release torque + close the bus. NEVER moves the robot."""
        with self._lock:
            if not self.connected:
                return self.status()
            try:
                self._arm.disconnect()
            except Exception as e:
                self.last_error = f"disconnect: {e}"
                log.warning("disconnect %s: %s", self._side, e)
            finally:
                self._arm = None
                self.connected_at = None
            # Give the OS a beat to release the serial FD before a possible reconnect.
            time.sleep(0.3)
            return self.status()

    def emergency_release_torque(self) -> dict:
        """Best-effort torque disable. Called from emergency stop paths."""
        with self._lock:
            arm = self._arm
        if arm is None:
            return self.status()
        try:
            arm.bus.disable_torque()
            self.torque_enabled = False
            log.info("emergency: torque released on %s arm", self._side)
        except Exception as e:
            log.warning("emergency: disable_torque failed on %s: %s", self._side, e)
        return self.disconnect()

    def release_torque_for_posing(self) -> dict:
        """Disable torque so the user can hand-pose this arm. The arm goes
        limp and will sag under gravity — caller is expected to warn the user
        and support the arm while posing. Bus stays connected; the drive loop
        will skip this arm until `lock_at_current()` re-enables torque.
        """
        with self._lock:
            if self._arm is None:
                raise RuntimeError(f"{self._side} arm not connected")
            try:
                self._arm.bus.disable_torque()
                self.torque_enabled = False
                log.info("[%s] torque RELEASED for hand-posing — support the arm",
                         self._side)
            except Exception as e:
                self.last_error = f"release_torque: {e}"
                log.exception("release_torque failed on %s", self._side)
                raise
            return self.status()

    def lock_at_current(self) -> dict:
        """Re-enable torque. To prevent the motor snapping to a stale
        Goal_Position (set before the release), we first write Goal_Position =
        Present_Position for every joint, THEN enable torque. The motor wakes
        up already at its current pose.
        """
        with self._lock:
            if self._arm is None:
                raise RuntimeError(f"{self._side} arm not connected")
            bus = self._arm.bus
            try:
                # Read present, write as goal, then enable torque.
                present = bus.sync_read("Present_Position")
                bus.sync_write("Goal_Position", present)
                bus.enable_torque()
                self.torque_enabled = True
                log.info("[%s] torque LOCKED at present position", self._side)
            except Exception as e:
                self.last_error = f"lock_at_current: {e}"
                log.exception("lock_at_current failed on %s", self._side)
                raise
            return self.status()

    # ── reads / writes ────────────────────────────────────────────────────
    def read_positions(self) -> dict[str, float]:
        """Returns {prefixed_joint: value}, or {} if not connected."""
        with self._lock:
            if self._arm is None:
                return {}
            prefix = f"{self._side}_arm_"
            raw = self._arm.bus.sync_read("Present_Position")
            return {f"{prefix}{k}": float(v) for k, v in raw.items()}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Send absolute joint positions for THIS arm.

        Keys may be prefixed ('left_arm_shoulder_pan') or unprefixed ('shoulder_pan');
        prefixed keys for the OTHER arm are silently dropped (defensive — the bimanual
        wrapper routes per-side, but stray keys shouldn't crash).

        Returns the dict that was actually sent (after bounds clamping).
        """
        with self._lock:
            if self._arm is None:
                raise RuntimeError(f"{self._side} arm not connected")

            prefix = f"{self._side}_arm_"
            other_prefix = "right_arm_" if self._side == "left" else "left_arm_"
            clamped: dict[str, float] = {}
            for key, val in action.items():
                if key.startswith(other_prefix):
                    continue
                pj = key if key.startswith(prefix) else f"{prefix}{key}"
                lo, hi = self.bounds.get(pj, (-180.0, 180.0))
                clamped[pj] = max(lo, min(hi, float(val)))

            arm_action = {
                f"{pj.removeprefix(prefix)}.pos": v
                for pj, v in clamped.items()
            }
            self._arm.send_action(arm_action)
            return clamped

    # ── internals ─────────────────────────────────────────────────────────
    def _open_arm(self):
        import yaml
        from lerobot.robots.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        cfg_path = REPO_ROOT / "config" / "xlerobot.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        r = cfg.get("robot") or {}
        port_key = "port_left_base" if self._side == "left" else "port_right_head"
        id_key = "left_arm_id" if self._side == "left" else "right_arm_id"
        port = r.get(port_key)
        arm_id = r.get(id_key, f"{self._side}_follower_arm")
        if not port:
            raise RuntimeError(f"config/xlerobot.yaml missing robot.{port_key}")

        calib_dir = REPO_ROOT / "config" / "calibration" / "so_follower"
        arm_cfg = SOFollowerRobotConfig(
            id=arm_id,
            port=port,
            use_degrees=True,
            max_relative_target=r.get("max_relative_target"),
            calibration_dir=calib_dir,
        )
        arm = SOFollower(arm_cfg)
        try:
            arm.connect(calibrate=False)
        except Exception as e:
            msg = str(e)
            if "calibrat" in msg.lower():
                raise RuntimeError(
                    f"{msg}\n\nCalibration JSON missing for id={arm_id}. Run:\n"
                    f"  uv run lerobot-calibrate --robot.type=so101_follower "
                    f"--robot.port={port} --robot.id={arm_id} "
                    f"--robot.calibration_dir=$(pwd)/config/calibration/so_follower"
                ) from e
            raise
        return arm

    def _sync_calibration(self, arm) -> None:
        """If the motor EEPROM disagrees with the JSON, push JSON → EEPROM.

        Protection against left/right calibration cross-contamination (see the
        right-elbow EEPROM bug from earlier). Safe to run on every connect:
        writes are no-ops if already in sync.
        """
        try:
            if arm.calibration and not arm.bus.is_calibrated:
                log.info("on-motor calibration mismatch on %s; writing JSON to EEPROM",
                         self._side)
                with arm.bus.torque_disabled():
                    arm.bus.write_calibration(arm.calibration)
        except Exception as e:
            log.warning("calibration sync failed on %s: %s", self._side, e)

    def _stiffen_motor_pid(self, arm) -> None:
        """Override lerobot's conservative `P_Coefficient = 16` set by
        SOFollower.configure(). lerobot's value is half of phospho's published
        `P=32` and the Feetech factory default — and the trade-off shows up as
        sluggish servo tracking that even tuned software smoothing can't fix.

        Bumping to 24 (one notch below phospho's 32) gives noticeably tighter
        position tracking without inducing oscillation on the SO-101's gear
        train. The gripper is excluded (it has its own protection settings
        from `_loosen_gripper_protections`).
        """
        if arm is None: return
        try:
            with arm.bus.torque_disabled():
                for motor in arm.bus.motors:
                    if motor == "gripper":
                        continue
                    arm.bus.write("P_Coefficient", motor, 20)
            log.info("%s arm motors: P_Coefficient = 20 (was 16 from lerobot default)",
                     self._side)
        except Exception as e:
            log.warning("could not stiffen motor PID on %s: %s", self._side, e)

    def _loosen_gripper_protections(self, arm) -> None:
        """Override the very conservative gripper protections that lerobot's
        SOFollower.configure() bakes in on every connect.

        lerobot writes (see lerobot/.../so_follower.py:166):
            Max_Torque_Limit  = 500    (50% of max)
            Protection_Current = 250   (50% — trips easily)
            Overload_Torque    = 25    (25% — after a trip, motor goes near-limp)

        Overload_Torque=25 is the killer: when the gripper touches its own
        mechanical stop while opening/closing, the motor drops to 25% torque
        and stops responding to subsequent position commands until power cycle.
        """
        if arm is None: return
        try:
            with arm.bus.torque_disabled():
                arm.bus.write("Max_Torque_Limit", "gripper", 900)
                arm.bus.write("Protection_Current", "gripper", 500)
                arm.bus.write("Overload_Torque", "gripper", 80)
            log.info("%s gripper protections loosened: max_torque=900 "
                     "protection_current=500 overload_torque=80", self._side)
        except Exception as e:
            log.warning("could not loosen gripper protections on %s: %s",
                        self._side, e)

    def _refresh_bounds(self) -> None:
        """Compute degree-space bounds for joints on THIS arm."""
        from lerobot.motors import MotorNormMode

        bounds = dict(self.bounds)  # keep prior bounds for other-arm keys
        bus = self._arm.bus
        calib = bus.calibration or {}
        prefix = f"{self._side}_arm_"
        for motor_name, c in calib.items():
            lo_ticks = float(getattr(c, "range_min", 0))
            hi_ticks = float(getattr(c, "range_max", 0))
            if hi_ticks <= lo_ticks:
                continue
            model = bus.motors[motor_name].model
            max_res = bus.model_resolution_table[model] - 1
            mode = bus.motors[motor_name].norm_mode
            if mode is MotorNormMode.DEGREES:
                mid = (lo_ticks + hi_ticks) / 2
                lo = (lo_ticks - mid) * 360 / max_res
                hi = (hi_ticks - mid) * 360 / max_res
            elif mode is MotorNormMode.RANGE_0_100:
                lo, hi = 0.0, 100.0
            elif mode is MotorNormMode.RANGE_M100_100:
                lo, hi = -100.0, 100.0
            else:
                lo, hi = lo_ticks, hi_ticks
            bounds[f"{prefix}{motor_name}"] = (lo, hi)
        self.bounds = bounds

    # ── status snapshot ───────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "side": self._side,
            "connected": self.connected,
            "connected_at": self.connected_at,
            "last_error": self.last_error,
            "torque_enabled": self.torque_enabled,
        }


class _BimanualMotors:
    """Holds one `_ArmSession` per side and routes API calls by side.

    Aggregate `bounds` is a single merged dict (keys are prefixed so they don't
    collide between left/right) — exposed for the VR session's joint clamping.
    """

    def __init__(self) -> None:
        self._arms: dict[ArmSide, _ArmSession] = {
            "left":  _ArmSession("left"),
            "right": _ArmSession("right"),
        }

    # ── per-side access ──────────────────────────────────────────────────
    def __getitem__(self, side: ArmSide) -> _ArmSession:
        if side not in ("left", "right"):
            raise KeyError(side)
        return self._arms[side]

    def session(self, side: ArmSide) -> _ArmSession:
        return self[side]

    def is_connected(self, side: ArmSide) -> bool:
        return self[side].connected

    @property
    def connected_sides(self) -> list[ArmSide]:
        return [s for s in ("left", "right") if self._arms[s].connected]

    @property
    def any_connected(self) -> bool:
        return bool(self.connected_sides)

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        """Merged bounds across both arms (keys are prefixed)."""
        merged: dict[str, tuple[float, float]] = dict(DEFAULT_BOUNDS)
        for arm in self._arms.values():
            merged.update(arm.bounds)
        return merged

    # ── per-side delegation ───────────────────────────────────────────────
    def connect(self, side: ArmSide) -> dict:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        return self[side].connect()

    def disconnect(self, side: ArmSide | None = None) -> dict:
        """Disconnect a specific side, or both if side is None."""
        if side is None:
            for s in ("left", "right"):
                self._arms[s].disconnect()
            return self.status()
        return self[side].disconnect()

    def emergency_release_torque(self) -> dict:
        """Release torque on every connected arm (best-effort)."""
        for s in self.connected_sides:
            try:
                self._arms[s].emergency_release_torque()
            except Exception as e:
                log.warning("emergency: %s arm: %s", s, e)
        return self.status()

    def release_torque_for_posing(self, side: ArmSide) -> dict:
        return self[side].release_torque_for_posing()

    def lock_at_current(self, side: ArmSide) -> dict:
        return self[side].lock_at_current()

    def is_torque_enabled(self, side: ArmSide) -> bool:
        return self[side].torque_enabled

    def read_positions(self, side: ArmSide | None = None) -> dict[str, float]:
        """Returns {prefixed_joint: value}. With `side=None`, reads BOTH connected
        arms and merges. With a side, reads only that arm."""
        if side is not None:
            return self[side].read_positions()
        merged: dict[str, float] = {}
        for s in self.connected_sides:
            merged.update(self._arms[s].read_positions())
        return merged

    def send_action(self, side: ArmSide, action: dict[str, float]) -> dict[str, float]:
        """Send absolute joint positions to one arm. The action dict may include
        keys for the other arm — those are dropped inside `_ArmSession.send_action`."""
        return self[side].send_action(action)

    def status(self) -> dict:
        """Aggregate status: per-side connection state + merged bounds + joint names."""
        return {
            "arms": {s: self._arms[s].status() for s in ("left", "right")},
            "connected_sides": self.connected_sides,
            "joints": list(JOINTS),
            "bounds": {j: list(self.bounds.get(j, DEFAULT_BOUNDS[j])) for j in JOINTS},
        }


MOTORS = _BimanualMotors()
# Backwards-compat alias used by api.py / vr_teleop.py imports.
SESSION = MOTORS


@atexit.register
def _shutdown_release_torque() -> None:
    """Release torque on Python exit. Never moves the robot."""
    if MOTORS.any_connected:
        log.info("atexit: releasing torque (no motion)")
        try:
            MOTORS.emergency_release_torque()
        except Exception as e:
            log.warning("atexit: %s", e)


def _install_sigterm_handler() -> None:
    """SIGTERM also triggers torque release (atexit handles SIGINT-driven KeyboardInterrupt)."""
    try:
        signal.signal(signal.SIGTERM, lambda *_: _shutdown_release_torque())
    except (ValueError, OSError):
        pass  # signal handlers only register from main thread


_install_sigterm_handler()
