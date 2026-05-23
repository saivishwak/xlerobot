"""Build an XLerobotConfig from config/xlerobot.yaml.

Shared between the CLI runners (scripts/) and the web app (webapp/backend/motors.py).
Passes every override explicitly to the XLerobotConfig constructor — setting class-level
attributes does NOT change dataclass defaults, since `@dataclass` captures them into
the generated `__init__` signature at decoration time.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any

import yaml

log = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"


def load_yaml() -> dict:
    if not CONFIG_YAML.is_file():
        return {}
    return yaml.safe_load(CONFIG_YAML.read_text()) or {}


def build_cameras(cfg: dict) -> dict[str, Any]:
    """Convert config/xlerobot.yaml's cameras section into lerobot CameraConfig objects."""
    from lerobot.cameras.configs import ColorMode, Cv2Rotation
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cams: dict[str, Any] = {}
    for name, c in (cfg.get("cameras") or {}).items():
        if c.get("type") != "opencv" or not c.get("path"):
            continue
        cams[name] = OpenCVCameraConfig(
            index_or_path=c["path"],
            fps=int(c.get("fps", 30)),
            width=int(c.get("width", 640)),
            height=int(c.get("height", 480)),
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.NO_ROTATION,
            fourcc=c.get("fourcc", "MJPG"),
        )
    return cams


def patch_motors_bus_lenient() -> None:
    """Make `MotorsBus._assert_motors_exist` PRUNE missing motors instead of raising.

    Reason: both reference scripts (XLerobot_xuweiwu/.../9_pi05_inference_dualarm.py and
    8_vr_teleop_with_dataset_recording_dualarm.py) instantiate the full XLerobot driver,
    whose motor list includes 3 lekiwi base wheels (left bus) + 2 head motors (right bus).
    Bimanual SO101 setups have only the 6 arm motors per side, so the connect-time presence
    check fails. *But* both scripts only ever read/write arm motors at runtime — the base/head
    declarations are ceremonial. We drop the absent motors from the bus's registry so connect
    succeeds, then runtime calls that only touch arm motors work unchanged.

    Loud warnings are printed for every motor dropped, so the user knows what's not actually
    on the bus.
    """
    from lerobot.motors.motors_bus import MotorsBus

    if getattr(MotorsBus, "_xlerobot_lenient_patch", False):
        return  # idempotent

    original = MotorsBus._assert_motors_exist

    def lenient(self) -> None:  # type: ignore[no-untyped-def]
        try:
            original(self)
            return
        except RuntimeError as exc:
            msg = str(exc)
            if "Missing motor IDs" not in msg:
                raise   # different failure (wrong model, etc.) — bubble up
            # Re-run the ping, this time to figure out which configured motors are present.
            found_ids: set[int] = set()
            for id_ in self.ids:
                if self.ping(id_) is not None:
                    found_ids.add(id_)
            to_drop = [name for name, motor in list(self.motors.items())
                       if motor.id not in found_ids]
            if not to_drop:
                raise  # something deeper is wrong
            log.warning(
                "[lenient-motors] Pruning %d absent motor(s) from %s on port %s: %s",
                len(to_drop), type(self).__name__, self.port, to_drop,
            )
            print(
                f"[lenient-motors] {type(self).__name__} on {self.port}: "
                f"dropping {to_drop} (not detected on bus)"
            )
            for name in to_drop:
                del self.motors[name]
            # Sanity-check that what remains is wholly present (raises if not).
            original(self)

    MotorsBus._assert_motors_exist = lenient  # type: ignore[assignment]
    MotorsBus._xlerobot_lenient_patch = True  # type: ignore[attr-defined]


def make_config(robot_id: str = "xlerobot") -> Any:
    """Build an XLerobotConfig with all overrides from config/xlerobot.yaml applied."""
    from lerobot.robots.xlerobot import XLerobotConfig

    cfg = load_yaml()
    r = cfg.get("robot") or {}
    cams = build_cameras(cfg)

    kwargs: dict[str, Any] = {"id": robot_id}
    if r.get("port_left_base"):
        kwargs["port_left_base"] = r["port_left_base"]
    if r.get("port_right_head"):
        kwargs["port_right_head"] = r["port_right_head"]
    if "max_relative_target" in r:
        kwargs["max_relative_target"] = r["max_relative_target"]
    kwargs["use_degrees"] = r.get("use_degrees", True)
    if cams:
        kwargs["cameras"] = cams

    return XLerobotConfig(**kwargs)
