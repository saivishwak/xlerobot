#!/usr/bin/env python3
"""Run Meta Quest 3 VR teleop and capture a LeRobot dataset.

Thin wrapper around XLerobot_xuweiwu/software/examples/8_vr_teleop_with_dataset_recording_dualarm.py.
Applies hardware overrides from config/xlerobot.yaml and CLI args, then delegates to the
upstream main() — no logic reimplemented here.

Prerequisites:
    bash scripts/setup_xlerobot.sh   (one-time)

Usage:
    uv run python scripts/run_vr_teleop_capture.py \\
        --task "Pick the red block and place it in the bin" \\
        --episodes 5 \\
        --episode-time 60
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import types

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "XLerobot_xuweiwu" / "software" / "examples" / "8_vr_teleop_with_dataset_recording_dualarm.py"
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, help="Task description (also stored in dataset).")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--episode-time", type=int, default=60, help="Max seconds per episode.")
    p.add_argument("--fps", type=int, default=None, help="Override control loop FPS.")
    p.add_argument("--repo-id", default=None, help="Override dataset HF repo id.")
    p.add_argument("--strict-motors", action="store_true",
                   help="Require all XLerobot motors to be present (default: lenient — "
                        "absent base/head motors are dropped from the bus registry).")
    return p.parse_args()


def load_yaml() -> dict:
    if not CONFIG_YAML.is_file():
        sys.exit(f"missing {CONFIG_YAML} — copy and edit config/xlerobot.yaml")
    return yaml.safe_load(CONFIG_YAML.read_text())


def patch_xlerobot_factory(robot_id: str) -> None:
    """Patch XLerobotConfig() so the upstream main()'s `XLerobotConfig(id=...)` returns a
    fully-overridden instance built from config/xlerobot.yaml.

    Setting class attributes on the dataclass does NOT change `__init__` defaults — the
    @dataclass decorator captures defaults into the generated signature at decoration
    time. So we wrap the class and substitute it in the upstream module's globals.
    """
    from _xlerobot_loader import make_config
    import lerobot.robots.xlerobot as xr

    original = xr.XLerobotConfig

    def patched(*args, **kwargs):  # signature is a superset of XLerobotConfig.__init__
        if args or any(k for k in kwargs if k != "id" and k != "use_degrees"):
            # Caller is being explicit — respect their args.
            return original(*args, **kwargs)
        return make_config(robot_id=kwargs.get("id", robot_id))

    xr.XLerobotConfig = patched  # type: ignore[assignment]


def install_lerobot_compat_shims() -> None:
    """The xuweiwu fork targets an older lerobot. Bridge the two import paths in-process."""
    import lerobot.robots.so_follower as so_follower
    import lerobot.robots.so_follower.robot_kinematic_processor as so_kin
    sys.modules.setdefault("lerobot.robots.so100_follower", so_follower)
    sys.modules.setdefault("lerobot.robots.so100_follower.robot_kinematic_processor", so_kin)

    # hw_to_dataset_features / build_dataset_frame moved from datasets.utils → utils.feature_utils.
    import lerobot.datasets.utils as ds_utils
    from lerobot.utils import feature_utils
    for name in ("hw_to_dataset_features", "build_dataset_frame"):
        if not hasattr(ds_utils, name):
            setattr(ds_utils, name, getattr(feature_utils, name))


def load_upstream() -> types.ModuleType:
    install_lerobot_compat_shims()
    spec = importlib.util.spec_from_file_location("vr_capture_upstream", UPSTREAM)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    args = parse_args()
    cfg = load_yaml()

    # The upstream module only instantiates XLerobotConfig() inside main(), so it's OK to
    # patch the constructor after the module is loaded.
    patch_xlerobot_factory(robot_id="xlerobot")
    if not args.strict_motors:
        from _xlerobot_loader import patch_motors_bus_lenient
        patch_motors_bus_lenient()

    upstream = load_upstream()
    upstream.TASK_DESCRIPTION = args.task
    upstream.HF_REPO_ID = args.repo_id or cfg["dataset"]["repo_id"]
    upstream.NUM_EPISODES = args.episodes
    upstream.EPISODE_TIME_SEC = args.episode_time
    if args.fps is not None:
        upstream.FPS = args.fps

    print("=" * 60)
    print(f"  Task         : {upstream.TASK_DESCRIPTION}")
    print(f"  Dataset repo : {upstream.HF_REPO_ID}")
    print(f"  Episodes     : {upstream.NUM_EPISODES} × ≤{upstream.EPISODE_TIME_SEC}s @ {upstream.FPS} fps")
    print(f"  VR endpoint  : https://{cfg['vr']['host_ip']}:{cfg['vr']['https_port']}")
    print("    → open this URL in the Quest 3 browser, accept the cert, start VR.")
    print("=" * 60)

    upstream.main()


if __name__ == "__main__":
    main()
