#!/usr/bin/env python3
"""Read current joint positions and save them as the home pose in config/xlerobot.yaml.

Use after manually moving the robot to a known-safe parking pose:

    newgrp dialout   # if not already in dialout for this shell
    uv run python scripts/save_home_pose.py

The webapp's disconnect-time auto-home will then drive both arms back to those exact
angles next time you click Home & disconnect.
"""
from __future__ import annotations

import pathlib
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    cfg = yaml.safe_load(CONFIG_YAML.read_text())
    r = cfg["robot"]

    calib_dir = REPO_ROOT / "config" / "calibration" / "so_follower"

    arms: dict[str, SOFollower] = {}
    try:
        for side, port_key, id_key in (
            ("left",  "port_left_base",  "left_arm_id"),
            ("right", "port_right_head", "right_arm_id"),
        ):
            port    = r[port_key]
            arm_id  = r.get(id_key, f"{side}_follower_arm")
            arm_cfg = SOFollowerRobotConfig(id=arm_id, port=port, use_degrees=True,
                                            calibration_dir=calib_dir)
            arm = SOFollower(arm_cfg)
            arm.connect(calibrate=False)
            arms[side] = arm
            print(f"[ok] connected {side} arm on {port}")

        # Read positions (motors-bus returns the per-motor name, not the prefixed name).
        home: dict[str, float] = {}
        for side, arm in arms.items():
            raw = arm.bus.sync_read("Present_Position")
            for motor, deg in raw.items():
                home[f"{side}_arm_{motor}"] = round(float(deg), 2)

        print("\nMeasured pose (degrees):")
        for k, v in home.items():
            print(f"  {k:30} {v:>7.2f}")

        # Write back to YAML under robot.home_pose.
        cfg["robot"]["home_pose"] = home
        CONFIG_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"\n[ok] wrote home_pose to {CONFIG_YAML}")
    finally:
        for arm in arms.values():
            try:
                arm.disconnect()
            except Exception as e:
                print(f"[warn] disconnect: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
