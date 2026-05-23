#!/usr/bin/env python3
"""Force-write the per-arm calibration JSON to the motor EEPROM.

Use when the on-motor `Min_Position_Limit` / `Max_Position_Limit` / `Homing_Offset`
registers don't match `config/calibration/so_follower/<arm>.json` (symptom: a joint
suddenly won't reach its full range, but the calibration file looks correct).

SOFollower.connect(calibrate=False) deliberately doesn't sync these — it expects you
to either run `lerobot-calibrate` interactively or live with the mismatch. This script
is the non-interactive equivalent: it pushes the JSON values to the motor EEPROM.

Usage:
    newgrp dialout
    uv run python scripts/sync_calibration.py            # both arms
    uv run python scripts/sync_calibration.py --arm right
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"


def open_arm(side: str):
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    cfg = yaml.safe_load(CONFIG_YAML.read_text())
    r = cfg["robot"]
    port_key = "port_left_base" if side == "left" else "port_right_head"
    id_key = "left_arm_id" if side == "left" else "right_arm_id"
    arm_cfg = SOFollowerRobotConfig(
        id=r.get(id_key, f"{side}_follower_arm"),
        port=r[port_key],
        use_degrees=True,
        calibration_dir=REPO_ROOT / "config" / "calibration" / "so_follower",
    )
    arm = SOFollower(arm_cfg)
    arm.connect(calibrate=False)
    return arm


def sync(arm, side: str) -> None:
    bus = arm.bus
    print(f"\n========== {side.upper()} ARM (port {bus.port}) ==========")
    if not arm.calibration:
        print(f"[skip] no calibration loaded for {side} — nothing to sync.")
        return

    before = bus.read_calibration()
    print("Before sync (on-motor EEPROM):")
    for m in bus.motors:
        c = before[m]; j = arm.calibration.get(m)
        diff = " ← MISMATCH" if (j and (c.range_min != j.range_min or c.range_max != j.range_max
                                        or c.homing_offset != j.homing_offset)) else ""
        print(f"  {m:18}  min={c.range_min:>5}  max={c.range_max:>5}  off={c.homing_offset:>6}{diff}")

    print("\nWriting JSON values to motor EEPROM (torque must be off first)...")
    with bus.torque_disabled():
        bus.write_calibration(arm.calibration)

    after = bus.read_calibration()
    print("\nAfter sync (on-motor EEPROM):")
    all_match = True
    for m in bus.motors:
        c = after[m]; j = arm.calibration.get(m)
        match = (j and c.range_min == j.range_min and c.range_max == j.range_max
                 and c.homing_offset == j.homing_offset)
        all_match = all_match and bool(match)
        flag = " ✓" if match else " ✗ STILL MISMATCH"
        print(f"  {m:18}  min={c.range_min:>5}  max={c.range_max:>5}  off={c.homing_offset:>6}{flag}")

    print(f"\n[{ 'ok' if all_match else 'WARN' }] {side} arm sync " +
          ("complete." if all_match else "INCOMPLETE — some writes did not stick."))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("left", "right", "both"), default="both")
    args = p.parse_args()

    sides = ("left", "right") if args.arm == "both" else (args.arm,)
    arms = {}
    try:
        for side in sides:
            try:
                arms[side] = open_arm(side)
            except Exception as e:
                print(f"[warn] could not connect {side}: {e}", file=sys.stderr)
        for side, arm in arms.items():
            sync(arm, side)
    finally:
        for arm in arms.values():
            try: arm.disconnect()
            except Exception: pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
