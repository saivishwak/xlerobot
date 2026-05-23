#!/usr/bin/env python3
"""Targeted gripper re-calibration. Fixes only the gripper's range_min /
range_max in the arm's calibration JSON; leaves all other joints' values
untouched.

The lerobot SOFollower calibration JSONs at config/calibration/so_follower/*.json
have the gripper's `range_min == range_max` (1- or 2-tick range), which means
on every connect, lerobot writes Min_Position_Limit/Max_Position_Limit to those
values and the motor refuses to move outside that 1-tick window. This script
re-captures the real open↔closed range without touching the other joints'
calibration.

Usage:
    newgrp dialout
    uv run python scripts/calibrate_gripper.py --arm right

Flow:
    1. Disables torque on the gripper motor only.
    2. Prompts you to move the gripper through its FULL open ↔ closed range,
       sampling position 5 Hz.
    3. You press ENTER when done. Script reports the captured min/max ticks.
    4. Patches range_min/range_max in the arm's JSON and re-writes the on-motor
       limits via `write_calibration`.
    5. Re-enables torque.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

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
    return arm, arm_cfg.id


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("left", "right"), required=True)
    p.add_argument("--margin", type=int, default=20,
                   help="ticks of safety margin to leave inside the captured range")
    args = p.parse_args()

    arm, arm_id = open_arm(args.arm)
    bus = arm.bus
    g = "gripper"

    try:
        bus.disable_torque(g)
        time.sleep(0.2)

        print("\n=== gripper calibration ===")
        print(f"Arm:   {args.arm}  (id={arm_id})")
        print(f"Port:  {bus.port}")
        print()
        print("Torque is now DISABLED on the gripper — you can move it by hand.")
        print()
        print("Manually move the gripper from FULLY OPEN to FULLY CLOSED a few")
        print("times. Don't force it past its mechanical stops; just sweep the")
        print("real range a couple of times so we sample both endpoints.")
        print()
        input("Press ENTER to start sampling…")
        print("Sampling… move the gripper now. Press ENTER when done.")

        # Sample loop. We can't wait for stdin and sample at the same time without
        # threading, but we can sample for a fixed duration. Simpler: prompt the
        # user to confirm visually.
        samples = []
        # Read in a tight loop until the user hits ENTER. We use a non-blocking
        # stdin trick via select.
        import select
        deadline = time.time() + 600  # 10-minute safety cap
        while time.time() < deadline:
            # Read RAW ticks (normalize=False) — we need on-motor units to write
            # back into Min/Max_Position_Limit, not the 0..100 normalised value.
            tick = int(bus.read("Present_Position", g, normalize=False))
            samples.append(tick)
            print(f"  tick = {tick}    (samples={len(samples)})   "
                  f"min so far={min(samples)} max so far={max(samples)}", end="\r")

            # Non-blocking ENTER check.
            r, _, _ = select.select([sys.stdin], [], [], 0.0)
            if r:
                sys.stdin.readline()
                break
            time.sleep(0.05)
        print()  # newline after the carriage-return spam

        if len(samples) < 5:
            print("\nERROR: not enough samples captured. Did you move the gripper?")
            return 2

        raw_min, raw_max = min(samples), max(samples)
        span = raw_max - raw_min
        if span < 100:
            print(f"\nERROR: captured range too small ({span} ticks). Move the "
                  f"gripper through a wider range and try again.")
            return 2

        # Apply safety margin so the on-motor limits don't sit exactly at the
        # mechanical stops — leaves headroom for the firmware's position clamp.
        new_min = raw_min + args.margin
        new_max = raw_max - args.margin
        print(f"\nCaptured range: {raw_min} to {raw_max} ticks  (span={span})")
        print(f"With ±{args.margin}-tick margin → range_min={new_min} range_max={new_max}")

        # Patch the JSON.
        calib_path = REPO_ROOT / "config" / "calibration" / "so_follower" / f"{arm_id}.json"
        data = json.loads(calib_path.read_text())
        old_min = data["gripper"]["range_min"]
        old_max = data["gripper"]["range_max"]
        data["gripper"]["range_min"] = new_min
        data["gripper"]["range_max"] = new_max
        calib_path.write_text(json.dumps(data, indent=4) + "\n")
        print(f"\nPatched {calib_path.name}:")
        print(f"  gripper.range_min: {old_min} → {new_min}")
        print(f"  gripper.range_max: {old_max} → {new_max}")

        # Push to motor EEPROM (this is what write_calibration would do on connect).
        # `normalize=False` writes the raw tick values directly — these registers
        # are not user-facing units.
        print("\nWriting new Min/Max_Position_Limit to the motor EEPROM…")
        bus.write("Min_Position_Limit", g, new_min, normalize=False)
        bus.write("Max_Position_Limit", g, new_max, normalize=False)
        print("Done.")

        # Verify.
        new_reg_min = int(bus.read("Min_Position_Limit", g, normalize=False))
        new_reg_max = int(bus.read("Max_Position_Limit", g, normalize=False))
        print(f"\nVerification: on-motor Min_Position_Limit={new_reg_min}, "
              f"Max_Position_Limit={new_reg_max}")

        if new_reg_min == new_min and new_reg_max == new_max:
            print("\n✓ Calibration written successfully. Re-run `scripts/test_gripper.py` "
                  "to confirm the gripper moves.")
        else:
            print("\n⚠ Mismatch between requested and on-motor values. Motor may need "
                  "a power cycle.")

        return 0
    finally:
        try:
            bus.enable_torque(g)
        except Exception:
            pass
        try: arm.disconnect()
        except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
