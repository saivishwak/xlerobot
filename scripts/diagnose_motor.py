#!/usr/bin/env python3
"""Read diagnostic registers for one or both arms — useful when a joint stops
reaching its full range. Tells you whether the cause is:

  - shrunken on-motor position limit (Min/Max_Position_Limit registers)
  - reduced torque limit (Torque_Limit)
  - latched hardware error (Status register has non-zero bits)
  - high temperature
  - calibration disagreement vs. on-motor limits

Usage:
    newgrp dialout          # if not already in dialout for this shell
    uv run python scripts/diagnose_motor.py            # both arms
    uv run python scripts/diagnose_motor.py --arm right
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"

# Bit meanings for STS3215 Status register (datasheet table 5-3).
STATUS_BITS = [
    (0x01, "Voltage"),
    (0x02, "Sensor"),
    (0x04, "Temperature"),
    (0x08, "Current"),
    (0x10, "Angle"),
    (0x20, "Overload"),
    (0x40, "Instruction"),
    (0x80, "Checksum"),
]


def decode_status(byte: int) -> str:
    if byte == 0:
        return "OK"
    flags = [name for mask, name in STATUS_BITS if byte & mask]
    return f"0x{byte:02x} → " + ", ".join(flags)


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


def diagnose(arm, side: str) -> None:
    bus = arm.bus
    print(f"\n========== {side.upper()} ARM (port {bus.port}) ==========")
    motors = list(bus.motors.keys())
    # Per-motor reads
    present = bus.sync_read("Present_Position")
    min_lim = bus.sync_read("Min_Position_Limit")
    max_lim = bus.sync_read("Max_Position_Limit")
    torq_lim = bus.sync_read("Torque_Limit")
    temp    = bus.sync_read("Present_Temperature")
    status  = bus.sync_read("Status")
    voltage = bus.sync_read("Present_Voltage")

    calib = bus.calibration or {}
    print(
        f"{'joint':18} {'present':>9} {'cal_min':>9} {'cal_max':>9}  "
        f"{'reg_min':>8} {'reg_max':>8}  {'torq':>5} {'°C':>3} {'V':>4}  status"
    )
    for m in motors:
        c = calib.get(m)
        cmin = c.range_min if c else None
        cmax = c.range_max if c else None
        print(
            f"{m:18} {present[m]:>9.2f} "
            f"{cmin if cmin is not None else '—':>9} "
            f"{cmax if cmax is not None else '—':>9}  "
            f"{min_lim[m]:>8} {max_lim[m]:>8}  "
            f"{torq_lim[m]:>5} {temp[m]:>3} {voltage[m]/10:>4.1f}  "
            f"{decode_status(int(status[m]))}"
        )


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
            diagnose(arm, side)
    finally:
        for side, arm in arms.items():
            try: arm.disconnect()
            except Exception: pass
    print("\nLegend:")
    print("  reg_min/reg_max = on-motor Min_Position_Limit / Max_Position_Limit registers.")
    print("  If these are tighter than cal_min/cal_max, the firmware is clipping motion.")
    print("  Overload bit in status = latched fault. Power-cycle the robot to clear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
