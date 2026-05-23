#!/usr/bin/env python3
"""Direct gripper exerciser — bypasses VR, IK, engage, and the drive loop.

Connects to one arm and drives the gripper between open (0) and closed (100)
a few times, printing the present position before/after each command. Tells
you whether the motor itself is responsive.

If THIS works but the webapp's VR teleop doesn't move the gripper, the bug is
in the VR data flow or drive loop. If THIS fails, the bug is in motor config /
calibration / hardware.

Usage:
    newgrp dialout
    uv run python scripts/test_gripper.py --arm right
    uv run python scripts/test_gripper.py --arm left
"""
from __future__ import annotations

import argparse
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
    return arm


def read_gripper(bus) -> dict[str, int]:
    """Snapshot the gripper's interesting registers."""
    g = "gripper"
    return {
        "Present_Position":    int(bus.sync_read("Present_Position",    motors=g)[g]),
        "Goal_Position":       int(bus.sync_read("Goal_Position",       motors=g)[g]),
        "Torque_Enable":       int(bus.sync_read("Torque_Enable",       motors=g)[g]),
        "Max_Torque_Limit":    int(bus.sync_read("Max_Torque_Limit",    motors=g)[g]),
        "Overload_Torque":     int(bus.sync_read("Overload_Torque",     motors=g)[g]),
        "Min_Position_Limit":  int(bus.sync_read("Min_Position_Limit",  motors=g)[g]),
        "Max_Position_Limit":  int(bus.sync_read("Max_Position_Limit",  motors=g)[g]),
        "Operating_Mode":      int(bus.sync_read("Operating_Mode",      motors=g)[g]),
        "Status":              int(bus.sync_read("Status",              motors=g)[g]),
        "Present_Temperature": int(bus.sync_read("Present_Temperature", motors=g)[g]),
    }


def fmt_status(byte: int) -> str:
    """Decode the STS3215 Status register bits."""
    bits = [
        (0x01, "Voltage"), (0x02, "Sensor"),  (0x04, "Temperature"),
        (0x08, "Current"), (0x10, "Angle"),   (0x20, "Overload"),
        (0x40, "Instruction"), (0x80, "Checksum"),
    ]
    if byte == 0:
        return "OK"
    return f"0x{byte:02x} → " + ", ".join(n for m, n in bits if byte & m)


def dump(label: str, snap: dict) -> None:
    print(f"  [{label}]")
    for k, v in snap.items():
        if k == "Status":
            print(f"    {k:22} = {v}  ({fmt_status(v)})")
        else:
            print(f"    {k:22} = {v}")


def drive_gripper(arm, target_pct: float) -> None:
    """Send one absolute position to the gripper. 0..100 (RANGE_0_100)."""
    arm.send_action({"gripper.pos": float(target_pct)})


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("left", "right"), required=True)
    p.add_argument("--cycles", type=int, default=3,
                   help="number of open→close→open cycles")
    p.add_argument("--hold-s", type=float, default=1.0,
                   help="seconds to hold each position before reading")
    args = p.parse_args()

    arm = open_arm(args.arm)
    try:
        bus = arm.bus

        print(f"=== {args.arm.upper()} arm gripper diagnostics ===")
        print("\nGripper state BEFORE any motion command:")
        dump("initial", read_gripper(bus))

        # Make sure torque is on.
        bus.enable_torque("gripper")
        print("\nTorque ENABLED on gripper.")

        for cycle in range(1, args.cycles + 1):
            for target in (100.0, 0.0):
                label = f"cycle {cycle} → goal={target}"
                print(f"\n{label}")
                snap_before = read_gripper(bus)
                drive_gripper(arm, target)
                time.sleep(args.hold_s)
                snap_after = read_gripper(bus)

                pres_before = snap_before["Present_Position"]
                pres_after = snap_after["Present_Position"]
                moved = pres_after - pres_before
                print(f"  Present: {pres_before} → {pres_after}  (moved {moved:+d} ticks)")
                if abs(moved) < 5:
                    print(f"  ⚠️  did not move; dumping registers:")
                    dump("after_no_motion", snap_after)
                else:
                    print(f"  ✓ moved; goal reg = {snap_after['Goal_Position']}, "
                          f"status = {fmt_status(snap_after['Status'])}, "
                          f"torque_enable = {snap_after['Torque_Enable']}")

        print("\nDone. Returning gripper to OPEN (0).")
        drive_gripper(arm, 0.0)
        time.sleep(args.hold_s)
        dump("final", read_gripper(bus))
        return 0
    finally:
        try: arm.disconnect()
        except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
