#!/usr/bin/env python3
"""Change a Feetech motor's ID by writing to its EEPROM.

New STS3215 motors ship with ID=1 from the factory. The bimanual SO101 layout
assigns IDs 1..6 per arm (shoulder_pan=1, shoulder_lift=2, elbow_flex=3,
wrist_flex=4, wrist_roll=5, gripper=6). Before a freshly-bought motor can join
the chain at any position other than shoulder_pan, its ID has to be flashed to
match its slot.

Recommended setup (avoids ID collisions during flashing):

    1. Power off the robot (12 V off).
    2. PHYSICALLY DISCONNECT the new motor from the rest of the chain. Plug
       its data cable directly into the USB-serial bridge so the new motor is
       the ONLY device on the bus.
    3. Power on (12 V on).
    4. In a dialout-enabled shell, run this script.
    5. Power off, re-attach the new motor in its proper chain position, power on.

If the new motor is already wired into the chain and you can't easily isolate
it, follow the warning in step 2 anyway — at least one of the two motors
sharing ID=1 will respond on the bus first; whose response you'll get is
non-deterministic. You may end up renaming the wrong one.

Usage:

    uv run python scripts/flash_motor_id.py \\
        --port /dev/ttyACM1 \\
        --from-id 1 \\
        --to-id 3

    # see all motors currently visible on the bus:
    uv run python scripts/flash_motor_id.py --port /dev/ttyACM1 --scan
"""
from __future__ import annotations

import argparse
import sys
import time


def scan(port: str, baudrate: int) -> None:
    """Ping every plausible ID on the bus and report which ones respond."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    placeholder = {"_": Motor(1, "sts3215", MotorNormMode.DEGREES)}
    bus = FeetechMotorsBus(port=port, motors=placeholder, calibration=None)
    bus.connect(handshake=False)
    try:
        print(f"scanning {port} @ {baudrate} baud (this takes ~10s)...")
        found = []
        for id_ in range(1, 21):  # 1..20 covers all plausible SO101/lekiwi IDs
            model_nb = bus.ping(id_, num_retry=1)
            if model_nb is not None:
                found.append((id_, model_nb))
                print(f"  ID {id_:>3}: model_number={model_nb}")
        if not found:
            print("\nNo motors detected. Check:")
            print("  - 12 V power LED on the motor (should be lit)")
            print("  - data cable connected between USB bridge and motor")
            print("  - port path is correct (try `ls /dev/ttyACM*`)")
            print("  - your shell has dialout group (run `newgrp dialout`)")
        else:
            print(f"\nFound {len(found)} motor(s) on the bus.")
    finally:
        bus.disconnect()


def flash(port: str, from_id: int, to_id: int, baudrate: int,
          model: str, force: bool) -> int:
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {"target": Motor(from_id, model, MotorNormMode.DEGREES)}
    bus = FeetechMotorsBus(port=port, motors=motors, calibration=None)

    print(f"[1/6] opening {port} @ {baudrate} baud...")
    bus.connect(handshake=False)

    try:
        print(f"[2/6] pinging motor at ID {from_id}...")
        model_nb = bus.ping(from_id)
        if model_nb is None:
            print(
                f"\n[FAIL] no motor responds at ID {from_id}.\n"
                f"  Verify: motor is the ONLY device on this bus, power is on,\n"
                f"  and the motor's current ID is actually {from_id}. To enumerate\n"
                f"  every responding ID, run with --scan.\n"
            )
            return 2
        print(f"      → got model_number={model_nb}")

        if not force:
            print(f"[3/6] checking that target ID {to_id} is free...")
            existing = bus.ping(to_id)
            if existing is not None:
                print(
                    f"\n[FAIL] another motor (model_number={existing}) already lives at ID {to_id}.\n"
                    f"  Disconnect it from the bus first, OR pass --force to proceed anyway\n"
                    f"  (you will have two motors at the same ID and the bus will get garbled).\n"
                )
                return 3
            print(f"      → ID {to_id} is unused")
        else:
            print(f"[3/6] --force given; SKIPPING collision check for ID {to_id}")

        print(f"[4/6] disabling torque + unlocking EEPROM on ID {from_id}...")
        try:
            bus.write("Torque_Enable", "target", 0)
        except Exception as e:
            print(f"      [warn] could not disable torque: {e}")
        try:
            # Some firmware revs require Lock=0 before writing EEPROM.
            bus.write("Lock", "target", 0, normalize=False)
        except Exception as e:
            print(f"      [warn] Lock write failed (may not exist on this fw): {e}")
        time.sleep(0.05)

        print(f"[5/6] writing new ID {to_id} to EEPROM...")
        bus.write("ID", "target", to_id, normalize=False)
        time.sleep(0.1)
    finally:
        bus.disconnect()

    # The motor now answers to to_id, not from_id. Reopen the bus with the
    # new mapping to lock the EEPROM back and verify.
    motors_new = {"target": Motor(to_id, model, MotorNormMode.DEGREES)}
    bus = FeetechMotorsBus(port=port, motors=motors_new, calibration=None)
    bus.connect(handshake=False)
    try:
        print(f"[6/6] verifying motor responds at new ID {to_id}...")
        model_nb_after = bus.ping(to_id)
        if model_nb_after is None:
            print(
                f"\n[FAIL] motor did NOT respond at new ID {to_id}.\n"
                f"  The EEPROM write may not have taken. Try power-cycling the robot\n"
                f"  and re-running `--scan` to see what ID it's at now.\n"
            )
            return 4
        try:
            bus.write("Lock", "target", 1, normalize=False)
        except Exception:
            pass
        print(f"      → confirmed (model_number={model_nb_after})")
    finally:
        bus.disconnect()

    print(f"\n[ok] motor on {port}: ID {from_id} → {to_id}")
    print("\nNext steps:")
    print("  1. Power-cycle the robot (12 V off / on) to commit EEPROM.")
    print("  2. Re-attach this motor in the chain at its physical slot.")
    print("  3. Re-run scripts/diagnose_motor.py --arm <left|right> to confirm")
    print("     it shows up and reports OK.")
    print("  4. Calibrate the arm whose motor you just replaced:")
    print(f"     uv run lerobot-calibrate \\")
    print(f"         --robot.type=so101_follower \\")
    print(f"         --robot.port={port} \\")
    print(f"         --robot.id=<left_follower_arm | right_follower_arm> \\")
    print(f"         --robot.calibration_dir=$(pwd)/config/calibration/so_follower")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True,
                   help="serial port the motor is on (e.g. /dev/ttyACM1)")
    p.add_argument("--baudrate", type=int, default=1_000_000,
                   help="bus baud rate (default 1_000_000 for SO101)")
    p.add_argument("--model", default="sts3215",
                   help="motor model name (lerobot table key, default sts3215)")
    p.add_argument("--scan", action="store_true",
                   help="list responding motor IDs on the bus and exit")
    p.add_argument("--from-id", type=int,
                   help="current motor ID (default for new motors: 1)")
    p.add_argument("--to-id", type=int,
                   help="desired motor ID (right arm elbow = 3)")
    p.add_argument("--force", action="store_true",
                   help="skip the collision check at --to-id")
    args = p.parse_args()

    if args.scan:
        scan(args.port, args.baudrate)
        return 0

    if args.from_id is None or args.to_id is None:
        p.error("--from-id and --to-id are required (use --scan to inspect the bus first)")

    if not (1 <= args.from_id <= 253):
        p.error(f"--from-id must be 1..253 (got {args.from_id})")
    if not (1 <= args.to_id <= 253):
        p.error(f"--to-id must be 1..253 (got {args.to_id})")
    if args.from_id == args.to_id:
        p.error("--from-id and --to-id are the same; nothing to do")

    return flash(args.port, args.from_id, args.to_id, args.baudrate, args.model, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
