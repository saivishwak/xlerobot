# Troubleshooting

Quick diagnostics for the most common issues. Run from the repo root in a `newgrp dialout` shell.

## Gripper doesn't move

Most likely cause: motor calibration JSON has `range_min == range_max` for the gripper. Lerobot writes this to the motor's `Min_Position_Limit`/`Max_Position_Limit` registers on every connect, and the motor refuses any goal outside that ~1-tick window.

Diagnose:
```bash
uv run python scripts/test_gripper.py --arm right
```

If the script reports zero motion and the dumped registers show `Min_Position_Limit ≈ Max_Position_Limit`, re-calibrate just the gripper:
```bash
uv run python scripts/calibrate_gripper.py --arm right
```

This disables torque on the gripper only, asks you to move it through its full open↔close range by hand, then writes the captured range back to both the calibration JSON and the motor EEPROM.

## Robot doesn't move during teleop

Check on the page, in order:
1. `arms.<side>.connected` true?
2. `engaged` toggle on, `active_arm` set?
3. The arm is **anchored** (badge says "anchored" — i.e., you've squeezed grip at least once this session)?
4. Calibration card shows `session_yaw_deg` close to the angle you're standing at?
5. Controller card's "age ms" badge green (data flowing)?

If all green and still no motion, check the backend log — every grip-press logs `gripper:` lines, and motor send failures log warnings.

## Motion feels wrong (direction)

If moving your hand right makes the EE move sideways/back/down, the VR→robot frame calibration is off. Either:
- Stand differently and re-press grip (re-anchor),
- Or re-run the **Calibrate** wizard (see [calibration.md](calibration.md)).

Look at `session_yaw_deg` on the Calibration card — it should match (or be close to) the angle you're standing at relative to the robot.

## Wrist drifts when controller is still

If you're seeing slow wrist drift even with your hand still, the patched XLeVR isn't running. Restart the backend (`make webapp-backend`) — the patch lives in `XLerobot_xuweiwu/XLeVR/xlevr/inputs/vr_ws_server.py` and only takes effect on backend restart.

If drift persists, the Quest controller may need re-calibration in the Quest system menu.

## "Release for posing" doesn't release torque

Check the backend log for the `release_torque_for_posing` call. If you see no log line, the API didn't reach the backend (Flask not running, network blip). If you see the log but the arm still holds, the bus write failed silently — power-cycle the robot and try again.

## VR endpoint loads but motion doesn't reach the backend

Open the Quest browser → developer console. If you see WebSocket connection failures:
- Wrong port: check `vr.websocket_port` in `config/xlerobot.yaml`.
- Cert not accepted: open the HTTPS URL on the Quest, you should see a "Proceed anyway" page. Some ISP routers (Jio, Airtel) block self-signed certs on port 8443. Switch to 5443/5442 in the YAML config.

## EE stops at the workspace boundary

The robot's reach is ~25 cm. If `offset_robot` keeps growing but `target_ee_pos` saturates, your hand is past the arm's reach. Walk closer to the robot OR re-grip (re-anchor) with the arm at a more central pose.

## "HOMING…" doesn't clear in the UI after the arm reaches home

The arm physically arrived, but the present-position check is stricter than the motor's mechanical resolution. The drive loop now declares "arrived" once the software target has converged, not the present position — restart the backend if you're still hitting this.

## Bus opens, then `Missing motor IDs` error

A motor on that arm isn't responding. Common causes: power not connected, USB cable loose, motor ID mismatch (each arm should have IDs 1–6). Run:
```bash
uv run python scripts/diagnose_motor.py --arm right
```
to dump every motor's registers + Status byte. A `Status: 0x20 (Overload)` means the motor latched a fault — power-cycle the robot to clear.
