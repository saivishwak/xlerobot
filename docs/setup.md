# Setup

One-time install + hardware config for a bimanual SO-101 XLeRobot.

## Install

```bash
make setup
```

This pulls submodules (`lerobot`, `XLeRobot`, `XLerobot_xuweiwu`, `openpi`), installs Python deps in a `.venv`, and prints any hardware checks that fail.

## Configure `config/xlerobot.yaml`

Set five things:

1. **Motor bus ports** — both arms. Find them with `uv run lerobot-find-port`.
   ```yaml
   robot:
     port_left_base:  /dev/ttyACM1
     port_right_head: /dev/ttyACM0
   ```
2. **Camera paths** — find with `ls /dev/v4l/by-path/`.
3. **Gripper convention** — if pulling the trigger opens (instead of closes), swap:
   ```yaml
   gripper:
     open_value: 0
     closed_value: 100
   ```
4. **VR network ports** — defaults are 8443/8442. If your ISP blocks them, switch to 5443/5442.
5. **Dataset repo** — for LeRobot recording. `<hf-user>/<dataset-name>`.

## Motor calibration

The `config/calibration/so_follower/{left,right}_follower_arm.json` files come from lerobot's calibration tool. If you've already calibrated, copy them in. If not:

```bash
uv run lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=$PORT \
  --robot.id=left_follower_arm \
  --robot.calibration_dir=$(pwd)/config/calibration/so_follower
```

Repeat for the right arm.

**One known gotcha**: lerobot's calibration sometimes captures a `range_min == range_max` for the gripper if you don't move it through its full open↔close cycle. Fix with:

```bash
uv run python scripts/calibrate_gripper.py --arm right
uv run python scripts/calibrate_gripper.py --arm left
```

## Add yourself to `dialout`

```bash
sudo usermod -aG dialout $USER
newgrp dialout         # picks up the group in this shell
```

Required for `/dev/ttyACM*` access. Without it the webapp can't open the motor bus.

## Run the webapp

```bash
make webapp     # http://localhost:5000
```

That's it. Open the page, connect an arm, follow [teleop.md](teleop.md).
