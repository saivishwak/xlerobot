# XLeRobot

Bimanual SO-101 VR teleop with [Meta Quest 3](https://www.meta.com/quest/quest-3/), [LeRobot v2](https://github.com/huggingface/lerobot) dataset recording, and a scaffolded [pi0.5](https://www.physicalintelligence.company/blog/pi05) VLA inference loop for fine-tuning on captured data.

## Quick start

```bash
make setup              # one-time install + submodules
$EDITOR config/xlerobot.yaml   # set ports, cameras, dataset repo
make webapp             # http://localhost:5000
```

Then in the browser: *Connect* each arm → open the VR endpoint URL on the Quest → squeeze grip and drive.

Detailed steps: **[docs/setup.md](docs/setup.md)**

## Docs

| | |
|---|---|
| [setup.md](docs/setup.md)            | Install, configure hardware, motor calibration |
| [teleop.md](docs/teleop.md)          | Run VR teleop, button reference, bimanual mode |
| [calibration.md](docs/calibration.md) | VR-frame calibration wizard + home pose capture |
| [recording.md](docs/recording.md)    | Capture LeRobot datasets for VLA training |
| [troubleshooting.md](docs/troubleshooting.md) | Common issues + diagnostics |

## What's in the box

```
.
├── lerobot/             (submodule) HF lerobot — LeRobotDataset, motor drivers
├── XLeRobot/            (submodule) XLeRobot main repo — XLeVR/ = WebXR pipeline
├── XLerobot_xuweiwu/    (submodule) bimanual SO-101 fork (branch pi05_dual_arm)
├── openpi/              (submodule) pi0.5 server (branch biso101_training_support)
├── config/
│   ├── xlerobot.yaml          hardware config (ports, cameras, dataset repo, home_pose)
│   ├── vr_calibration.yaml    auto-managed; per-arm VR→robot rotation matrix
│   └── calibration/so_follower/{left,right}_follower_arm.json   motor calibrations
├── webapp/              Flask backend + React/Mantine frontend
├── scripts/             CLI tools (gripper calibration, motor diagnostics, etc.)
└── docs/                ← you are here
```

## Webapp pages

| Page | Purpose |
|---|---|
| Dashboard  | Hardware health checks |
| Cameras    | Live MJPEG previews + role assignment (head, left_wrist, right_wrist) |
| VR Teleop  | The main control surface — connect arms, run the calibration wizard, record episodes |

## Safety model

- **No autonomous motion**. Connect = motors hold; disconnect = torque off, no homing. The only motion the app initiates is the user-clicked *Go to Home*.
- **EMERGENCY STOP** button on the VR Teleop page disables all torque immediately.
- **Per-tick joint caps** + **goal-staleness watchdog** + **engage gate** = three independent safety guards on every motion command.

## Notes on submodules

`make setup` copies XLeRobot-specific files (`xlerobot/`, `SO101Robot.py`, etc.) into the `lerobot/` submodule tree per the XLeRobot docs. Those paths are added to `lerobot/.git/info/exclude` so the submodule's working tree stays clean for upstream pulls.
