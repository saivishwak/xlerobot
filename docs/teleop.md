# VR Teleop

Drive one or both SO-101 arms with a Meta Quest 3.

## Controllers

| Button | What it does |
|---|---|
| **Grip** (side, middle finger) | **Hold to drive**, **first press = anchor for teleop**. Releasing stops motion. |
| **Trigger** (front, index finger) | Close gripper while held. Released = open. |
| **A** (right) / **X** (left) | Toggle engage for that arm. Pressing A while right is active = disengage; pressing X = switch to left. |
| **B** (right) | Toggle dataset recording. |
| Thumbstick / Y / menu | Unused. |

## Per-session flow

1. **Open the webapp**: `http://<workstation>:5000`. Click *Connect* on each arm you want to use.
2. **Open the VR endpoint URL on the Quest browser** (shown on the page). Accept the self-signed cert, enter VR.
3. **Calibrate** if you haven't, or if you're standing somewhere new — see [calibration.md](calibration.md). Once-per-setup; the calibration is saved.
4. **Squeeze grip** on a controller to anchor that arm's EE pose. The card shows "anchored" and `anchor_ee_pos`.
5. **Hold grip + move your hand**. The arm follows. Pull trigger to close the gripper.
6. **Release grip** to stop. Re-grip = re-anchor (useful if you've walked around).
7. (Optional) Press **A** instead of toggling Engage in the UI. Press **B** to start/stop dataset recording.

## Bimanual

Both arms can be connected and torqued simultaneously, but only **one** is actively driven by VR at a time. The *Active arm* segmented control on the Engagement card switches between them. A/X buttons do the same from inside VR.

## Speed slider

Default is **1.0** (true 1:1 hand-to-EE motion). Drop to 0.5 for fine work. The per-tick joint caps are the hard safety limit underneath.

## Safety

- **EMERGENCY STOP** button (top of page) — instantly disables torque on both arms. The robot freezes wherever it is.
- **Watchdog** — if VR goals stop arriving (controller down, Wi-Fi blip), the drive loop stops within 0.3 s.
- **Per-tick joint caps** — max joint speeds capped (e.g. shoulder_pan 60°/s). Independent of the speed slider.
- **No autonomous motion**, ever. Disconnect = torque off, no homing. The only motion the app initiates is the user-clicked *Go to Home*.

See [troubleshooting.md](troubleshooting.md) if motion feels wrong or doesn't happen.
