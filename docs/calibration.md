# Calibration

Two independent setup steps, both done once per robot+user. Both persist to YAML.

## Home pose

The pose every recorded episode starts from. Critical for VLA training: every demonstration must start from the same proprioception state, or the policy can't generalise.

**Capture procedure** (per arm, on the Home pose card):

1. Click **Release for posing** → torque on that arm goes off. Support the arm by hand — it'll sag under gravity, especially the shoulder lift.
2. **Hand-pose** the arm to your desired starting position. Recommended: "robot waving hello" — upper arm raised ~45°, forearm forward, gripper open at chest height. Avoid extremes (fully extended or fully folded — IK gets sloppy).
3. Click **Capture** → reads joint positions, writes them to `robot.home_pose.<side>_arm_*` in `config/xlerobot.yaml` (comments preserved).
4. Click **Lock at current** → re-enables torque, holding at the just-captured pose (no snap-back to a stale goal).

**Go to Home**: from any subsequent pose, click *Go to Home* and the arm slowly interpolates back via the same drive loop teleop uses (same per-tick caps + KP). Slow enough to abort with EMERGENCY STOP if needed.

**Auto-home before each recorded episode** (recommended for VLA):
```yaml
dataset:
  home_before_episode: true
```
Then every B-press to start recording first homes all connected arms, waits for them to settle, then opens the episode.

## VR frame calibration

Tells the system which direction in VR-world space is "user-forward" and "user-up" for your body. Without this, motion direction is wrong unless you happen to stand exactly facing the room's VR-default direction.

**Procedure** (per arm, on the Calibration card):

1. Click **Calibrate** for that arm. Card switches to wizard mode.
2. **Step 1 — Forward axis**: put on headset. Squeeze grip on that controller, **keep it held**, move your hand straight forward (toward the robot, away from your body) by ~10 cm. Release grip. The card shows live motion magnitude.
3. **Step 2 — Up axis**: squeeze grip again, **keep it held**, move your hand straight up by ~10 cm. Release grip.
4. Calibration finalises. The 3×3 VR→robot rotation matrix is computed via Gram-Schmidt orthogonalisation of the two captured vectors and saved to **`config/vr_calibration.yaml`**.
5. The card now shows the captured `session_yaw_deg` plus a *saved at: …* timestamp.

Subsequent webapp restarts load the saved matrix automatically. You only need to re-run the wizard if you change where you stand or how you orient yourself relative to the robot.

**Why two motions, not one**: a single forward motion only solves for yaw. Capturing up too lets the system handle tilted users (e.g., sitting reclined). The orthogonalisation gives a full 3-DOF rotation, not just a 1-DOF yaw.

**Re-anchor vs re-calibrate**: every grip-press re-anchors the EE position (where the gripper sits at the moment of grip-press). That's different from the VR-frame calibration above, which only changes if you click *Calibrate*. Anchor = "where is my hand starting from now"; calibration = "what does forward/up mean".

## Files written

| File | Written by | When |
|---|---|---|
| `config/xlerobot.yaml` (`robot.home_pose`) | *Capture* button | On click |
| `config/vr_calibration.yaml` | Calibration wizard | When step 2 (up axis) finalises |

Both are auto-managed. To re-do, use the UI; don't edit by hand.
