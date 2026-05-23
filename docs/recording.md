# Dataset Recording

Captures bimanual SO-101 teleop demonstrations to a LeRobot v2 dataset. Use the recorded data to fine-tune pi0.5 or another VLA.

## What gets recorded

Each frame contains:
- `action` — commanded joint positions for both arms (12-vector: 6 joints × 2 arms)
- `observation.state` — present joint positions (same 12-vector)
- `observation.images.<role>` — video stream per camera with a role assigned (head, left_wrist, right_wrist)

At 30 Hz, with 3 cameras at 640×480 RGB.

## Configure

In `config/xlerobot.yaml`:

```yaml
dataset:
  repo_id: <hf-user>/<dataset-name>   # e.g. saivishwak/xlerobot-vr-teleop
  fps: 30
  push_to_hub: false                  # set true to push to HF after finalizing
  home_before_episode: true           # auto-home all arms at start of each episode
```

Set camera roles on the **Cameras page** (not the Teleop page). Each camera needs a role for it to be included as an `observation.images.*` feature.

## Recording flow

1. Connect both arms, calibrate VR (once, see [calibration.md](calibration.md)).
2. Capture home pose (once, see [calibration.md](calibration.md)).
3. Set a task description. Either via API or the dataset config's `task_default`.
4. **Press B** on the right controller — or click *Start recording* in the UI.
   - If `home_before_episode: true`, all arms slowly move to home first.
   - Then a new episode opens.
5. Squeeze grip + perform the demonstration.
6. **Press B again** — episode is saved to disk.
7. Repeat for as many episodes as you want.

Frames are recorded **every drive-loop tick** (30 Hz) while recording is active, regardless of whether you're actively teleoperating that tick. Passive arms still contribute their `observation.state`.

## Where it lives

Episodes are written to `$HF_LEROBOT_HOME/<repo_id>/` (default `~/.cache/huggingface/lerobot/<repo_id>/`).

If `push_to_hub: true`, the recorder pushes to the Hub on `emergency_stop` (which calls `finalize`).

## Stopping cleanly

- **Toggle off recording (B button or UI)** to save the current episode.
- **Emergency Stop** also flushes the in-flight episode and finalizes the dataset.
- Disconnecting an arm does *not* save — recording can continue with the remaining arm(s) (their joints contribute, the disconnected arm's joints come through as zeros).

### View

Quick sanity check before viewing

```bash
# Confirm episodes are on disk
ls ~/.cache/huggingface/lerobot/saivishwak/xlerobot-vr-teleop/data/chunk-000/
# -> file-000.parquet

# Confirm v3 episode metadata exists
ls ~/.cache/huggingface/lerobot/saivishwak/xlerobot-vr-teleop/meta/episodes/
# -> chunk-000/file-000.parquet

# Open viewer for the most recent
uv run python scripts/lerobot_dataset_viz_main.py --repo-id saivishwak/xlerobot-vr-teleop \
  --episode-index $(ls ~/.cache/huggingface/lerobot/saivishwak/xlerobot-vr-teleop/data/chunk-000/ |
wc -l | awk '{print $1-1}')
```

Use the project wrapper instead of calling `lerobot-dataset-viz` directly. The vendored LeRobot checkout defaults to a version tag lookup for v3 datasets; the wrapper leaves the vendor code untouched, loads the Hub `main` revision instead, and defaults the dataloader to `--num-workers 0` to avoid shared-memory issues on small machines. To view another revision:

```bash
LEROBOT_DATASET_REVISION=my-branch uv run python scripts/lerobot_dataset_viz_main.py \
  --repo-id saivishwak/xlerobot-vr-teleop \
  --episode-index 0
```