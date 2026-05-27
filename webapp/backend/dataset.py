"""LeRobot v2 dataset recording for VR teleop.

Wraps `lerobot.datasets.LeRobotDataset.create()` for write-mode capture during
VR sessions. Each frame contains:
  - `action`              : commanded joint positions for both arms (12-vector)
  - `observation.state`   : present joint positions for both arms (12-vector)
  - `observation.images.<role>` : (H, W, 3) RGB frame from each configured camera

Episode boundary control:
  - `start_episode(task)` is called on the rising edge of either the B button
    on the right Quest controller or the UI "Start recording" toggle.
  - `add_frame(...)` is called every drive-loop tick while in an episode.
  - `end_episode()` is called on the next falling edge.
  - `finalize()` flushes everything to disk and (optionally) pushes to the Hub.

Camera frames come from `webapp.backend.cameras.get_stream(role).snapshot()`
which returns JPEG bytes — we decode once per frame to keep CameraStream's
existing single-frame producer architecture untouched.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import shutil
import threading
import time
from typing import Any, Optional

import cv2  # decode JPEG → numpy (BGR)
import numpy as np
import yaml

from . import cameras as cam_mod
from .motors import JOINTS_PER_ARM

log = logging.getLogger(__name__)
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Order must match what the drive loop writes into `action` / `observation.state`.
# All 12 joints (6 per arm × 2). If an arm is disconnected we still write
# placeholder values (NaN-like → use the LAST commanded value if any, else 0.0)
# so the schema stays fixed across episodes.
JOINT_ORDER: list[str] = [
    f"{side}_arm_{j}" for side in ("left", "right") for j in JOINTS_PER_ARM
]


class DatasetRecorder:
    """LeRobotDataset writer wrapper. One instance lives inside `VRTeleopSession`.

    Thread model: instantiation + start/end calls happen on whichever thread
    flips the recording flag (drive loop, for the B button; Flask handler thread
    for the UI button). `add_frame` is only called from the drive loop. Internal
    state mutations are guarded by `self._lock` since they can race.
    """

    def __init__(self, repo_id: str, fps: int,
                 camera_roles: list[str], camera_shape: tuple[int, int, int],
                 root: Optional[pathlib.Path] = None,
                 push_to_hub: bool = False) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._lock = threading.Lock()
        self.repo_id = repo_id
        self.fps = int(fps)
        self.camera_roles = list(camera_roles)
        self.camera_shape = tuple(camera_shape)
        self.push_to_hub_flag = bool(push_to_hub)
        self._in_episode = False
        self._episode_count = 0
        self._frame_count = 0
        self._current_task: str = ""
        self._last_saved_episode_index: Optional[int] = None
        self._last_saved_episode_frames: int = 0

        features = self._build_features(JOINT_ORDER, self.camera_roles, self.camera_shape)
        resolved_root = pathlib.Path(resolve_root(str(root) if root else None, repo_id))
        has_info = (resolved_root / "meta" / "info.json").is_file()
        has_episode_meta = any((resolved_root / "meta" / "episodes").glob("*/*.parquet"))
        if has_info and has_episode_meta:
            self._dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=str(resolved_root),
                revision="main",
                image_writer_threads=2,
            )
        else:
            needs_create = True
            if resolved_root.exists():
                if has_info and not has_episode_meta:
                    info_path = resolved_root / "meta" / "info.json"
                    try:
                        info = json.loads(info_path.read_text())
                    except Exception as e:
                        raise RuntimeError(
                            "existing dataset root is not finalized/readable "
                            f"(missing meta/episodes parquet and unreadable info.json): {resolved_root}. "
                            "Move that directory aside or choose a new recording root."
                        ) from e
                    total_eps = int(info.get("total_episodes", -1))
                    total_frames = int(info.get("total_frames", -1))
                    if total_eps == 0 and total_frames == 0:
                        # Valid empty dataset after deleting the last episode:
                        # rebuild a clean writable root from info.json.
                        backup = resolved_root.parent / f".{resolved_root.name}.empty-backup-{int(time.time()*1000)}"
                        os.replace(resolved_root, backup)
                        try:
                            self._dataset = LeRobotDataset.create(
                                repo_id=repo_id,
                                fps=int(info["fps"]),
                                features=info["features"],
                                root=str(resolved_root),
                                robot_type=str(info.get("robot_type") or "xlerobot-bimanual-so101"),
                                use_videos=any(
                                    isinstance(v, dict) and v.get("dtype") == "video"
                                    for v in (info.get("features") or {}).values()
                                ),
                                image_writer_threads=2,
                            )
                            self._dataset.finalize()
                            shutil.rmtree(backup, ignore_errors=True)
                            needs_create = False
                        except Exception:
                            if not resolved_root.exists() and backup.exists():
                                os.replace(backup, resolved_root)
                            raise
                    else:
                        raise RuntimeError(
                            "existing dataset root is not finalized/readable "
                            f"(missing meta/episodes parquet): {resolved_root}. "
                            "Move that directory aside or choose a new recording root."
                        )
                else:
                    raise RuntimeError(f"dataset root exists but is not a LeRobot dataset: {resolved_root}")
            if needs_create:
                self._dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=self.fps,
                    features=features,
                    root=str(resolved_root),
                    robot_type="xlerobot-bimanual-so101",
                    use_videos=True,                     # MP4-encode image streams
                    image_writer_threads=2,              # async JPEG → video encode
                )
        self._episode_count = int(getattr(self._dataset.meta, "total_episodes", 0))
        if self._episode_count > 0:
            try:
                last_ep = self._dataset.meta.episodes[self._episode_count - 1]
                self._last_saved_episode_index = self._episode_count - 1
                self._last_saved_episode_frames = int(last_ep.get("length", 0))
            except Exception:
                self._last_saved_episode_index = self._episode_count - 1
                self._last_saved_episode_frames = 0
        log.info("dataset recorder ready: repo_id=%s fps=%d cameras=%s root=%s",
                 repo_id, self.fps, self.camera_roles, self._dataset.root)

    @staticmethod
    def _build_features(joint_order: list[str], camera_roles: list[str],
                         camera_shape: tuple[int, int, int]) -> dict:
        """Construct the LeRobot v2 features dict. The `names` field is required
        for state/action features; for video it's ['height','width','channels']."""
        nj = len(joint_order)
        feats: dict[str, dict[str, Any]] = {
            "action": {
                "dtype": "float32",
                "shape": (nj,),
                "names": list(joint_order),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (nj,),
                "names": list(joint_order),
            },
        }
        for role in camera_roles:
            feats[f"observation.images.{role}"] = {
                "dtype": "video",
                "shape": tuple(camera_shape),         # (H, W, 3)
                "names": ["height", "width", "channels"],
            }
        return feats

    # ── episode lifecycle ───────────────────────────────────────────────────
    @property
    def in_episode(self) -> bool:
        with self._lock:
            return self._in_episode

    @property
    def episode_count(self) -> int:
        with self._lock:
            return self._episode_count

    @property
    def frame_count_in_episode(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def last_saved_episode_index(self) -> Optional[int]:
        with self._lock:
            return self._last_saved_episode_index

    @property
    def last_saved_episode_frames(self) -> int:
        with self._lock:
            return self._last_saved_episode_frames

    def start_episode(self, task: str = "") -> None:
        with self._lock:
            if self._in_episode:
                log.warning("start_episode called while already in episode; ignoring")
                return
            self._in_episode = True
            self._frame_count = 0
            self._current_task = task or "bimanual-vr-teleop"
            log.info("episode %d started (task=%r)", self._episode_count + 1, self._current_task)

    def end_episode(self) -> bool:
        """Save the current episode buffer to disk. Returns True if saved, False
        if there was no active episode or no frames."""
        with self._lock:
            if not self._in_episode:
                log.warning("end_episode called while not in an episode; ignoring")
                return False
            had_frames = self._frame_count > 0
            self._in_episode = False
            if not had_frames:
                # Discard empty buffer rather than save a 0-frame episode.
                try: self._dataset.clear_episode_buffer()
                except Exception as e: log.warning("clear_episode_buffer: %s", e)
                log.info("episode discarded (0 frames)")
                return False
            episode_count = self._episode_count + 1
            frame_count = self._frame_count
        # Save outside the lock — save_episode does I/O + may take a while
        # (especially with batch_encoding_size=1, the video gets encoded now).
        try:
            self._dataset.save_episode()
            with self._lock:
                self._episode_count = episode_count
                self._last_saved_episode_index = episode_count - 1
                self._last_saved_episode_frames = frame_count
            log.info("episode %d saved (%d frames)",
                     episode_count, frame_count)
            return True
        except Exception as e:
            log.exception("save_episode failed: %s", e)
            return False

    def add_frame(self, action: dict[str, float],
                   present: dict[str, float],
                   camera_frames: dict[str, Optional[np.ndarray]]) -> None:
        """Append one frame to the current episode buffer. Caller must already
        have called `start_episode`. `camera_frames` keys must include every
        configured camera role; missing or None entries are replaced with a
        zero-filled frame of the expected shape (so the dataset's schema stays
        stable even if a camera glitches mid-episode)."""
        with self._lock:
            if not self._in_episode:
                return
            action_vec  = self._joint_dict_to_array(action)
            present_vec = self._joint_dict_to_array(present)

            frame: dict[str, Any] = {
                "task": self._current_task,
                "action": action_vec,
                "observation.state": present_vec,
            }
            for role in self.camera_roles:
                img = camera_frames.get(role)
                if img is None or not isinstance(img, np.ndarray):
                    # Fill missing/failed frame with zeros — preserves schema.
                    img = np.zeros(self.camera_shape, dtype=np.uint8)
                else:
                    # Ensure the right shape + dtype.
                    if img.shape != self.camera_shape:
                        img = cv2.resize(img,
                                          (self.camera_shape[1], self.camera_shape[0]))
                    if img.dtype != np.uint8:
                        img = img.astype(np.uint8)
                frame[f"observation.images.{role}"] = img

            try:
                self._dataset.add_frame(frame)
                self._frame_count += 1
            except Exception as e:
                # Don't bring down the drive loop on a single bad frame.
                log.warning("add_frame failed (frame %d): %s", self._frame_count, e)

    @staticmethod
    def _joint_dict_to_array(joints: dict[str, float]) -> np.ndarray:
        """Project a possibly-incomplete joint dict onto the fixed JOINT_ORDER.
        Missing keys → 0.0 (e.g., an arm that isn't connected)."""
        return np.array(
            [float(joints.get(k, 0.0)) for k in JOINT_ORDER],
            dtype=np.float32,
        )

    # ── teardown ────────────────────────────────────────────────────────────
    def finalize(self) -> None:
        """Flush all pending state to disk. If `push_to_hub_flag` is set, push
        the dataset after finalize. Safe to call multiple times."""
        with self._lock:
            if self._in_episode and self._frame_count > 0:
                # Save the in-flight episode before finalizing.
                self._in_episode = False
                try:
                    self._dataset.save_episode()
                    self._episode_count += 1
                except Exception as e:
                    log.warning("finalize: save_episode failed: %s", e)
        try:
            self._dataset.finalize()
            log.info("dataset finalized; %d episode(s) at %s",
                     self._episode_count, self._dataset.root)
        except Exception as e:
            log.warning("finalize failed: %s", e)
        if self.push_to_hub_flag:
            try:
                self._dataset.push_to_hub()
                log.info("pushed dataset %s to Hub", self.repo_id)
            except Exception as e:
                log.warning("push_to_hub failed: %s", e)


# ─── helpers used by VRTeleopSession to build the recorder ─────────────────

def grab_camera_frames() -> dict[str, Optional[np.ndarray]]:
    """Snapshot every role-assigned camera. Returns {role: RGB-ndarray-or-None}.

    Uses the existing CameraStream singletons so we don't open a second
    VideoCapture per camera. Decodes the latest JPEG to numpy via cv2.imdecode,
    then converts OpenCV's BGR layout to RGB for LeRobot/PIL video encoding.
    Subscriptions are reference-counted; we acquire while reading and release
    afterwards (matches how `/camera/<id>/snapshot` works)."""
    out: dict[str, Optional[np.ndarray]] = {}
    cams = cam_mod.enumerate_cameras()
    for c in cams:
        if not c.role:
            continue
        stream = cam_mod.get_stream(c.role)
        if stream is None:
            out[c.role] = None
            continue
        stream.acquire()
        try:
            # Read whatever the producer thread has at the moment — non-blocking.
            jpeg = stream.last_jpeg
            if jpeg is None:
                out[c.role] = None
                continue
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out[c.role] = img
        finally:
            stream.release()
    return out


def load_dataset_config() -> dict[str, Any]:
    """Read the `dataset:` block from config/xlerobot.yaml. Returns sensible
    defaults if the section is missing."""
    defaults = {
        "repo_id": "xlerobot/vr-teleop",
        "fps": 30,
        "push_to_hub": False,
        "task_default": "bimanual-vr-teleop",
        "camera_height": 480,
        "camera_width":  640,
        # If True, every recorded episode begins with all connected arms
        # interpolating to the saved home pose first. Required for VLA data
        # collection so all episodes start from the same proprioception state.
        "home_before_episode": False,
        # Absolute filesystem path where episodes are written. Null/missing
        # → use HuggingFace's default (`$HF_LEROBOT_HOME` or
        # `~/.cache/huggingface/lerobot/<repo_id>/`). Set this in YAML or via
        # the Recording card's "Storage path" input to write elsewhere
        # (e.g., a big external drive).
        "root": None,
    }
    try:
        cfg = yaml.safe_load((REPO_ROOT / "config" / "xlerobot.yaml").read_text()) or {}
        ds = cfg.get("dataset") or {}
        defaults.update({k: v for k, v in ds.items() if v is not None})
    except Exception as e:
        log.warning("could not read dataset config: %s; using defaults", e)
    return defaults


def resolve_root(root: Optional[str], repo_id: str) -> str:
    """Resolve the dataset root path. Handles ~ expansion and falls back to
    the HF default (`$HF_LEROBOT_HOME/<repo_id>` or
    `~/.cache/huggingface/lerobot/<repo_id>`) when None/empty."""
    import os
    if root:
        return os.path.abspath(os.path.expanduser(str(root)))
    hf_home = os.environ.get("HF_LEROBOT_HOME")
    if hf_home:
        return os.path.abspath(os.path.expanduser(os.path.join(hf_home, repo_id)))
    return os.path.abspath(os.path.expanduser(
        f"~/.cache/huggingface/lerobot/{repo_id}"
    ))


def role_camera_list() -> tuple[list[str], tuple[int, int, int]]:
    """Return the list of camera roles to record + a single shared (H, W, 3)
    shape that all roles must conform to (we resize on add_frame if needed)."""
    cfg = load_dataset_config()
    roles: list[str] = []
    for c in cam_mod.enumerate_cameras():
        if c.role:
            roles.append(c.role)
    shape = (int(cfg["camera_height"]), int(cfg["camera_width"]), 3)
    return roles, shape


def delete_last_episode(repo_id: str, root: Optional[str]) -> tuple[int, str]:
    """Delete the most recently saved episode in-place.

    Returns:
        (new_total_episodes, resolved_root_path)
    """
    from lerobot.datasets.dataset_tools import delete_episodes as _delete_episodes
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    resolved_root = pathlib.Path(resolve_root(root, repo_id))
    if not (resolved_root / "meta" / "info.json").is_file():
        raise RuntimeError(f"dataset not found at {resolved_root}")

    dataset = LeRobotDataset(repo_id=repo_id, root=resolved_root)
    total = int(dataset.meta.total_episodes)
    if total <= 0:
        raise RuntimeError("no saved episodes to delete")

    stamp = int(time.time() * 1000)
    backup_root = resolved_root.parent / f".{resolved_root.name}.backup-delete-{stamp}"
    if backup_root.exists():
        raise RuntimeError(f"backup path already exists: {backup_root}")

    # Special case: deleting the sole episode should leave a valid empty dataset.
    if total == 1:
        os.replace(resolved_root, backup_root)
        try:
            ds_new = LeRobotDataset.create(
                repo_id=repo_id,
                fps=int(dataset.meta.fps),
                features=dataset.meta.features,
                root=str(resolved_root),
                robot_type=str(dataset.meta.robot_type),
                use_videos=bool(dataset.meta.video_keys),
                image_writer_threads=2,
            )
            ds_new.finalize()
            shutil.rmtree(backup_root, ignore_errors=True)
            return 0, str(resolved_root)
        except Exception:
            if not resolved_root.exists() and backup_root.exists():
                os.replace(backup_root, resolved_root)
            raise

    # General case: materialize edited dataset in a temp sibling, then atomically swap.
    tmp_root = resolved_root.parent / f".{resolved_root.name}.tmp-delete-{stamp}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)

    _delete_episodes(
        dataset,
        [total - 1],
        output_dir=tmp_root,
        repo_id=repo_id,
    )

    os.replace(resolved_root, backup_root)
    try:
        os.replace(tmp_root, resolved_root)
        # Guardrail: ensure info.json reflects the deletion before committing.
        post = LeRobotDataset(repo_id=repo_id, root=resolved_root)
        if int(post.meta.total_episodes) != (total - 1):
            raise RuntimeError(
                "delete verification failed: info.json total_episodes did not update "
                f"(expected {total - 1}, got {post.meta.total_episodes})"
            )
        shutil.rmtree(backup_root, ignore_errors=True)
    except Exception:
        if not resolved_root.exists() and backup_root.exists():
            os.replace(backup_root, resolved_root)
        elif backup_root.exists():
            # If swap succeeded but verification failed, restore original dataset.
            shutil.rmtree(resolved_root, ignore_errors=True)
            os.replace(backup_root, resolved_root)
        raise
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

    return total - 1, str(resolved_root)


def last_episode_summary(repo_id: str, root: Optional[str]) -> tuple[Optional[int], int]:
    """Return (last_episode_index, last_episode_frames) for a dataset root."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    resolved_root = pathlib.Path(resolve_root(root, repo_id))
    if not (resolved_root / "meta" / "info.json").is_file():
        return None, 0
    dataset = LeRobotDataset(repo_id=repo_id, root=resolved_root)
    total = int(dataset.meta.total_episodes)
    if total <= 0:
        return None, 0
    try:
        ep = dataset.meta.episodes[total - 1]
        return total - 1, int(ep.get("length", 0))
    except Exception:
        return total - 1, 0
