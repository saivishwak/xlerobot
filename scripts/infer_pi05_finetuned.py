#!/usr/bin/env python3
"""Run on-robot inference with a finetuned LeRobot PI0.5 checkpoint.

Loads the checkpoint locally (no OpenPI server) and drives the XLerobot
bimanual SO-101 stack — the same robot/dataset layout used during VR recording.

Prerequisites:
    bash scripts/setup_xlerobot.sh   # copies xlerobot into lerobot submodule

Usage:
    uv run python scripts/infer_pi05_finetuned.py \\
        --policy-path outputs/pi05_finetune/checkpoints/005000/pretrained_model \\
        --task "Pick up the medicine and place it in the bowl" \\
        --episodes 2 --episode-time 120

    # Dry-run homing only (no policy load, no inference):
    uv run python scripts/infer_pi05_finetuned.py --dry-run-home
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"

# Must match webapp/backend/dataset.py JOINT_ORDER (LeRobot action / observation.state).
_JOINTS_PER_ARM = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
JOINT_ORDER: list[str] = [
    f"{side}_arm_{j}" for side in ("left", "right") for j in _JOINTS_PER_ARM
]

# Per-tick joint caps (degrees) — match webapp/backend/vr_teleop.py PER_TICK_DEG_CAPS.
_PER_TICK_DEG_CAPS: dict[str, float] = {
    "shoulder_pan": 5.0,
    "shoulder_lift": 5.0,
    "elbow_flex": 5.0,
    "wrist_flex": 6.0,
    "wrist_roll": 6.0,
    "gripper": 15.0,
}
_HOMING_TOL_DEG = 0.5
_HOMING_KP = 0.75
# VR recording uses KP=1.0 → action label is rate-limited absolute command (see _shape_action_like_recording).
_DEFAULT_VR_KP = 1.0


def _load_yaml() -> dict[str, Any]:
    if not CONFIG_YAML.is_file():
        sys.exit(f"missing {CONFIG_YAML}")
    return yaml.safe_load(CONFIG_YAML.read_text()) or {}


def _parse_args() -> argparse.Namespace:
    cfg = _load_yaml()
    ds = cfg.get("dataset") or {}
    pi05 = cfg.get("pi05") or {}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--policy-path",
        default=None,
        help="Path to finetuned checkpoint (.../pretrained_model). Not needed with --dry-run-home.",
    )
    p.add_argument(
        "--task",
        default=None,
        help="Natural-language task prompt. Not needed with --dry-run-home.",
    )
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--episode-time", type=int, default=120, help="Max seconds per episode.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument(
        "--fps",
        type=int,
        default=int(ds.get("fps", pi05.get("control_fps", 30))),
        help="Control loop Hz (default: dataset.fps, usually 30 — match training, not pi05.control_fps).",
    )
    p.add_argument(
        "--action-horizon",
        type=int,
        default=int(pi05.get("action_horizon", 50)),
        help="Max chunk size from policy (upper bound).",
    )
    p.add_argument(
        "--open-loop-steps",
        type=int,
        default=35,
        help=(
            "Policy chunk steps before re-inferring (default: 35 @ 30Hz ≈ 1.2s). "
            "Higher = smoother; very low values (e.g. 5) cause jitter at replan boundaries."
        ),
    )
    p.add_argument(
        "--settle-steps",
        type=int,
        default=60,
        help=(
            "Hold present pose for this many control ticks after homing (default: 60 @ 30Hz = 2s). "
            "Matches VR demos where the operator pauses at home before moving."
        ),
    )
    p.add_argument(
        "--replan-blend",
        type=float,
        default=0.25,
        help=(
            "Blend factor for the first action after each new chunk [0..1]. "
            "Lower = smoother across replans; 1.0 disables blending."
        ),
    )
    p.add_argument(
        "--phase1-task",
        default=None,
        help=(
            "Optional language prompt for the first segment (e.g. reach medicine only). "
            "Use with --phase1-sec."
        ),
    )
    p.add_argument(
        "--phase1-sec",
        type=float,
        default=0.0,
        help="Seconds to use --phase1-task before switching to --task (default: 0 = disabled).",
    )
    p.add_argument(
        "--strict-motors",
        action="store_true",
        help="Require all XLerobot motors on the bus (default: lenient prune).",
    )
    p.add_argument(
        "--skip-home",
        action="store_true",
        help="Do not move to saved home pose before inference.",
    )
    p.add_argument(
        "--home-before-episode",
        action=argparse.BooleanOptionalAction,
        default=bool(ds.get("home_before_episode", True)),
        help="Return to home pose at the start of each episode (default: from dataset config).",
    )
    p.add_argument(
        "--home-timeout",
        type=float,
        default=60.0,
        help="Max seconds to spend homing before continuing anyway.",
    )
    p.add_argument(
        "--max-relative-target",
        type=float,
        default=None,
        help="Max joint change (deg) per policy command (default: robot.max_relative_target in yaml).",
    )
    p.add_argument(
        "--policy-ema-alpha",
        type=float,
        default=0.34,
        help=(
            "EMA on raw policy targets before VR shaping [0..1]. "
            "Lower is smoother; 1.0 disables."
        ),
    )
    p.add_argument(
        "--command-ema-alpha",
        type=float,
        default=0.22,
        help=(
            "EMA smoothing for final command [0..1]. Lower is smoother, higher is snappier. "
            "Too low (~0.14) can stall at home; too high (~0.28) reaches targets but jitters."
        ),
    )
    p.add_argument(
        "--joint-deadband-deg",
        type=float,
        default=0.75,
        help=(
            "Suppress command updates smaller than this vs the previous filtered command (deg). "
            "Too high (~1.0) can stall at home; too low (~0.6) is snappy but jittery."
        ),
    )
    p.add_argument(
        "--clamp-to-present",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Clamp each command vs measured joint pose (robot.max_relative_target). "
            "Default off: training labels are rate-limited vs the previous command, not "
            "present; present clamp often causes oscillation during policy chunks."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned settings and exit (no robot connection).",
    )
    p.add_argument(
        "--dry-run-home",
        action="store_true",
        help="Connect and move to saved home pose only; skip policy load and inference.",
    )
    args = p.parse_args()
    if args.dry_run and args.dry_run_home:
        p.error("use only one of --dry-run or --dry-run-home")
    if not args.dry_run and not args.dry_run_home:
        if not args.policy_path or not args.task:
            p.error("--policy-path and --task are required for inference")
    return args


def _ensure_xlerobot_import() -> None:
    try:
        import lerobot.robots.xlerobot  # noqa: F401
    except ImportError as e:
        sys.exit(
            "lerobot.robots.xlerobot is not installed.\n"
            "Run: bash scripts/setup_xlerobot.sh\n"
            f"Original error: {e}"
        )


def _build_state_vector(obs: dict[str, Any], joint_names: list[str]) -> np.ndarray:
    return np.array([float(obs[f"{name}.pos"]) for name in joint_names], dtype=np.float32)


def _to_hwc_uint8_image(img: Any) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"camera image must be rank-3, got shape={arr.shape}")
    # Accept both CHW and HWC, normalize to HWC uint8 for prepare_observation_for_inference.
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0 + 1e-6:
                arr = np.clip(arr, 0.0, 1.0)
                arr = np.rint(arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _build_observation(
    obs: dict[str, Any],
    joint_names: list[str],
) -> dict[str, np.ndarray]:
    camera_obs = {k: v for k, v in obs.items() if k in ("head", "left_wrist", "right_wrist")}
    if set(camera_obs) != {"head", "left_wrist", "right_wrist"}:
        missing = {"head", "left_wrist", "right_wrist"} - set(camera_obs)
        sys.exit(f"missing camera observations: {sorted(missing)}")
    return {
        "observation.images.head": _to_hwc_uint8_image(camera_obs["head"]),
        "observation.images.left_wrist": _to_hwc_uint8_image(camera_obs["left_wrist"]),
        "observation.images.right_wrist": _to_hwc_uint8_image(camera_obs["right_wrist"]),
        "observation.state": _build_state_vector(obs, joint_names),
    }


def _read_home_pose(cfg: dict[str, Any]) -> dict[str, float]:
    hp = (cfg.get("robot") or {}).get("home_pose") or {}
    if not hp:
        sys.exit(
            "robot.home_pose is empty in config/xlerobot.yaml.\n"
            "Capture a home pose first (webapp VR Teleop → Capture home, or "
            "scripts/save_home_pose.py)."
        )
    return {str(k): float(v) for k, v in hp.items()}


def _cap_for_joint_key(key: str) -> float:
    for suffix, cap in _PER_TICK_DEG_CAPS.items():
        if f"_{suffix}.pos" in key:
            return cap
    return 2.0


def _vr_kp(cfg: dict[str, Any]) -> float:
    return float((cfg.get("vr") or {}).get("kp", _DEFAULT_VR_KP))


def _shape_action_like_recording(
    policy_targets: dict[str, float],
    present: dict[str, float],
    last_sent: dict[str, float],
    *,
    kp: float,
) -> dict[str, float]:
    """Match VR dataset labels: per-tick cap vs previous command, optional P blend.

    Recording stores `action` = motor command after the same logic in vr_teleop.py
    (not the raw policy/VR target). With kp>=0.999 this is:
        cmd = last_sent + clip(target - last_sent, -cap, cap)
    """
    shaped: dict[str, float] = {}
    for key, target in policy_targets.items():
        cap = _cap_for_joint_key(key)
        prev = last_sent.get(key, present.get(key, target))
        delta = max(-cap, min(cap, target - prev))
        clamped = prev + delta
        if kp >= 0.999:
            shaped[key] = clamped
        else:
            here = present.get(key, clamped)
            shaped[key] = here + kp * (clamped - here)
    return shaped


def _clamp_max_relative(
    command: dict[str, float],
    present: dict[str, float],
    max_rel: float,
) -> dict[str, float]:
    """Clamp absolute goals vs present (same semantics as XLerobot max_relative_target)."""
    out: dict[str, float] = {}
    for key, goal in command.items():
        here = present.get(key, goal)
        delta = max(-max_rel, min(max_rel, goal - here))
        out[key] = here + delta
    return out


def _top_joint_deltas(command: dict[str, float], present: dict[str, float], top_k: int = 4) -> list[tuple[str, float]]:
    deltas = []
    for key, goal in command.items():
        here = present.get(key, goal)
        deltas.append((key, float(goal - here)))
    deltas.sort(key=lambda x: abs(x[1]), reverse=True)
    return deltas[:top_k]


def _apply_joint_deadband(
    command: dict[str, float],
    reference: dict[str, float],
    deadband_deg: float,
) -> dict[str, float]:
    if deadband_deg <= 0:
        return dict(command)
    out: dict[str, float] = {}
    for key, goal in command.items():
        ref = reference.get(key, goal)
        out[key] = ref if abs(goal - ref) < deadband_deg else goal
    return out


def _blend_action_dict(
    new_cmd: dict[str, float],
    prev_cmd: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    if alpha >= 0.999 or not prev_cmd:
        return dict(new_cmd)
    if alpha <= 0:
        return dict(prev_cmd)
    out: dict[str, float] = {}
    for key, goal in new_cmd.items():
        prev = prev_cmd.get(key, goal)
        out[key] = (1.0 - alpha) * prev + alpha * goal
    return out


def _task_for_step(
    args: argparse.Namespace,
    *,
    step: int,
    settle_steps: int,
    fps: float,
) -> str:
    if args.phase1_task and args.phase1_sec > 0:
        phase1_end = settle_steps + int(args.phase1_sec * fps)
        if step < phase1_end:
            return args.phase1_task
    return args.task


def _ema_command(
    command: dict[str, float],
    prev_command: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    if alpha >= 0.999:
        return dict(command)
    if alpha <= 0:
        return dict(prev_command) if prev_command else dict(command)
    out: dict[str, float] = {}
    for key, goal in command.items():
        prev = prev_command.get(key, goal)
        out[key] = (1.0 - alpha) * prev + alpha * goal
    return out


def _send_positions(
    robot: Any,
    command: dict[str, float],
    *,
    present: dict[str, float] | None = None,
) -> None:
    """Send joint goals; clamp here so older XLerobot send_action bugs are avoided."""
    max_rel = getattr(robot.config, "max_relative_target", None)
    if present is not None and max_rel is not None:
        command = _clamp_max_relative(command, present, float(max_rel))
    saved_max_rel = robot.config.max_relative_target
    robot.config.max_relative_target = None
    try:
        robot.send_action(command)
    finally:
        robot.config.max_relative_target = saved_max_rel


def _connect_robot_with_retries(
    robot: Any,
    *,
    attempts: int = 4,
    retry_sleep_s: float = 1.5,
) -> None:
    """Connect robot/cameras with retry for transient OpenCV warmup failures."""
    last_err: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            robot.connect(calibrate=False)
            if attempt > 1:
                print(f"Robot connected on retry {attempt}/{attempts}.")
            return
        except Exception as e:  # pragma: no cover - hardware path
            last_err = e
            msg = str(e)
            camera_hint = "OpenCVCamera" in msg or "Timed out waiting for frame from camera" in msg
            print(
                f"Connect attempt {attempt}/{attempts} failed: {e.__class__.__name__}: {msg}"
                + (" (camera warmup/read issue)" if camera_hint else "")
            )
            try:
                robot.disconnect()
            except Exception:
                pass
            if attempt < attempts:
                time.sleep(retry_sleep_s)
    assert last_err is not None
    raise last_err


def go_to_home_pose(
    robot: Any,
    home_pose: dict[str, float],
    *,
    fps: float,
    timeout_s: float,
    precise_sleep: Any,
) -> None:
    """Rate-limited move to `robot.home_pose` joint targets (degrees)."""
    targets = {f"{name}.pos": deg for name, deg in home_pose.items()}
    keys = list(targets.keys())
    last_sent: dict[str, float] = {}
    dt = 1.0 / fps
    deadline = time.perf_counter() + timeout_s

    print(f"Homing to saved pose ({len(keys)} joints, timeout={timeout_s:.0f}s)...")
    while time.perf_counter() < deadline:
        loop_start = time.perf_counter()
        obs = robot.get_observation(include_cameras=False)
        present = {k: float(obs[k]) for k in keys if k in obs}

        clamped: dict[str, float] = {}
        converged = True
        for key, target in targets.items():
            cap = _cap_for_joint_key(key)
            prev = last_sent.get(key, present.get(key, target))
            step = max(-cap, min(cap, target - prev))
            clamped[key] = prev + step
            if abs(clamped[key] - target) > _HOMING_TOL_DEG:
                converged = False

        command = {
            key: present.get(key, tgt) + _HOMING_KP * (tgt - present.get(key, tgt))
            for key, tgt in clamped.items()
        }
        _send_positions(robot, command, present=present)
        last_sent = dict(command)

        if converged:
            print("Home pose reached.")
            return

        remaining = dt - (time.perf_counter() - loop_start)
        if remaining > 0:
            precise_sleep(remaining)

    print("Warning: homing timed out; continuing anyway.")


def _run_home_only(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """Connect, homing only, disconnect — no policy."""
    home_pose = _read_home_pose(cfg)
    _ensure_xlerobot_import()
    from _xlerobot_loader import make_config, patch_motors_bus_lenient
    from lerobot.robots.xlerobot import XLerobot
    from lerobot.utils.robot_utils import precise_sleep

    if not args.strict_motors:
        patch_motors_bus_lenient()

    robot = XLerobot(make_config(robot_id="xlerobot"))
    _connect_robot_with_retries(robot)
    try:
        go_to_home_pose(
            robot,
            home_pose,
            fps=float(args.fps),
            timeout_s=float(args.home_timeout),
            precise_sleep=precise_sleep,
        )
        print("Dry-run home complete.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        robot.disconnect()


def _verify_policy_checkpoint(policy_path: pathlib.Path, policy: Any) -> None:
    """Log that a local finetuned checkpoint (not only base weights) is loaded."""
    weights = policy_path / "model.safetensors"
    if not weights.is_file():
        sys.exit(f"missing finetuned weights: {weights}")
    train_step = policy_path.parent / "training_state" / "training_step.json"
    step = "unknown"
    if train_step.is_file():
        step = str(json.loads(train_step.read_text()).get("step", "?"))
    print(f"  Weights file      : {weights} ({weights.stat().st_size // 1_000_000} MB)")
    print(f"  Training step     : {step}")
    print(f"  Base pretrained   : {getattr(policy.config, 'pretrained_path', 'n/a')}")
    print(f"  Action joints     : {list(policy.config.action_feature_names or [])}")


def _load_pi05_policy_with_compat(policy_path: pathlib.Path, device: torch.device) -> Any:
    """Load PI05 strictly via root-level shared compatibility loader."""
    from _pi05_loader import load_pi05_policy_with_compat

    return load_pi05_policy_with_compat(policy_path, device)


def _actions_to_robot_dict(action_row: torch.Tensor, joint_names: list[str]) -> dict[str, float]:
    row = action_row.detach().cpu().numpy().reshape(-1)
    if row.shape[0] != len(joint_names):
        raise ValueError(f"expected {len(joint_names)} action dims, got {row.shape[0]}")
    return {f"{name}.pos": float(row[i]) for i, name in enumerate(joint_names)}


def main() -> None:
    cfg = _load_yaml()
    args = _parse_args()

    if args.dry_run_home:
        print("=" * 60)
        print("  Mode              : dry-run-home (homing only)")
        print(f"  FPS               : {args.fps}")
        print(f"  Home timeout      : {args.home_timeout}s")
        print("=" * 60)
        _run_home_only(args, cfg)
        return

    policy_path = pathlib.Path(args.policy_path).resolve()
    if not policy_path.is_dir():
        sys.exit(f"policy path not found: {policy_path}")

    print("=" * 60)
    print(f"  Policy checkpoint : {policy_path}")
    print(f"  Task              : {args.task}")
    print(f"  Episodes          : {args.episodes} x <= {args.episode_time}s @ {args.fps} fps")
    print(f"  Action horizon    : {args.action_horizon} (chunk cap)")
    print(f"  Open-loop steps   : {args.open_loop_steps} (re-infer interval)")
    print(f"  Settle steps      : {args.settle_steps} (hold pose after homing)")
    print(f"  Replan blend      : {args.replan_blend}")
    if args.phase1_task and args.phase1_sec > 0:
        print(f"  Phase-1 task      : {args.phase1_task!r} for {args.phase1_sec}s")
    print(f"  Device            : {args.device}")
    print(f"  Home before run   : {not args.skip_home}")
    print(f"  Home per episode  : {args.home_before_episode and not args.skip_home}")
    robot_cfg_yaml = cfg.get("robot") or {}
    max_rel = args.max_relative_target
    if max_rel is None:
        max_rel = robot_cfg_yaml.get("max_relative_target")
    print(
        f"  Clamp to present  : {args.clamp_to_present}"
        + (f" (max {max_rel} deg)" if args.clamp_to_present and max_rel is not None else "")
    )
    print(f"  Policy EMA alpha  : {args.policy_ema_alpha}")
    print(f"  Command EMA alpha : {args.command_ema_alpha}")
    print(f"  Joint deadband    : {args.joint_deadband_deg} deg")
    print("=" * 60)
    if args.dry_run:
        return

    home_pose = _read_home_pose(cfg) if not args.skip_home else {}

    _ensure_xlerobot_import()
    from _xlerobot_loader import make_config, patch_motors_bus_lenient
    from lerobot.common.control_utils import predict_action
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.robots.xlerobot import XLerobot
    from lerobot.utils.robot_utils import precise_sleep

    if not args.strict_motors:
        patch_motors_bus_lenient()

    if args.max_relative_target is not None:
        cfg.setdefault("robot", {})["max_relative_target"] = args.max_relative_target

    device = torch.device(args.device)
    policy = _load_pi05_policy_with_compat(policy_path, device)
    _verify_policy_checkpoint(policy_path, policy)

    policy_joint_names = list(policy.config.action_feature_names or [])
    if len(policy_joint_names) != 12:
        sys.exit(f"expected 12 action joints in checkpoint config, got {len(policy_joint_names)}")
    if policy_joint_names != JOINT_ORDER:
        print(
            "Warning: checkpoint action_feature_names order differs from dataset "
            f"JOINT_ORDER.\n  checkpoint: {policy_joint_names}\n  dataset:   {JOINT_ORDER}"
        )

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(policy_path),
    )

    robot_cfg = make_config(robot_id="xlerobot")
    robot = XLerobot(robot_cfg)
    _connect_robot_with_retries(robot)
    import lerobot.robots.xlerobot.xlerobot as xlerobot_module

    print(f"  XLerobot driver    : {xlerobot_module.__file__}")

    dt = 1.0 / float(args.fps)
    vr_kp = _vr_kp(cfg)
    print(f"  VR-style cmd shape : kp={vr_kp} per-tick caps (matches dataset action labels)")
    last_sent: dict[str, float] = {}
    last_filtered: dict[str, float] = {}
    last_policy_smoothed: dict[str, float] = {}
    try:
        if home_pose and not args.home_before_episode:
            go_to_home_pose(
                robot,
                home_pose,
                fps=float(args.fps),
                timeout_s=float(args.home_timeout),
                precise_sleep=precise_sleep,
            )

        for ep in range(args.episodes):
            print(f"\n=== Episode {ep + 1}/{args.episodes} ===")
            if home_pose and args.home_before_episode:
                go_to_home_pose(
                    robot,
                    home_pose,
                    fps=float(args.fps),
                    timeout_s=float(args.home_timeout),
                    precise_sleep=precise_sleep,
                )
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()
            last_sent.clear()
            last_filtered.clear()
            last_policy_smoothed.clear()
            policy.config.n_action_steps = min(
                int(args.open_loop_steps),
                int(policy.config.chunk_size),
            )

            t_end = time.perf_counter() + args.episode_time
            step = 0
            logged_action_debug = False
            settle_steps = max(0, int(args.settle_steps))

            while time.perf_counter() < t_end:
                loop_start = time.perf_counter()

                raw_obs = robot.get_observation()
                present_dict = {f"{n}.pos": float(raw_obs[f"{n}.pos"]) for n in JOINT_ORDER}

                if step < settle_steps:
                    hold_cmd = dict(present_dict)
                    _send_positions(robot, hold_cmd, present=present_dict)
                    last_sent = dict(hold_cmd)
                    last_filtered = dict(hold_cmd)
                    step += 1
                    remaining = dt - (time.perf_counter() - loop_start)
                    if remaining > 0:
                        precise_sleep(remaining)
                    continue

                if step == settle_steps:
                    policy.reset()
                    preprocessor.reset()
                    postprocessor.reset()
                    last_sent.clear()
                    last_filtered.clear()
                    last_policy_smoothed.clear()

                obs_frame = _build_observation(raw_obs, JOINT_ORDER)
                task_prompt = _task_for_step(
                    args, step=step, settle_steps=settle_steps, fps=float(args.fps)
                )
                queue_empty_before = len(getattr(policy, "_action_queue", ())) == 0
                action = predict_action(
                    obs_frame,
                    policy,
                    device,
                    preprocessor,
                    postprocessor,
                    use_amp=bool(getattr(policy.config, "use_amp", False)),
                    task=task_prompt,
                    robot_type=robot.name,
                )
                action_dict = _actions_to_robot_dict(action, policy_joint_names)
                if args.policy_ema_alpha < 0.999:
                    policy_ref = last_policy_smoothed if last_policy_smoothed else action_dict
                    action_dict = _ema_command(
                        action_dict, policy_ref, float(args.policy_ema_alpha)
                    )
                    last_policy_smoothed = dict(action_dict)
                if queue_empty_before and last_filtered and args.replan_blend < 0.999:
                    action_dict = _blend_action_dict(
                        action_dict, last_filtered, float(args.replan_blend)
                    )
                shaped = _shape_action_like_recording(
                    action_dict,
                    present_dict,
                    last_sent,
                    kp=vr_kp,
                )
                final_cmd = shaped
                if args.clamp_to_present:
                    max_rel_cfg = getattr(robot.config, "max_relative_target", None)
                    if max_rel_cfg is not None:
                        final_cmd = _clamp_max_relative(
                            shaped, present_dict, float(max_rel_cfg)
                        )
                final_cmd = _apply_joint_deadband(
                    final_cmd, last_filtered, float(args.joint_deadband_deg)
                )
                final_cmd = _ema_command(final_cmd, last_filtered, float(args.command_ema_alpha))
                if not logged_action_debug:
                    present = _build_state_vector(raw_obs, JOINT_ORDER)
                    raw_cmd = np.array([action_dict[f"{n}.pos"] for n in JOINT_ORDER])
                    shaped_cmd = np.array([shaped[f"{n}.pos"] for n in JOINT_ORDER])
                    sent_cmd = np.array([final_cmd[f"{n}.pos"] for n in JOINT_ORDER])
                    raw_delta = raw_cmd - present
                    shaped_delta = shaped_cmd - present
                    final_delta = sent_cmd - present
                    print(
                        f"  First step |policy-present| max={np.abs(raw_delta).max():.2f} deg "
                        f"(raw policy, before VR shaping)"
                    )
                    print(
                        f"  First step |shaped-present| max={np.abs(shaped_delta).max():.2f} deg "
                        f"(after VR shaping only)"
                    )
                    sent_note = (
                        "after present clamp"
                        if args.clamp_to_present
                        else "after EMA/deadband (VR caps only)"
                    )
                    print(
                        f"  First step |sent-present|   max={np.abs(final_delta).max():.2f} deg "
                        f"(final command {sent_note})"
                    )
                    for name, delta in _top_joint_deltas(final_cmd, present_dict):
                        print(f"    sent delta {name}: {delta:+.2f} deg")
                    logged_action_debug = True

                send_present = present_dict if args.clamp_to_present else None
                _send_positions(robot, final_cmd, present=send_present)
                last_sent = dict(final_cmd)
                last_filtered = dict(final_cmd)

                step += 1
                remaining = dt - (time.perf_counter() - loop_start)
                if remaining > 0:
                    precise_sleep(remaining)

            print(f"Episode {ep + 1} finished ({step} control steps).")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
