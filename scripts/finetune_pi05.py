#!/usr/bin/env python3
"""Finetune a PI0.5 policy on the current LeRobot dataset.

This script is a thin wrapper around LeRobot's native training entrypoint:
`lerobot/src/lerobot/scripts/lerobot_train.py`.

Defaults are loaded from `config/xlerobot.yaml`:
  - dataset.repo_id
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"


def _load_yaml() -> dict[str, Any]:
    if not CONFIG_YAML.is_file():
        sys.exit(f"missing {CONFIG_YAML}")
    return yaml.safe_load(CONFIG_YAML.read_text()) or {}


def _parse_args() -> argparse.Namespace:
    cfg = _load_yaml()
    ds = cfg.get("dataset") or {}
    default_repo = str(ds.get("repo_id") or "your-org/your-dataset")

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-repo-id", default=default_repo, help="LeRobot dataset repo_id.")
    p.add_argument("--pretrained-path", default="lerobot/pi05_base",
                   help="Base checkpoint to finetune from (HF id or local path).")
    p.add_argument("--output-dir", default="outputs/pi05_finetune", help="Training output directory.")
    p.add_argument("--job-name", default="pi05_finetune_xlerobot", help="Training job name.")
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--log-freq", type=int, default=100)
    p.add_argument("--save-freq", type=int, default=5_000)
    p.add_argument("--eval-freq", type=int, default=5_000)
    p.add_argument("--policy-repo-id", default="",
                   help="Optional HF repo id for pushing policy checkpoints.")
    p.add_argument("--push-to-hub", action="store_true",
                   help="If set, push policy checkpoints to Hugging Face Hub.")
    p.add_argument("--wandb-enable", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument(
        "--oom-safe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable low-memory training settings (default: enabled). Use --no-oom-safe to disable.",
    )
    p.add_argument(
        "--cuda-alloc-conf",
        default="expandable_segments:True",
        help="Value for PYTORCH_CUDA_ALLOC_CONF passed to training subprocess.",
    )
    p.add_argument(
        "--rename-map-json",
        default='{"observation.images.head":"observation.images.base_0_rgb","observation.images.left_wrist":"observation.images.left_wrist_0_rgb","observation.images.right_wrist":"observation.images.right_wrist_0_rgb"}',
        help="JSON mapping from dataset observation keys to policy expected keys.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print command without executing.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from an existing checkpoint (uses --output-dir/checkpoints/last by default).",
    )
    p.add_argument(
        "--resume-from",
        default="",
        help=(
            "Optional resume source path. Can be a run dir (with checkpoints/), "
            "a checkpoint dir (.../checkpoints/020000), or a train_config.json path."
        ),
    )
    return p.parse_args()


def _resolve_resume_config_path(args: argparse.Namespace) -> pathlib.Path:
    candidates: list[pathlib.Path] = []
    if args.resume_from:
        src = pathlib.Path(args.resume_from).expanduser().resolve()
        if src.is_file():
            candidates.append(src)
        else:
            candidates.append(src / "train_config.json")
            candidates.append(src / "pretrained_model" / "train_config.json")
            candidates.append(src / "checkpoints" / "last" / "pretrained_model" / "train_config.json")
    else:
        out = pathlib.Path(args.output_dir).expanduser().resolve()
        candidates.append(out / "checkpoints" / "last" / "pretrained_model" / "train_config.json")

    for p in candidates:
        if p.is_file():
            return p

    tried = "\n  - ".join(str(p) for p in candidates)
    sys.exit(
        "Could not find resume train_config.json. Tried:\n"
        f"  - {tried}\n"
        "Pass --resume-from with a valid checkpoint/run path."
    )


def _build_cmd(args: argparse.Namespace) -> tuple[list[str], pathlib.Path | None]:
    if args.resume:
        config_path = _resolve_resume_config_path(args)
        cmd = [
            "uv",
            "run",
            "lerobot-train",
            f"--config_path={config_path}",
            "--resume=true",
        ]
        return cmd, config_path

    try:
        rename_map = json.loads(args.rename_map_json)
    except json.JSONDecodeError as e:
        sys.exit(f"invalid --rename-map-json: {e}")
    rename_map_json = json.dumps(rename_map, separators=(",", ":"))

    effective_batch_size = args.batch_size
    freeze_vision = "false"
    train_expert_only = "false"
    if args.oom_safe:
        # PI0.5 full finetuning is memory-heavy on 24GB cards; default to safer settings.
        effective_batch_size = min(args.batch_size, 2)
        freeze_vision = "true"
        train_expert_only = "true"

    cmd = [
        "uv",
        "run",
        "lerobot-train",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--policy.path={args.pretrained_path}",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--steps={args.steps}",
        f"--batch_size={effective_batch_size}",
        f"--num_workers={args.num_workers}",
        f"--policy.device={args.device}",
        f"--policy.dtype={args.dtype}",
        "--policy.gradient_checkpointing=true",
        f"--policy.freeze_vision_encoder={freeze_vision}",
        f"--policy.train_expert_only={train_expert_only}",
        "--policy.push_to_hub=false",
        f"--rename_map={rename_map_json}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq}",
        f"--eval_freq={args.eval_freq}",
        f"--wandb.enable={'true' if args.wandb_enable else 'false'}",
    ]
    if args.push_to_hub:
        cmd.append("--policy.push_to_hub=true")
        if args.policy_repo_id:
            cmd.append(f"--policy.repo_id={args.policy_repo_id}")
    return cmd, None


def main() -> None:
    args = _parse_args()
    cmd, resume_cfg = _build_cmd(args)
    print("Running:\n  " + " \\\n  ".join(shlex.quote(x) for x in cmd))
    if args.resume and resume_cfg is not None:
        print(f"Resume mode enabled from: {resume_cfg}")
    elif args.oom_safe:
        print("OOM-safe mode enabled: batch_size capped at 2, vision encoder frozen, expert-only training enabled.")
    if args.dry_run:
        return
    env = dict(os.environ)
    if args.cuda_alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


if __name__ == "__main__":
    main()

