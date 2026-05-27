#!/usr/bin/env python3
"""Push a local LeRobot dataset directory to Hugging Face Hub.

Examples:
  uv run python scripts/push_dataset.py
  uv run python scripts/push_dataset.py --repo-id saivishwak/xlerobot-vr-teleop
  uv run python scripts/push_dataset.py --root /custom/path/to/dataset
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"


def _load_dataset_cfg() -> dict:
    if not CONFIG_YAML.is_file():
        return {}
    try:
        cfg = yaml.safe_load(CONFIG_YAML.read_text()) or {}
    except Exception:
        return {}
    return cfg.get("dataset") or {}


def _resolve_root(repo_id: str, root: str | None) -> pathlib.Path:
    if root:
        return pathlib.Path(root).expanduser().resolve()
    import os

    env_home = os.environ.get("HF_LEROBOT_HOME")
    if env_home:
        hf_home = pathlib.Path(env_home).expanduser().resolve()
    else:
        hf_home = pathlib.Path("~/.cache/huggingface/lerobot").expanduser().resolve()
    return (hf_home / repo_id).resolve()


def _parse_args() -> argparse.Namespace:
    ds_cfg = _load_dataset_cfg()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-id",
        default=ds_cfg.get("repo_id", "saivishwak/xlerobot-vr-teleop"),
        help="HF dataset repo id (e.g. user/name).",
    )
    p.add_argument(
        "--root",
        default=ds_cfg.get("root"),
        help=(
            "Local dataset root path. If omitted, uses dataset.root from config; "
            "otherwise falls back to $HF_LEROBOT_HOME/<repo_id> or ~/.cache/huggingface/lerobot/<repo_id>."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved settings and exit without uploading.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = _resolve_root(str(args.repo_id), args.root)

    print(f"Repo ID        : {args.repo_id}")
    print(f"Local root     : {root}")

    if not root.is_dir():
        sys.exit(f"dataset root not found: {root}")
    if not (root / "meta" / "info.json").is_file():
        sys.exit(f"not a LeRobot dataset root (missing meta/info.json): {root}")
    if args.dry_run:
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=str(args.repo_id), root=root)
    ds.push_to_hub()
    print(f"Uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
