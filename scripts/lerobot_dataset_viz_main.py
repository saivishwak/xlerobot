#!/usr/bin/env python3
"""Run LeRobot dataset viz against a non-version Hub revision.

This keeps the vendored LeRobot submodule untouched while avoiding its
``get_safe_version()`` fallback on untagged v3.0 datasets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_cache() -> None:
    project_root = Path(__file__).resolve().parents[1]
    datasets_cache = project_root / ".cache" / "huggingface" / "datasets"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))


def main() -> None:
    configure_cache()

    from lerobot.datasets import LeRobotDataset
    from lerobot.scripts import lerobot_dataset_viz

    revision = os.environ.get("LEROBOT_DATASET_REVISION", "main")
    if "--num-workers" not in sys.argv:
        sys.argv.extend(["--num-workers", "0"])

    def load_dataset_from_revision(repo_id: str, *args, **kwargs):
        kwargs.setdefault("revision", revision)
        return LeRobotDataset(repo_id, *args, **kwargs)

    lerobot_dataset_viz.LeRobotDataset = load_dataset_from_revision
    lerobot_dataset_viz.main()


if __name__ == "__main__":
    main()
