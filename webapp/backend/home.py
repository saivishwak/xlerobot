"""Home-pose persistence for `config/xlerobot.yaml`.

Read/write the `robot.home_pose:` block, preserving all surrounding comments
and structure. PyYAML's `safe_dump` would rewrite the entire file and drop the
comments the user wrote in `xlerobot.yaml`, which they explicitly want kept.
We use a line-based replacement that touches ONLY the home_pose block.

Joint keys are prefixed (`left_arm_shoulder_pan`, etc.) and values are degrees.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any

import yaml

log = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG_PATH = REPO_ROOT / "config" / "xlerobot.yaml"

JOINT_SUFFIXES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)


def read_home_pose() -> dict[str, float]:
    """Load `robot.home_pose` from the YAML, return as a flat {prefixed: deg}
    dict. Returns {} if the block is missing."""
    try:
        cfg = yaml.safe_load(CFG_PATH.read_text()) or {}
    except Exception as e:
        log.warning("could not read %s: %s", CFG_PATH, e)
        return {}
    robot = cfg.get("robot") or {}
    hp = robot.get("home_pose") or {}
    return {str(k): float(v) for k, v in hp.items()}


def home_pose_for_side(side: str) -> dict[str, float]:
    """Subset of the saved home pose for one arm (prefixed keys)."""
    full = read_home_pose()
    prefix = f"{side}_arm_"
    return {k: v for k, v in full.items() if k.startswith(prefix)}


def write_home_pose(new_values: dict[str, float]) -> None:
    """Update the `robot.home_pose:` block in `xlerobot.yaml`, replacing only
    the keys present in `new_values`. Preserves every line outside that block
    exactly — comments, ordering, spacing.

    Strategy:
      1. Read existing home_pose values from YAML (so we don't drop joints
         the user only partially overwrote).
      2. Merge new_values into the existing dict.
      3. Walk the file line-by-line, find the `home_pose:` header under
         `robot:`, identify its indented child block, and replace those
         lines with a freshly rendered version of the merged dict.
    """
    existing = read_home_pose()
    merged = {**existing, **{str(k): float(v) for k, v in new_values.items()}}

    text = CFG_PATH.read_text()
    lines = text.splitlines()

    # Find the home_pose: line inside the robot: block.
    # We look for a line that starts with whitespace followed by `home_pose:`,
    # nested inside the `robot:` top-level block. (We don't support multiple
    # home_pose: blocks; if you have a non-standard layout, edit by hand.)
    in_robot = False
    robot_indent: int | None = None
    hp_line_idx: int | None = None
    hp_indent: int | None = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(stripped)
        if stripped.startswith("robot:"):
            in_robot = True
            robot_indent = indent
            continue
        if in_robot:
            # Top-level key at same indent as `robot:` ends the robot block.
            if indent <= (robot_indent or 0) and stripped.endswith(":"):
                # We left the robot: block before finding home_pose
                if hp_line_idx is None:
                    in_robot = False
                    break
            if stripped.startswith("home_pose:") and indent > (robot_indent or 0):
                hp_line_idx = i
                hp_indent = indent
                break

    # Render the new home_pose body. 6 keys × 2 arms = 12 lines.
    child_indent = (hp_indent if hp_indent is not None else 2) + 2
    new_body: list[str] = []
    for side in ("left", "right"):
        for suffix in JOINT_SUFFIXES:
            key = f"{side}_arm_{suffix}"
            if key in merged:
                new_body.append(f"{' ' * child_indent}{key}: {merged[key]:.4f}")

    if hp_line_idx is None:
        # No home_pose: block found — append one under robot:.
        # If there's no robot: block at all, that's a malformed config; bail.
        if not in_robot:
            raise RuntimeError(
                f"{CFG_PATH} has no 'robot:' block — add one manually before "
                f"using Capture Home."
            )
        # Append after the last child of robot:.
        # Find where the robot: block ends.
        end_idx = len(lines)
        for i in range(len(lines)):
            stripped = lines[i].lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(lines[i]) - len(stripped)
            if indent == (robot_indent or 0) and i > 0 and stripped.endswith(":"):
                # This is a new top-level key after robot:
                if any(lines[j].lstrip().startswith("robot:") for j in range(i)):
                    end_idx = i
                    break
        header = f"{' ' * ((robot_indent or 0) + 2)}home_pose:"
        new_lines = lines[:end_idx] + [header] + new_body + lines[end_idx:]
        CFG_PATH.write_text("\n".join(new_lines) + "\n")
        log.info("home_pose block ADDED to %s (%d joints)", CFG_PATH, len(new_body))
        return

    # Determine end of existing home_pose block (first line at indent <= hp_indent).
    end_idx = hp_line_idx + 1
    while end_idx < len(lines):
        line = lines[end_idx]
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            end_idx += 1
            continue
        indent = len(line) - len(stripped)
        if indent <= (hp_indent or 0):
            break
        end_idx += 1

    # Replace the block: keep the `home_pose:` header line, swap the children.
    out_lines = lines[:hp_line_idx + 1] + new_body + lines[end_idx:]
    CFG_PATH.write_text("\n".join(out_lines) + "\n")
    log.info("home_pose updated in %s (%d joints written)", CFG_PATH, len(new_body))


def home_pose_status() -> dict[str, Any]:
    """Status dict for the UI: per-side captured-or-not + joint values."""
    full = read_home_pose()
    out: dict[str, Any] = {"left": {}, "right": {}}
    for side in ("left", "right"):
        prefix = f"{side}_arm_"
        per_arm = {k: v for k, v in full.items() if k.startswith(prefix)}
        out[side] = {
            "captured": len(per_arm) == len(JOINT_SUFFIXES),
            "joints": per_arm,
        }
    return out
