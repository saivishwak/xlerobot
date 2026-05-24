"""Per-arm VR→robot calibration persistence.

The calibration wizard captures a 3×3 rotation matrix per arm that maps VR-world
coordinates to robot base coordinates. Re-running the wizard every session is
tedious — so we save the result to `config/vr_calibration.yaml` and reload it
on startup. Users can keep using the same calibration as long as their VR setup
(headset position, where they stand) hasn't changed.

File format (`config/vr_calibration.yaml`):

    left:
      session_vr_to_robot:
        - [m00, m01, m02]
        - [m10, m11, m12]
        - [m20, m21, m22]
      calibrated_at: '2026-05-24T12:34:56'
      forward_motion_m: 0.103
      up_motion_m: 0.092
      left_motion_m: 0.088
      invert_lateral: false
      confidence: good
      wrist_flex_sign: 1.0
      wrist_roll_sign: -1.0
    right: { ... same shape ... }

This file is auto-managed by the calibration wizard. Sides that haven't been
calibrated yet are simply absent. Edit by re-running the wizard, not by hand.
"""
from __future__ import annotations

import datetime
import logging
import pathlib
from typing import Any

import numpy as np
import yaml

log = logging.getLogger(__name__)
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG_PATH = REPO_ROOT / "config" / "vr_calibration.yaml"


def read_all() -> dict[str, dict[str, Any]]:
    """Load all per-arm calibrations. Returns {} if the file doesn't exist yet."""
    if not CFG_PATH.exists():
        return {}
    try:
        return yaml.safe_load(CFG_PATH.read_text()) or {}
    except Exception as e:
        log.warning("could not read %s: %s", CFG_PATH, e)
        return {}


def read_for_arm(side: str) -> dict[str, Any] | None:
    """Calibration data for one arm, or None if not yet saved."""
    return (read_all() or {}).get(side)


def matrix_for_arm(side: str) -> np.ndarray | None:
    """The 3×3 session_vr_to_robot matrix for one arm, or None if not saved
    OR if the saved data is malformed (wrong shape, bad values)."""
    data = read_for_arm(side)
    if not data:
        return None
    raw = data.get("session_vr_to_robot")
    if not raw:
        return None
    try:
        M = np.array(raw, dtype=float)
        if M.shape != (3, 3):
            log.warning("[%s] saved session_vr_to_robot has wrong shape %s; ignoring",
                        side, M.shape)
            return None
        if not np.all(np.isfinite(M)):
            log.warning("[%s] saved session_vr_to_robot has non-finite values; ignoring", side)
            return None
        ortho = _orthonormalize_matrix(M)
        error = float(np.linalg.norm(M.T @ M - np.eye(3)))
        if error > 0.05:
            log.warning(
                "[%s] saved session_vr_to_robot is too skewed (orthogonality error %.3f); ignoring",
                side, error,
            )
            return None
        if error > 1e-3:
            log.warning(
                "[%s] saved session_vr_to_robot was slightly non-orthonormal (%.4f); using closest rotation",
                side, error,
            )
        return ortho
    except Exception as e:
        log.warning("[%s] saved session_vr_to_robot is malformed: %s; ignoring", side, e)
        return None


def write_for_arm(side: str, matrix: np.ndarray,
                   forward_motion_m: float = 0.0,
                   up_motion_m: float = 0.0,
                   left_motion_m: float = 0.0,
                   invert_lateral: bool | None = None,
                   confidence: str = "good",
                   wrist_flex_sign: float | None = None,
                   wrist_roll_sign: float | None = None) -> None:
    """Persist one arm's calibration. Preserves other arms' entries by reading
    the file first, mutating, and writing back. Note: invert toggles live in
    config/xlerobot.yaml's `vr:` section, not here."""
    existing = read_all()
    M = _orthonormalize_matrix(np.array(matrix, dtype=float))
    entry: dict[str, Any] = {
        "session_vr_to_robot": [[float(v) for v in row] for row in M],
        "calibrated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "forward_motion_m": float(forward_motion_m),
        "up_motion_m": float(up_motion_m),
        "left_motion_m": float(left_motion_m),
        "confidence": str(confidence or "good"),
    }
    if invert_lateral is not None:
        entry["invert_lateral"] = bool(invert_lateral)
    if wrist_flex_sign is not None:
        entry["wrist_flex_sign"] = 1.0 if float(wrist_flex_sign) >= 0 else -1.0
    if wrist_roll_sign is not None:
        entry["wrist_roll_sign"] = 1.0 if float(wrist_roll_sign) >= 0 else -1.0
    existing[side] = entry
    CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# VR→robot calibration data. Auto-managed by the calibration wizard in\n"
        "# the webapp — edit by re-running the wizard, not by hand.\n"
    )
    body = yaml.safe_dump(existing, sort_keys=False, default_flow_style=None)
    CFG_PATH.write_text(header + body)
    log.info("[%s] VR calibration saved to %s", side, CFG_PATH)


def status() -> dict[str, dict[str, Any]]:
    """Status dict for the UI — per side, indicates whether saved + when."""
    saved = read_all()
    out: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        data = saved.get(side) or {}
        out[side] = {
            "saved": "session_vr_to_robot" in data,
            "calibrated_at": data.get("calibrated_at"),
            "forward_motion_m": float(data.get("forward_motion_m", 0.0)),
            "up_motion_m": float(data.get("up_motion_m", 0.0)),
            "left_motion_m": float(data.get("left_motion_m", 0.0)),
            "invert_lateral": data.get("invert_lateral"),
            "confidence": data.get("confidence", "unknown"),
            "wrist_flex_sign": data.get("wrist_flex_sign"),
            "wrist_roll_sign": data.get("wrist_roll_sign"),
        }
    return out


def _orthonormalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return the nearest proper 3D rotation matrix."""
    u, _, vt = np.linalg.svd(matrix)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return rot


def read_invert_lateral_flags() -> dict[str, bool]:
    """Read per-arm `vr.invert_lateral_<side>` flags from config/xlerobot.yaml.
    Returns {'left': bool, 'right': bool}. Missing keys default to False."""
    return {s: bool(_yaml_invert_raw().get(s)) for s in ("left", "right")}


def read_invert_lateral_overrides() -> dict[str, bool]:
    """For each side, is the YAML flag EXPLICITLY set (so it should override
    the calibration wizard's auto-decision)?

    Distinguishes 'key absent / null' (wizard decides) from 'key present with
    bool value' (manual override; wizard skips its decision). The wizard's
    step-3 lateral check catches matrix-math mirroring but NOT physical motor
    mirroring (mirror-mounted arm with reversed sign convention), so users
    need a manual escape hatch."""
    raw = _yaml_invert_raw()
    return {s: (raw.get(s) is not None) for s in ("left", "right")}


def _yaml_invert_raw() -> dict[str, Any]:
    """Internal: return the raw values (or None if absent) for invert flags."""
    import pathlib
    cfg_path = pathlib.Path(__file__).resolve().parents[2] / "config" / "xlerobot.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception as e:
        log.warning("could not read %s for invert flags: %s", cfg_path, e)
        return {"left": None, "right": None}
    vr = cfg.get("vr") or {}
    return {
        "left":  vr.get("invert_lateral_left"),    # None / True / False
        "right": vr.get("invert_lateral_right"),
    }
