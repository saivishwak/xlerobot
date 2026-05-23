"""Flask API blueprint — JSON endpoints + camera MJPEG streams + /api/vr/*."""
from __future__ import annotations

from dataclasses import asdict

from flask import Blueprint, Response, abort, jsonify, request, stream_with_context

from . import cameras as cam_mod
from . import doctor as doc_mod
from . import vr_teleop as vr_mod

bp = Blueprint("api", __name__)


# ───── Doctor ─────────────────────────────────────────────────────────────
@bp.get("/api/doctor")
def api_doctor():
    return jsonify(checks=doc_mod.run_doctor())


# ───── Cameras ────────────────────────────────────────────────────────────
@bp.get("/api/cameras")
def api_cameras():
    cams = [asdict(c) for c in cam_mod.enumerate_cameras()]
    return jsonify(cameras=cams, roles=list(cam_mod._ROLES))


@bp.post("/api/cameras/assign")
def api_assign():
    body = request.get_json(silent=True) or {}
    by_path = body.get("by_path")
    role = body.get("role")
    if not by_path:
        abort(400, "by_path required")
    try:
        cam_mod.assign_role(by_path, role)
    except ValueError as e:
        abort(400, str(e))
    cam_mod.reset_streams()
    cams = [asdict(c) for c in cam_mod.enumerate_cameras()]
    return jsonify(ok=True, cameras=cams)


@bp.get("/camera/<cam_id>/stream")
def camera_stream(cam_id: str):
    if cam_mod.find_camera(cam_id) is None:
        abort(404, f"unknown camera: {cam_id}")
    return Response(
        stream_with_context(cam_mod.mjpeg_iter(cam_id)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@bp.get("/camera/<cam_id>/snapshot")
def camera_snapshot(cam_id: str):
    if cam_mod.find_camera(cam_id) is None:
        abort(404, f"unknown camera: {cam_id}")
    stream = cam_mod.get_stream(cam_id)
    assert stream is not None
    jpeg, err = stream.snapshot()
    if jpeg is None:
        abort(503, f"camera unavailable: {err or 'no frame'}")
    return Response(jpeg, mimetype="image/jpeg")


# ───── VR Teleop ──────────────────────────────────────────────────────────
@bp.get("/api/vr/status")
def api_vr_status():
    return jsonify(vr_mod.SESSION.status())


@bp.post("/api/vr/connect")
def api_vr_connect():
    body = request.get_json(silent=True) or {}
    arm = body.get("arm")
    if arm not in ("left", "right"):
        abort(400, "arm must be 'left' or 'right'")
    try:
        return jsonify(vr_mod.SESSION.connect(arm))
    except RuntimeError as e:
        return jsonify(error=str(e), **vr_mod.SESSION.status()), 409
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}", **vr_mod.SESSION.status()), 500


@bp.post("/api/vr/disconnect")
def api_vr_disconnect():
    """Disconnect a specific arm via `{arm: 'left'|'right'}`, or omit `arm` to
    disconnect both. Always returns the post-disconnect status."""
    body = request.get_json(silent=True) or {}
    arm = body.get("arm")
    if arm is not None and arm not in ("left", "right"):
        abort(400, "arm must be 'left', 'right', or omitted")
    return jsonify(vr_mod.SESSION.disconnect(side=arm))


@bp.post("/api/vr/engage")
def api_vr_engage():
    """Body: `{engaged: bool, scale?: 0.1..1.0, active_arm?: 'left'|'right'}`.

    - `engaged=true` requires at least one arm connected. If both arms are
      connected, `active_arm` MUST be supplied to choose which one VR drives.
    - `engaged=false` releases the active-arm gate (both arms stay torqued).
    """
    body = request.get_json(silent=True) or {}
    if "engaged" not in body:
        abort(400, "engaged required")
    engaged = bool(body["engaged"])
    scale = body.get("scale")
    if scale is not None:
        try: scale = float(scale)
        except (TypeError, ValueError): abort(400, "scale must be numeric")
    active_arm = body.get("active_arm")
    if active_arm is not None and active_arm not in ("left", "right"):
        abort(400, "active_arm must be 'left' or 'right'")
    try:
        return jsonify(vr_mod.SESSION.engage(
            engaged=engaged, scale=scale, active_arm=active_arm,
        ))
    except (RuntimeError, ValueError) as e:
        return jsonify(error=str(e), **vr_mod.SESSION.status()), 409


@bp.post("/api/vr/emergency_stop")
def api_vr_emergency_stop():
    return jsonify(vr_mod.SESSION.emergency_stop())


@bp.post("/api/vr/torque/release")
def api_vr_torque_release():
    """Body: `{arm}`. Disable torque on one arm so the user can hand-pose it.
    The arm goes limp and will sag under gravity. Drive loop will skip this arm
    until `/api/vr/torque/lock` is called."""
    body = request.get_json(silent=True) or {}
    arm = body.get("arm")
    if arm not in ("left", "right"):
        abort(400, "arm must be 'left' or 'right'")
    try:
        return jsonify(vr_mod.SESSION.release_torque_for_posing(arm))
    except RuntimeError as e:
        return jsonify(error=str(e), **vr_mod.SESSION.status()), 409


@bp.post("/api/vr/torque/lock")
def api_vr_torque_lock():
    """Body: `{arm}`. Re-enable torque at the arm's CURRENT position (no
    snap-back to a stale goal). Used after hand-posing."""
    body = request.get_json(silent=True) or {}
    arm = body.get("arm")
    if arm not in ("left", "right"):
        abort(400, "arm must be 'left' or 'right'")
    try:
        return jsonify(vr_mod.SESSION.lock_torque(arm))
    except RuntimeError as e:
        return jsonify(error=str(e), **vr_mod.SESSION.status()), 409


@bp.post("/api/vr/home/capture")
def api_vr_home_capture():
    """Body: `{arm?: "left"|"right"}`. With `arm` omitted, captures the present
    pose of every connected arm. Writes to config/xlerobot.yaml's
    `robot.home_pose:` block, preserving all other content + comments."""
    body = request.get_json(silent=True) or {}
    arm = body.get("arm")
    if arm is not None and arm not in ("left", "right"):
        abort(400, "arm must be 'left', 'right', or omitted")
    try:
        return jsonify(vr_mod.SESSION.capture_home(side=arm))
    except RuntimeError as e:
        return jsonify(error=str(e), **vr_mod.SESSION.status()), 409
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}",
                       **vr_mod.SESSION.status()), 500


@bp.post("/api/vr/home/go")
def api_vr_home_go():
    """Body: `{arm?: "left"|"right"}`. Begins a slow, per-tick-clamped
    interpolation from the current pose to the saved home pose. Returns when
    homing has been QUEUED (not when it's finished). Status's
    `arms.<side>.home.homing` flag indicates in-flight motion."""
    body = request.get_json(silent=True) or {}
    arm = body.get("arm")
    if arm is not None and arm not in ("left", "right"):
        abort(400, "arm must be 'left', 'right', or omitted")
    try:
        return jsonify(vr_mod.SESSION.go_home(side=arm))
    except RuntimeError as e:
        return jsonify(error=str(e), **vr_mod.SESSION.status()), 409


@bp.post("/api/vr/home/cancel")
def api_vr_home_cancel():
    body = request.get_json(silent=True) or {}
    arm = body.get("arm")
    if arm is not None and arm not in ("left", "right"):
        abort(400, "arm must be 'left', 'right', or omitted")
    return jsonify(vr_mod.SESSION.cancel_homing(side=arm))


@bp.post("/api/vr/calibrate/start")
def api_vr_calibrate_start():
    """Begin a motion-based guided calibration for one arm. Body: `{arm}`.

    After calling: user squeezes grip on that controller (anchor), moves their
    hand FORWARD ~10 cm in the direction they consider "forward" relative to
    their body, then releases grip. The backend captures the motion vector
    and computes the per-session VR→robot rotation matrix from it.
    """
    body = request.get_json(silent=True) or {}
    arm = body.get("arm")
    if arm not in ("left", "right"):
        abort(400, "arm must be 'left' or 'right'")
    try:
        return jsonify(vr_mod.SESSION.start_calibration(arm))
    except (RuntimeError, ValueError) as e:
        return jsonify(error=str(e), **vr_mod.SESSION.status()), 409


@bp.post("/api/vr/calibrate/cancel")
def api_vr_calibrate_cancel():
    body = request.get_json(silent=True) or {}
    arm = body.get("arm")
    if arm not in ("left", "right"):
        abort(400, "arm must be 'left' or 'right'")
    return jsonify(vr_mod.SESSION.cancel_calibration(arm))


@bp.post("/api/vr/recording")
def api_vr_recording():
    """Body: `{enabled: bool, task?: str}`. Mirror of the B button on the right
    Quest controller. On `enabled=true`, lazily constructs the LeRobotDataset
    writer and opens a new episode. On `enabled=false`, saves the current episode
    to disk and returns."""
    body = request.get_json(silent=True) or {}
    if "enabled" not in body:
        abort(400, "enabled required")
    task = body.get("task") or ""
    vr_mod.SESSION.set_recording(bool(body["enabled"]), task=task)
    return jsonify(vr_mod.SESSION.status())
