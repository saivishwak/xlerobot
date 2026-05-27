import asyncio
import time
import threading
from types import SimpleNamespace

from flask import Flask
from webapp.backend import api as api_mod
from webapp.backend import vr_teleop as vr_mod


class _FakeRecorder:
    repo_id = "test/repo"
    episode_count = 0
    frame_count_in_episode = 0
    in_episode = True

    def __init__(self):
        self.frames = []

    def start_episode(self, task=""):
        self.task = task

    def add_frame(self, action, present, camera_frames):
        self.frames.append((action, present, camera_frames))

    def end_episode(self):
        self.episode_count += 1
        return True

    def finalize(self):
        self.finalized = True


class _FakeMotors:
    connected_sides = ["left", "right"]

    def __init__(self, present):
        self._present = present

    def read_positions(self):
        return dict(self._present)


def _joint_values(side, base):
    prefix = f"{side}_arm_"
    return {
        f"{prefix}shoulder_pan": base + 0,
        f"{prefix}shoulder_lift": base + 1,
        f"{prefix}elbow_flex": base + 2,
        f"{prefix}wrist_flex": base + 3,
        f"{prefix}wrist_roll": base + 4,
        f"{prefix}gripper": base + 5,
    }


def test_record_frame_uses_same_tick_commanded_actions(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    left_command = _joint_values("left", 100)
    right_command = _joint_values("right", 200)

    session._recording = True
    session._recorder = recorder
    session._engaged = True
    monkeypatch.setattr(vr_mod, "MOTORS", _FakeMotors(present))
    monkeypatch.setattr(vr_mod._dataset, "grab_camera_frames", lambda: {})

    session._record_frame_if_active(
        commanded_this_tick={"left": left_command, "right": right_command}
    )

    action, observed, _ = recorder.frames[0]
    assert observed == present
    assert action == {**left_command, **right_command}


def test_record_frame_falls_back_to_present_for_passive_arm(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    present = {**_joint_values("left", 10), **_joint_values("right", 20)}
    right_command = _joint_values("right", 200)

    session._recording = True
    session._recorder = recorder
    monkeypatch.setattr(vr_mod, "MOTORS", _FakeMotors(present))
    monkeypatch.setattr(vr_mod._dataset, "grab_camera_frames", lambda: {})

    session._record_frame_if_active(commanded_this_tick={"right": right_command})

    action, _, _ = recorder.frames[0]
    assert {k: action[k] for k in _joint_values("left", 0)} == _joint_values("left", 10)
    assert {k: action[k] for k in _joint_values("right", 0)} == right_command


def test_set_recording_rejects_empty_task(monkeypatch):
    session = vr_mod.VRTeleopSession()
    monkeypatch.setattr(vr_mod._dataset, "load_dataset_config", lambda: {"home_before_episode": False})

    assert session.set_recording(True, task="   ") is False
    assert session._recording is False
    assert "task description required" in (session._last_error or "")


def test_set_recording_task_cache_can_be_cleared():
    session = vr_mod.VRTeleopSession()

    session.set_recording_task("  Pick the red block  ")
    assert session._last_task == "Pick the red block"

    session.set_recording_task("   ")
    assert session._last_task == ""


def test_recording_api_rejects_empty_task():
    app = Flask(__name__)
    app.register_blueprint(api_mod.bp)

    resp = app.test_client().post(
        "/api/vr/recording",
        json={"enabled": True, "task": "   "},
    )

    assert resp.status_code == 400


def test_recording_task_api_caches_prompt(monkeypatch):
    session = vr_mod.VRTeleopSession()
    monkeypatch.setattr(api_mod.vr_mod, "SESSION", session)
    app = Flask(__name__)
    app.register_blueprint(api_mod.bp)

    resp = app.test_client().post(
        "/api/vr/recording/task",
        json={"task": "  Pick the red block  "},
    )

    assert resp.status_code == 200
    assert session._last_task == "Pick the red block"


def test_recording_restart_waits_for_previous_stop_to_finish(monkeypatch):
    session = vr_mod.VRTeleopSession()
    old_rec = _FakeRecorder()
    new_rec = _FakeRecorder()
    new_rec.episode_count = 1
    entered_end = threading.Event()
    allow_end = threading.Event()
    start_done = threading.Event()

    def slow_end_episode():
        entered_end.set()
        assert allow_end.wait(timeout=2.0)
        old_rec.episode_count += 1
        return True

    old_rec.end_episode = slow_end_episode
    session._recording = True
    session._recorder = old_rec
    session._last_task = "Pick the red block"

    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {
            "home_before_episode": False,
            "repo_id": "test/repo",
            "fps": 30,
            "push_to_hub": False,
            "root": None,
        },
    )
    monkeypatch.setattr(vr_mod._dataset, "role_camera_list", lambda: (["head"], (2, 2, 3)))
    monkeypatch.setattr(vr_mod._dataset, "resolve_root", lambda root, repo_id: "/tmp/test/repo")
    monkeypatch.setattr(vr_mod._dataset, "DatasetRecorder", lambda **kwargs: new_rec)

    stop_thread = threading.Thread(target=lambda: session.set_recording(False))
    stop_thread.start()
    assert entered_end.wait(timeout=2.0)

    start_thread = threading.Thread(
        target=lambda: (session.set_recording(True, task="Pick the red block"), start_done.set())
    )
    start_thread.start()
    time.sleep(0.05)
    assert not start_done.is_set()

    allow_end.set()
    stop_thread.join(timeout=2.0)
    start_thread.join(timeout=2.0)

    assert start_done.is_set()
    assert getattr(old_rec, "finalized", False) is True
    assert session._recording is True
    assert session._recorder is new_rec
    assert new_rec.task == "Pick the red block"
    assert session._episodes_saved == 1


def test_b_button_start_requires_synced_task(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()

    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {
            "home_before_episode": False,
            "repo_id": "test/repo",
            "fps": 30,
            "push_to_hub": False,
            "root": None,
        },
    )
    monkeypatch.setattr(vr_mod._dataset, "role_camera_list", lambda: (["head"], (2, 2, 3)))
    monkeypatch.setattr(vr_mod._dataset, "resolve_root", lambda root, repo_id: "/tmp/test/repo")
    monkeypatch.setattr(vr_mod._dataset, "DatasetRecorder", lambda **kwargs: recorder)

    session._handle_record_button("right")

    assert session._recording is False
    assert "task description required" in (session._last_error or "")

    session.set_recording_task("Pick the red block")
    session._handle_record_button("right")

    assert session._recording is True
    assert recorder.task == "Pick the red block"


def test_vr_button_release_is_forwarded_for_repeat_b_toggles():
    from xlevr.inputs.vr_ws_server import VRWebSocketServer

    async def run():
        queue = asyncio.Queue()
        server = VRWebSocketServer(command_queue=queue, config=SimpleNamespace())

        await server.process_single_controller(
            "right",
            {"gripActive": False, "buttons": {"B": True}},
        )
        await server.process_single_controller(
            "right",
            {"gripActive": False, "buttons": {"B": False}},
        )
        await server.process_single_controller(
            "right",
            {"gripActive": False, "buttons": {"B": True}},
        )

        goals = [await asyncio.wait_for(queue.get(), timeout=0.5) for _ in range(3)]
        return [goal.buttons for goal in goals]

    assert asyncio.run(run()) == [{"B": True}, {"B": False}, {"B": True}]


def test_delete_last_recorded_episode_requires_stop():
    session = vr_mod.VRTeleopSession()
    session._recording = True

    out = session.delete_last_recorded_episode()

    assert out["recording"] is True
    assert "stop recording" in (out["last_error"] or "")


def test_delete_last_recorded_episode_updates_counters(monkeypatch):
    session = vr_mod.VRTeleopSession()
    session._episodes_saved = 3
    session._last_dataset_root = "/tmp/old-root"

    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {"repo_id": "test/repo", "root": None},
    )
    monkeypatch.setattr(
        vr_mod._dataset,
        "delete_last_episode",
        lambda repo_id, root: (2, "/tmp/new-root"),
    )
    monkeypatch.setattr(
        vr_mod._dataset,
        "last_episode_summary",
        lambda repo_id, root: (1, 420),
    )

    out = session.delete_last_recorded_episode()

    assert out["recording_info"]["episodes_saved"] == 2
    assert out["recording_info"]["root"] == "/tmp/new-root"
    assert out["recording_info"]["last_episode_index"] == 1
    assert out["recording_info"]["last_episode_frames"] == 420
    assert session._episodes_saved == 2
    assert session._last_dataset_root == "/tmp/new-root"


def test_delete_last_recorded_episode_api(monkeypatch):
    session = vr_mod.VRTeleopSession()
    monkeypatch.setattr(api_mod.vr_mod, "SESSION", session)
    monkeypatch.setattr(
        session,
        "delete_last_recorded_episode",
        lambda: {"ok": True, "recording": False},
    )

    app = Flask(__name__)
    app.register_blueprint(api_mod.bp)
    resp = app.test_client().post("/api/vr/recording/delete_last", json={})

    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True, "recording": False}


def test_start_recording_after_delete_keeps_saved_count(monkeypatch):
    session = vr_mod.VRTeleopSession()
    recorder = _FakeRecorder()
    recorder.episode_count = 2
    recorder.last_saved_episode_index = 1
    recorder.last_saved_episode_frames = 420

    session._episodes_saved = 2
    session._recording = False
    session._last_task = "Pick the red block"

    monkeypatch.setattr(
        vr_mod._dataset,
        "load_dataset_config",
        lambda: {
            "home_before_episode": False,
            "repo_id": "test/repo",
            "fps": 30,
            "push_to_hub": False,
            "root": None,
        },
    )
    monkeypatch.setattr(vr_mod._dataset, "role_camera_list", lambda: (["head"], (2, 2, 3)))
    monkeypatch.setattr(vr_mod._dataset, "resolve_root", lambda root, repo_id: "/tmp/test/repo")
    monkeypatch.setattr(vr_mod._dataset, "DatasetRecorder", lambda **kwargs: recorder)

    assert session.set_recording(True, task="Pick the red block") is True
    status = session.status()
    assert status["recording_info"]["episodes_saved"] == 2
    assert status["recording_info"]["last_episode_index"] == 1
    assert status["recording_info"]["last_episode_frames"] == 420



