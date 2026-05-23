#!/usr/bin/env python3
"""Run pi0.5 VLA inference on the bimanual SO101 XLeRobot.

Thin wrapper around XLerobot_xuweiwu/software/examples/9_pi05_inference_dualarm.py.
The actual policy runs out-of-process via the openpi WebSocket server (start it with
scripts/run_openpi_server.sh in another terminal).

Prerequisites:
    bash scripts/setup_xlerobot.sh             # robot-side venv
    cd openpi && uv sync                       # policy-server venv (one-time)
    bash scripts/run_openpi_server.sh          # start policy server in another shell

Usage:
    uv run python scripts/run_pi05_inference.py \\
        --task "Pick the red block and place it in the bin" \\
        --episodes 2 --episode-time 120

Baseline caveat (per plan): the published pi0.5 bimanual-SO101 config loads the *generic*
pi05_base checkpoint, not a bimanual-SO101 finetune. Expect poor zero-shot behavior until
you finetune on your VR-captured dataset.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import socket
import sys
import types

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT / "XLerobot_xuweiwu" / "software" / "examples" / "9_pi05_inference_dualarm.py"
CONFIG_YAML = REPO_ROOT / "config" / "xlerobot.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--episode-time", type=int, default=120)
    p.add_argument("--server-host", default=None, help="Override openpi server host.")
    p.add_argument("--server-port", type=int, default=None)
    p.add_argument("--strict-motors", action="store_true",
                   help="Require all XLerobot motors to be present (default: lenient — "
                        "absent base/head motors are dropped from the bus registry).")
    return p.parse_args()


def load_yaml() -> dict:
    if not CONFIG_YAML.is_file():
        sys.exit(f"missing {CONFIG_YAML}")
    return yaml.safe_load(CONFIG_YAML.read_text())


def patch_xlerobot_factory(robot_id: str) -> None:
    """Replace lerobot.robots.xlerobot.XLerobotConfig with one whose defaults come from
    config/xlerobot.yaml. See scripts/run_vr_teleop_capture.py for the long explanation."""
    from _xlerobot_loader import make_config
    import lerobot.robots.xlerobot as xr

    original = xr.XLerobotConfig

    def patched(*args, **kwargs):
        if args or any(k for k in kwargs if k != "id" and k != "use_degrees"):
            return original(*args, **kwargs)
        return make_config(robot_id=kwargs.get("id", robot_id))

    xr.XLerobotConfig = patched  # type: ignore[assignment]


def server_alive(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def load_upstream() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("pi05_upstream", UPSTREAM)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    args = parse_args()
    cfg = load_yaml()
    patch_xlerobot_factory(robot_id="xlerobot")
    if not args.strict_motors:
        from _xlerobot_loader import patch_motors_bus_lenient
        patch_motors_bus_lenient()

    host = args.server_host or cfg["pi05"]["server_host"]
    port = args.server_port or cfg["pi05"]["server_port"]

    if not server_alive(host, port):
        sys.exit(
            f"openpi server not reachable at {host}:{port}\n"
            f"start it first: bash scripts/run_openpi_server.sh"
        )

    upstream = load_upstream()
    upstream.TASK_DESCRIPTION = args.task
    upstream.NUM_EPISODES = args.episodes
    upstream.EPISODE_TIME_SEC = args.episode_time
    upstream.OPENPI_SERVER_IP = host
    # The upstream script hardcodes port=8000 when constructing WebsocketClientPolicy;
    # if the user changes the port via CLI, patch the construction by monkey-patching
    # the class to default to our port.
    if port != 8000:
        from openpi_client import websocket_client_policy as _wsc
        orig = _wsc.WebsocketClientPolicy.__init__
        def _init(self, host=host, port=port, *a, **kw):
            return orig(self, host=host, port=port, *a, **kw)
        _wsc.WebsocketClientPolicy.__init__ = _init

    print("=" * 60)
    print(f"  Task             : {upstream.TASK_DESCRIPTION}")
    print(f"  Episodes         : {upstream.NUM_EPISODES} × ≤{upstream.EPISODE_TIME_SEC}s")
    print(f"  Policy server    : {host}:{port}  ✓ reachable")
    print("=" * 60)

    upstream.main()


if __name__ == "__main__":
    main()
