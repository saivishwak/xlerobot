#!/usr/bin/env bash
# Idempotent bootstrap for VR teleop + LeRobot dataset capture + pi0.5 baseline.
#
# What this does:
#   1. Copy XLerobot_xuweiwu's xlerobot robot/model/util files into the lerobot submodule tree
#      so `from lerobot.robots.xlerobot import ...` resolves (per XLeRobot docs).
#   2. Mark those copied paths as locally-excluded in the lerobot submodule.
#   3. Install lerobot[feetech,intelrealsense] into the project venv via `uv add`.
#   4. Install XLeVR's runtime deps (websockets, pynput, etc.) into the project venv.
#   5. Install openpi-client (lightweight WebSocket client) into the project venv.
#   6. Drop a .pth in the project venv site-packages so `XLeVR.*` resolves on sys.path.
#   7. Generate XLeVR self-signed HTTPS cert + key if missing (Quest 3 browser requires HTTPS).
#
# Not done here (run separately):
#   - Full openpi install. openpi pins torch==2.7.1 and jax==0.5.3 which conflict with the
#     robot-side environment. Run `cd openpi && uv sync` to set up its own isolated venv.
#     See scripts/run_openpi_server.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
ok()  { printf '\033[1;32m[ ok ]\033[0m %s\n' "$*"; }
warn(){ printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }

# ──────────────────────────────────────────────────────────────────────────────
# 1. Copy xlerobot files into lerobot tree (docs-recommended)
# ──────────────────────────────────────────────────────────────────────────────
log "Copying XLerobot files into lerobot submodule..."

declare -a COPIES=(
    "XLerobot_xuweiwu/software/src/robots/xlerobot:lerobot/src/lerobot/robots/xlerobot"
    "XLerobot_xuweiwu/software/src/robots/xlerobot_2wheels:lerobot/src/lerobot/robots/xlerobot_2wheels"
    "XLerobot_xuweiwu/software/src/teleporators/xlerobot_vr:lerobot/src/lerobot/teleoperators/xlerobot_vr"
    "XLerobot_xuweiwu/software/src/model/SO101Robot.py:lerobot/src/lerobot/model/SO101Robot.py"
    "XLerobot_xuweiwu/software/src/utils/quadratic_spline_via_ipol.py:lerobot/src/lerobot/utils/quadratic_spline_via_ipol.py"
)

for entry in "${COPIES[@]}"; do
    src="${entry%%:*}"
    dst="${entry##*:}"
    if [[ ! -e "$src" ]]; then
        warn "source missing, skipping: $src"
        continue
    fi
    if [[ -e "$dst" ]]; then
        ok "exists, skipping: $dst"
    else
        cp -a "$src" "$dst"
        ok "copied: $src -> $dst"
    fi
done

# ──────────────────────────────────────────────────────────────────────────────
# 2. Mark copied paths as locally-excluded inside the lerobot submodule
# ──────────────────────────────────────────────────────────────────────────────
log "Updating lerobot exclude (local-only)..."
# Submodules redirect `.git` to ../.git/modules/<name>, so resolve via git itself.
LR_EXCLUDE="$(git -C lerobot rev-parse --git-path info/exclude 2>/dev/null || true)"
case "$LR_EXCLUDE" in
    /*) ;;                                  # already absolute
    *)  [[ -n "$LR_EXCLUDE" ]] && LR_EXCLUDE="lerobot/$LR_EXCLUDE" ;;
esac
if [[ -n "$LR_EXCLUDE" && -f "$LR_EXCLUDE" ]]; then
    declare -a EXCLUDES=(
        "src/lerobot/robots/xlerobot/"
        "src/lerobot/robots/xlerobot_2wheels/"
        "src/lerobot/teleoperators/xlerobot_vr/"
        "src/lerobot/model/SO101Robot.py"
        "src/lerobot/utils/quadratic_spline_via_ipol.py"
        "*.egg-info/"
        ".venv/"
    )
    for line in "${EXCLUDES[@]}"; do
        if ! grep -qxF "$line" "$LR_EXCLUDE"; then
            echo "$line" >> "$LR_EXCLUDE"
        fi
    done
    ok "$LR_EXCLUDE updated"
else
    warn "lerobot submodule info/exclude not found — submodule may not be initialized"
fi

# ──────────────────────────────────────────────────────────────────────────────
# 3. Install lerobot (editable) + extras
# ──────────────────────────────────────────────────────────────────────────────
log "Installing lerobot[feetech,intelrealsense] (editable)..."
uv add --editable "./lerobot[feetech,intelrealsense,dataset]"
ok "lerobot installed"

# ──────────────────────────────────────────────────────────────────────────────
# 4. Install XLeVR's runtime deps
# ──────────────────────────────────────────────────────────────────────────────
log "Installing XLeVR runtime deps (websockets, pynput, ...)..."
uv add websockets pynput pyyaml aiohttp
ok "XLeVR deps installed"

# ──────────────────────────────────────────────────────────────────────────────
# 5. Install openpi-client (lightweight WS client; NOT full openpi)
# ──────────────────────────────────────────────────────────────────────────────
log "Installing openpi-client (editable)..."
uv add --editable "./openpi/packages/openpi-client"
ok "openpi-client installed"

# ──────────────────────────────────────────────────────────────────────────────
# 6. Drop a .pth so `XLeVR.*` resolves on sys.path
# ──────────────────────────────────────────────────────────────────────────────
log "Wiring XLeVR onto sys.path via .pth..."
SITE_PKGS="$(uv run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PTH_FILE="$SITE_PKGS/xlerobot_paths.pth"
PTH_LINE="$REPO_ROOT/XLeRobot"
if [[ -f "$PTH_FILE" ]] && grep -qxF "$PTH_LINE" "$PTH_FILE"; then
    ok ".pth already configured: $PTH_FILE"
else
    echo "$PTH_LINE" > "$PTH_FILE"
    ok "wrote $PTH_FILE -> $PTH_LINE"
fi

# ──────────────────────────────────────────────────────────────────────────────
# 7. Generate XLeVR self-signed HTTPS cert if missing
# ──────────────────────────────────────────────────────────────────────────────
log "Ensuring XLeVR HTTPS cert..."
CERT="XLeRobot/XLeVR/cert.pem"
KEY="XLeRobot/XLeVR/key.pem"
if [[ -f "$CERT" && -f "$KEY" ]]; then
    ok "cert + key already present"
else
    HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
    HOST_IP="${HOST_IP:-127.0.0.1}"
    openssl req -x509 -newkey rsa:2048 -nodes -keyout "$KEY" -out "$CERT" \
        -days 365 -subj "/CN=$HOST_IP" \
        -addext "subjectAltName=IP:$HOST_IP,IP:127.0.0.1,DNS:localhost" >/dev/null 2>&1
    ok "generated cert for $HOST_IP"
fi

# ──────────────────────────────────────────────────────────────────────────────
# Done — smoke imports
# ──────────────────────────────────────────────────────────────────────────────
log "Running smoke imports..."
uv run python - <<'PY'
def try_import(label, stmt):
    try:
        exec(stmt, {})
        print(f"[ok]   {label}")
    except Exception as e:
        print(f"[FAIL] {label}: {e}")

try_import("lerobot.robots.xlerobot",
           "from lerobot.robots.xlerobot import XLerobotConfig, XLerobot")
try_import("XLeVR.vr_monitor",
           "from XLeVR.vr_monitor import VRMonitor")
try_import("openpi_client.websocket_client_policy",
           "from openpi_client import websocket_client_policy")
PY

ok "setup complete"
log "Next:"
log "  • Edit config/xlerobot.yaml with your USB paths, RealSense serials, and motor port names."
log "  • For pi0.5 server: cd openpi && uv sync   (sets up its own isolated venv)"
log "  • Then: make teleop  /  make pi05-server  /  make pi05-infer"
