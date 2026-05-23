#!/usr/bin/env bash
# Start the openpi pi0.5 policy WebSocket server.
#
# openpi has heavy/conflicting deps (torch==2.7.1, jax[cuda12]==0.5.3) and is installed
# in its own isolated venv via `cd openpi && uv sync`. This script just dispatches there.
#
# Baseline caveat: the `pi05_bimanual_so101_lora` config in openpi loads the generic
# pi05_base checkpoint as starting weights — NOT a bimanual-SO101 finetune. Until you
# finetune on the VR-captured dataset, zero-shot behavior on a bimanual SO101 will be poor.
#
# Override the checkpoint dir via $OPENPI_CHECKPOINT_DIR (defaults to the pi05_base mirror
# referenced in openpi/src/openpi/training/config.py).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/openpi"

CONFIG="${OPENPI_CONFIG:-pi05_bimanual_so101_lora}"
CKPT_DIR="${OPENPI_CHECKPOINT_DIR:-gs://openpi-assets/checkpoints/pi05_base}"
PORT="${OPENPI_PORT:-8000}"

echo "[openpi-server] config=$CONFIG"
echo "[openpi-server] ckpt  =$CKPT_DIR"
echo "[openpi-server] port  =$PORT"

# Ensure openpi's venv exists (idempotent).
if [[ ! -d ".venv" ]]; then
    echo "[openpi-server] first-time setup: uv sync (this can take a while)"
    uv sync
fi

exec uv run scripts/serve_policy.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config="$CONFIG" \
    --policy.dir="$CKPT_DIR"
