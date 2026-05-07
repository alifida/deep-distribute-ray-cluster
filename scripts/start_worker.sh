#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 192.18.99 6379"
  exit 1
fi

HEAD_IP="$1"
HEAD_PORT="${2:-6379}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
if [[ ! -x "${VENV_DIR}/bin/python" ]] || [[ "$("${VENV_DIR}/bin/python" -c 'import sys; print(sys.prefix)' 2>/dev/null || true)" != "${VENV_DIR}" ]]; then
  echo "[worker] .venv missing or stale, running setup_node.sh..."
  bash "${PROJECT_DIR}/scripts/setup_node.sh"
fi
if ! "${VENV_DIR}/bin/python" -c "import ray" >/dev/null 2>&1; then
  echo "[worker] ray not found in venv, installing ray..."
  "${VENV_DIR}/bin/python" -m pip install ray
fi
RAY_CMD=("${VENV_DIR}/bin/python" -m ray.scripts.scripts)

NUM_GPUS="${NUM_GPUS:-1}"

echo "[worker] stopping old Ray runtime"
"${RAY_CMD[@]}" stop || true

echo "[worker] connecting to ${HEAD_IP}:${HEAD_PORT}"
"${RAY_CMD[@]}" start \
  --address="${HEAD_IP}:${HEAD_PORT}" \
  --num-gpus="${NUM_GPUS}"

echo "[worker] started and joined cluster"
