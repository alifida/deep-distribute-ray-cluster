#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
if [[ ! -x "${VENV_DIR}/bin/python" ]] || [[ "$("${VENV_DIR}/bin/python" -c 'import sys; print(sys.prefix)' 2>/dev/null || true)" != "${VENV_DIR}" ]]; then
  echo "[head] .venv missing or stale, running setup_node.sh..."
  bash "${PROJECT_DIR}/scripts/setup_node.sh"
fi
if ! "${VENV_DIR}/bin/python" -c "import ray" >/dev/null 2>&1; then
  echo "[head] ray not found in venv, installing ray..."
  "${VENV_DIR}/bin/python" -m pip install ray
fi
RAY_CMD=("${VENV_DIR}/bin/python" -m ray.scripts.scripts)

RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
NUM_GPUS="${NUM_GPUS:-1}"
RESTART_RAY="${RESTART_RAY:-0}"

if [[ "${RESTART_RAY}" == "1" ]]; then
  echo "[head] RESTART_RAY=1 -> stopping old Ray runtime"
  "${RAY_CMD[@]}" stop || true
else
  if "${RAY_CMD[@]}" status >/dev/null 2>&1; then
    echo "[head] Ray cluster already running. Skipping restart."
    echo "[head] Use RESTART_RAY=1 bash scripts/start_head.sh to force restart."
    echo "[head] dashboard: http://127.0.0.1:${DASHBOARD_PORT}"
    exit 0
  fi
fi

echo "[head] starting Ray head on port ${RAY_PORT}"
"${RAY_CMD[@]}" start \
  --head \
  --port="${RAY_PORT}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${DASHBOARD_PORT}" \
  --num-gpus="${NUM_GPUS}"

echo "[head] started"
echo "[head] dashboard: http://127.0.0.1:${DASHBOARD_PORT}"
