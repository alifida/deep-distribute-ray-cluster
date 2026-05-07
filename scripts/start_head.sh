#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/.venv/bin/activate"

RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
NUM_GPUS="${NUM_GPUS:-1}"

echo "[head] stopping old Ray runtime"
ray stop || true

echo "[head] starting Ray head on port ${RAY_PORT}"
ray start \
  --head \
  --port="${RAY_PORT}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="${DASHBOARD_PORT}" \
  --num-gpus="${NUM_GPUS}"

echo "[head] started"
echo "[head] dashboard: http://127.0.0.1:${DASHBOARD_PORT}"
