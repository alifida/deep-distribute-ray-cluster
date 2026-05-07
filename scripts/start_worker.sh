#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <head_ip> [head_port]"
  exit 1
fi

HEAD_IP="$1"
HEAD_PORT="${2:-6379}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "[worker] .venv not found, running setup_node.sh..."
  bash "${PROJECT_DIR}/scripts/setup_node.sh"
fi
source "${VENV_DIR}/bin/activate"
if ! command -v ray >/dev/null 2>&1; then
  echo "[worker] ray not found in venv, installing ray..."
  pip install ray
fi

NUM_GPUS="${NUM_GPUS:-1}"

echo "[worker] stopping old Ray runtime"
ray stop || true

echo "[worker] connecting to ${HEAD_IP}:${HEAD_PORT}"
ray start \
  --address="${HEAD_IP}:${HEAD_PORT}" \
  --num-gpus="${NUM_GPUS}"

echo "[worker] started and joined cluster"
