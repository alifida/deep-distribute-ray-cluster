#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${PROJECT_DIR}/scripts/cluster_config.env"
if [[ -f "${CONFIG_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
  set +a
fi

VENV_DIR="${PROJECT_DIR}/.venv"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[stop-worker] .venv not found. Nothing to stop."
  exit 0
fi

RAY_CMD=("${VENV_DIR}/bin/python" -m ray.scripts.scripts)
echo "[stop-worker] stopping Ray runtime"
"${RAY_CMD[@]}" stop || true
echo "[stop-worker] done"
