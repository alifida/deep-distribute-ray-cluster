#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

echo "[setup] project: ${PROJECT_DIR}"
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  CURRENT_PREFIX="$("${VENV_DIR}/bin/python" -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
  if [[ "${CURRENT_PREFIX}" != "${VENV_DIR}" ]]; then
    echo "[setup] existing .venv points to old path, recreating..."
    rm -rf "${VENV_DIR}"
  fi
fi

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo "[setup] done"
echo "[setup] activate with: source ${VENV_DIR}/bin/activate"
