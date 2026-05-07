#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

echo "[setup] project: ${PROJECT_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt"

echo "[setup] done"
echo "[setup] activate with: source ${VENV_DIR}/bin/activate"
