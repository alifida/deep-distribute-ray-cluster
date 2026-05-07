#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"
if [[ ! -x "${VENV_DIR}/bin/python" ]] || [[ "$("${VENV_DIR}/bin/python" -c 'import sys; print(sys.prefix)' 2>/dev/null || true)" != "${VENV_DIR}" ]]; then
  echo "[ui] .venv missing or stale, running setup_node.sh..."
  bash "${PROJECT_DIR}/scripts/setup_node.sh"
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

cd "${PROJECT_DIR}"
echo "[ui] starting FastAPI UI on http://${HOST}:${PORT}"
"${VENV_DIR}/bin/python" -m uvicorn api:app --host "${HOST}" --port "${PORT}"
