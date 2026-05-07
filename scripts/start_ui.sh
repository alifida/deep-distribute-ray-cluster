#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${PROJECT_DIR}/.venv/bin/activate"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"

cd "${PROJECT_DIR}"
echo "[ui] starting FastAPI UI on http://${HOST}:${PORT}"
uvicorn api:app --host "${HOST}" --port "${PORT}"
