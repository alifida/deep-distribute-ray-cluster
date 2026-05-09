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
NODE_MANAGER_PORT="${NODE_MANAGER_PORT:-10001}"
OBJECT_MANAGER_PORT="${OBJECT_MANAGER_PORT:-10002}"
RAY_CLIENT_SERVER_PORT="${RAY_CLIENT_SERVER_PORT:-10003}"
DASHBOARD_AGENT_LISTEN_PORT="${DASHBOARD_AGENT_LISTEN_PORT:-10004}"
METRICS_EXPORT_PORT="${METRICS_EXPORT_PORT:-10005}"
MIN_WORKER_PORT="${MIN_WORKER_PORT:-11000}"
MAX_WORKER_PORT="${MAX_WORKER_PORT:-11999}"

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
  --node-manager-port="${NODE_MANAGER_PORT}" \
  --object-manager-port="${OBJECT_MANAGER_PORT}" \
  --ray-client-server-port="${RAY_CLIENT_SERVER_PORT}" \
  --dashboard-agent-listen-port="${DASHBOARD_AGENT_LISTEN_PORT}" \
  --metrics-export-port="${METRICS_EXPORT_PORT}" \
  --min-worker-port="${MIN_WORKER_PORT}" \
  --max-worker-port="${MAX_WORKER_PORT}" \
  --num-gpus="${NUM_GPUS}"

echo "[head] started"
echo "[head] dashboard: http://127.0.0.1:${DASHBOARD_PORT}"
echo "[head] fixed ports:"
echo "  gcs: ${RAY_PORT}"
echo "  node_manager: ${NODE_MANAGER_PORT}"
echo "  object_manager: ${OBJECT_MANAGER_PORT}"
echo "  ray_client: ${RAY_CLIENT_SERVER_PORT}"
echo "  dashboard_agent: ${DASHBOARD_AGENT_LISTEN_PORT}"
echo "  metrics_export: ${METRICS_EXPORT_PORT}"
echo "  workers: ${MIN_WORKER_PORT}-${MAX_WORKER_PORT}"
