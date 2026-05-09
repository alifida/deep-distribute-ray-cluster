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

HEAD_IP="${1:-${HEAD_IP:-}}"
HEAD_PORT="${2:-${HEAD_PORT:-6379}}"
if [[ -z "${HEAD_IP}" ]]; then
  echo "Usage: $0 <HEAD_IP> [HEAD_PORT]"
  echo "Or set HEAD_IP in scripts/cluster_config.env"
  exit 1
fi
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

# Ensure Ray worker processes can import project modules (ray_ps_async).
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

NUM_GPUS="${NUM_GPUS:-1}"
NODE_MANAGER_PORT="${WORKER_NODE_MANAGER_PORT:-${NODE_MANAGER_PORT:-10011}}"
OBJECT_MANAGER_PORT="${WORKER_OBJECT_MANAGER_PORT:-${OBJECT_MANAGER_PORT:-10012}}"
DASHBOARD_AGENT_LISTEN_PORT="${WORKER_DASHBOARD_AGENT_LISTEN_PORT:-${DASHBOARD_AGENT_LISTEN_PORT:-10014}}"
METRICS_EXPORT_PORT="${WORKER_METRICS_EXPORT_PORT:-${METRICS_EXPORT_PORT:-10015}}"
MIN_WORKER_PORT="${WORKER_MIN_WORKER_PORT:-${MIN_WORKER_PORT:-12000}}"
MAX_WORKER_PORT="${WORKER_MAX_WORKER_PORT:-${MAX_WORKER_PORT:-12999}}"
RAY_TMPDIR="${RAY_TMPDIR:-${PROJECT_DIR}/.ray_tmp}"

mkdir -p "${RAY_TMPDIR}"
export RAY_TMPDIR

echo "[worker] stopping old Ray runtime"
"${RAY_CMD[@]}" stop || true

echo "[worker] connecting to ${HEAD_IP}:${HEAD_PORT}"
echo "[worker] ray temp dir: ${RAY_TMPDIR}"
"${RAY_CMD[@]}" start \
  --address="${HEAD_IP}:${HEAD_PORT}" \
  --node-manager-port="${NODE_MANAGER_PORT}" \
  --object-manager-port="${OBJECT_MANAGER_PORT}" \
  --dashboard-agent-listen-port="${DASHBOARD_AGENT_LISTEN_PORT}" \
  --metrics-export-port="${METRICS_EXPORT_PORT}" \
  --min-worker-port="${MIN_WORKER_PORT}" \
  --max-worker-port="${MAX_WORKER_PORT}" \
  --num-gpus="${NUM_GPUS}"

echo "[worker] started and joined cluster"
echo "[worker] fixed ports:"
echo "  node_manager: ${NODE_MANAGER_PORT}"
echo "  object_manager: ${OBJECT_MANAGER_PORT}"
echo "  dashboard_agent: ${DASHBOARD_AGENT_LISTEN_PORT}"
echo "  metrics_export: ${METRICS_EXPORT_PORT}"
echo "  workers: ${MIN_WORKER_PORT}-${MAX_WORKER_PORT}"
