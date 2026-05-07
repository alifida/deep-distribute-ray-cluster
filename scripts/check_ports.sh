#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <target_ip> [quick|full]"
  echo "  quick: checks key control ports only"
  echo "  full:  checks control + worker port ranges (default)"
  exit 1
fi

TARGET_IP="$1"
MODE="${2:-full}"

check_port() {
  local host="$1"
  local port="$2"
  if timeout 1 bash -c ">/dev/tcp/${host}/${port}" 2>/dev/null; then
    echo "OK   ${host}:${port}"
    return 0
  fi
  echo "FAIL ${host}:${port}"
  return 1
}

expand_range() {
  local start="$1"
  local end="$2"
  local p
  for ((p=start; p<=end; p++)); do
    echo "$p"
  done
}

declare -a CONTROL_PORTS=(
  6379
  8265
  10001 10002 10003 10004 10005
  10011 10012 10014 10015
)

declare -a WORKER_RANGE_HEAD=()
declare -a WORKER_RANGE_WORKER=()
while IFS= read -r p; do WORKER_RANGE_HEAD+=("$p"); done < <(expand_range 11000 11999)
while IFS= read -r p; do WORKER_RANGE_WORKER+=("$p"); done < <(expand_range 12000 12999)

echo "[check] target=${TARGET_IP} mode=${MODE}"
echo

fail_count=0

echo "[check] control ports"
for port in "${CONTROL_PORTS[@]}"; do
  if ! check_port "${TARGET_IP}" "${port}"; then
    fail_count=$((fail_count + 1))
  fi
done

if [[ "${MODE}" == "full" ]]; then
  echo
  echo "[check] worker port range 11000-11999"
  for port in "${WORKER_RANGE_HEAD[@]}"; do
    if ! check_port "${TARGET_IP}" "${port}"; then
      fail_count=$((fail_count + 1))
    fi
  done

  echo
  echo "[check] worker port range 12000-12999"
  for port in "${WORKER_RANGE_WORKER[@]}"; do
    if ! check_port "${TARGET_IP}" "${port}"; then
      fail_count=$((fail_count + 1))
    fi
  done
fi

echo
if [[ "${fail_count}" -eq 0 ]]; then
  echo "[check] PASS: all tested ports are reachable."
else
  echo "[check] FAIL: ${fail_count} port checks failed."
  exit 2
fi
