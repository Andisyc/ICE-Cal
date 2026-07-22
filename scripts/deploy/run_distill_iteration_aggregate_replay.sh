#!/usr/bin/env bash
set -u -o pipefail

cd /ssd1/cyx/UniLab || exit 2

RUN_DIR="${1:-/ssd1/cyx/UniLab/logs/distill_workflow/20260722-151433_g1-walk-stand-ownerfix-r2}"
ITERATION="${2:-7}"
OUT_ROOT="${3:-/ssd1/cyx/UniLab/logs/distill_iteration_replay}"
TOKEN="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="${OUT_ROOT}/${TOKEN}_iteration_${ITERATION}_aggregate_replay"

mkdir -p "${OUT_DIR}"

{
  echo "cwd=$(pwd)"
  echo "run_dir=${RUN_DIR}"
  echo "iteration=${ITERATION}"
  echo "out_dir=${OUT_DIR}"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "git_status=$(git status --short 2>/dev/null || true)"
  echo "uv=$(command -v uv || true)"
  echo "python=$(uv run python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
} > "${OUT_DIR}/precheck.txt" 2>&1

uv run python scripts/deploy/replay_distill_iteration_aggregate.py \
  --run-dir "${RUN_DIR}" \
  --iteration "${ITERATION}" \
  --device cpu \
  --report "${OUT_DIR}/replay-report.json" \
  > "${OUT_DIR}/console.log" 2>&1
STATUS=$?

tar -czf "${OUT_DIR}-RETURN_ME.tar.gz" -C "$(dirname "${OUT_DIR}")" "$(basename "${OUT_DIR}")"

echo "status=${STATUS}"
echo "report=${OUT_DIR}/replay-report.json"
echo "archive=${OUT_DIR}-RETURN_ME.tar.gz"

exit "${STATUS}"
