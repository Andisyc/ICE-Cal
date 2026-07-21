#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

RUN_DIR="${RUN_DIR:-/ssd1/cyx/UniLab/logs/distill_workflow/g1_walk_stand_rt10}"
WORK_ROOT="${WORK_ROOT:-/ssd1/cyx/UniLab/logs/native_core}"
WALK_TEACHER="${WALK_TEACHER:-/ssd1/cyx/UniLab/model/G1WalkFlat/model_5000.pt}"
STAND_TEACHER="${STAND_TEACHER:-/ssd1/cyx/UniLab/model/G1StandStill/model_5000.pt}"
GPU_DEVICE="${GPU_DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-512}"
FRESH_UPDATES="${FRESH_UPDATES:-16384}"
LIFECYCLE_UPDATES="${LIFECYCLE_UPDATES:-16384}"
LIFECYCLE_ROUNDS="${LIFECYCLE_ROUNDS:-3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-43200}"
RUN_GROUPS="${RUN_GROUPS:-all}"
STAGE_NAMES="${STAGE_NAMES:-all}"
NATIVE_ABORT_ON_CORRUPTION="${NATIVE_ABORT_ON_CORRUPTION:-0}"

OWNER_PATHS="$(
  uv run python - "${RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
manifest_path = run / "run_manifest.json"
manifest = json.loads(manifest_path.read_text())
iterations = manifest.get("dagger_iterations") or []
if not iterations:
    raise SystemExit(f"no dagger_iterations in {manifest_path}")

last = iterations[-1]
aggregate = last.get("aggregate_dataset_path")
checkpoint = last.get("checkpoint_path")
if not aggregate or not checkpoint:
    raise SystemExit(f"missing aggregate/checkpoint in latest iteration of {manifest_path}")

print(aggregate)
print(checkpoint)
PY
)"

AGGREGATE="$(printf '%s\n' "${OWNER_PATHS}" | sed -n '1p')"
CHECKPOINT="$(printf '%s\n' "${OWNER_PATHS}" | sed -n '2p')"

if [[ -z "${EXISTING_APPORT:-}" ]]; then
  EXISTING_APPORT="$(
    find /var/lib/apport/coredump /var/crash -maxdepth 1 -type f \
      \( -name 'core._usr_bin_python3_10*' -o -name 'core._usr_bin_python3.10*' -o -name '*.crash' \) \
      -printf '%T@ %p\n' 2>/dev/null |
      sort -nr |
      head -n 1 |
      cut -d' ' -f2-
  )"
fi

if [[ -z "${EXISTING_APPORT}" ]]; then
  echo "ERROR: no existing apport/core file found." >&2
  echo "Set EXISTING_APPORT=/path/to/core_or_crash and rerun." >&2
  exit 2
fi

echo "RUN_DIR=${RUN_DIR}"
echo "AGGREGATE=${AGGREGATE}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "EXISTING_APPORT=${EXISTING_APPORT}"
echo "WORK_ROOT=${WORK_ROOT}"
echo "GPU_DEVICE=${GPU_DEVICE}"
echo "RUN_GROUPS=${RUN_GROUPS}"
echo "STAGE_NAMES=${STAGE_NAMES}"
echo "NATIVE_ABORT_ON_CORRUPTION=${NATIVE_ABORT_ON_CORRUPTION}"

EXTRA_ARGS=()
if [[ "${NATIVE_ABORT_ON_CORRUPTION}" == "1" ]]; then
  EXTRA_ARGS+=(--native-abort-on-corruption)
fi

uv run scripts/deploy/diagnose_distill_real_owner_one_shot.py \
  --work-root "${WORK_ROOT}" \
  --aggregate "${AGGREGATE}" \
  --checkpoint "${CHECKPOINT}" \
  --teacher-checkpoint "${WALK_TEACHER}" \
  --stand-teacher-checkpoint "${STAND_TEACHER}" \
  --existing-apport "${EXISTING_APPORT}" \
  --gpu-device "${GPU_DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --fresh-updates "${FRESH_UPDATES}" \
  --lifecycle-updates "${LIFECYCLE_UPDATES}" \
  --lifecycle-rounds "${LIFECYCLE_ROUNDS}" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  --groups "${RUN_GROUPS}" \
  --stage-names "${STAGE_NAMES}" \
  "${EXTRA_ARGS[@]}"
