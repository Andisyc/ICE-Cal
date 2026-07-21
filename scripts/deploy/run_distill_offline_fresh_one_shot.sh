#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

RUN_GROUPS="${RUN_GROUPS:-offline_device}" \
BATCH_SIZE="${BATCH_SIZE:-2048}" \
FRESH_UPDATES="${FRESH_UPDATES:-8192}" \
LIFECYCLE_UPDATES="${LIFECYCLE_UPDATES:-1}" \
LIFECYCLE_ROUNDS="${LIFECYCLE_ROUNDS:-1}" \
bash "${SCRIPT_DIR}/run_distill_real_owner_one_shot.sh"
