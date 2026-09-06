#!/usr/bin/env bash
set -euo pipefail

# Run from the synced Linux checkout; keep the server's established resource envelope.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if pgrep -f '[s]cripts/train_offpolicy.py' >/dev/null; then
  echo "An offpolicy training process is already running." >&2
  exit 1
fi
total_cpus=$(nproc)
reserved_cpus=$((total_cpus / 8))
if (( reserved_cpus < 2 )); then reserved_cpus=2; fi
last_cpu=$((total_cpus - reserved_cpus - 1))
if (( last_cpu < 0 )); then
  echo "Not enough CPUs for training with system capacity reserved." >&2
  exit 1
fi
run_stamp=$(date +%Y%m%d-%H%M%S)
run_dir="$PWD/logs/FADAPrivilegedOracle_simple_height_${run_stamp}"
cache_root=/ssd1/chengyuxuan/icecal_cache
mkdir -p "$run_dir" "$cache_root/tmp" "$cache_root/uv" \
  "$cache_root/triton" "$cache_root/torchinductor" "$cache_root/torch"
export TRITON_CACHE_DIR="$cache_root/triton"
export TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
export TORCH_HOME="$cache_root/torch"
export UV_CACHE_DIR="$cache_root/uv"
export TMPDIR="$cache_root/tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD/src"
export ICE_CAL_ORACLE_LINEAGE_ID="simple-height-${run_stamp}"
taskset -c "0-${last_cpu}" nice -n 5 ionice -c2 -n5 \
  uv run --frozen --no-sync python scripts/train_offpolicy.py \
  algo=sac \
  task=sac/g1_walk_flat/mujoco_fada_privileged_oracle_simple_height_grouped_dr_lineage \
  training.device=cuda:0 training.no_play=true training.log_dir="$run_dir" \
  algo.num_envs=512 algo.batch_size=2048 algo.max_iterations=5000 \
  2>&1 | tee "$run_dir/train.log"
