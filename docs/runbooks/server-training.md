# Server training resource control

This runbook defines the operator-side resource envelope for foreground training on a single-GPU
Linux server. It complements the selected Hydra task profile; it does not replace the profile or
authorize a run.

## Why SSH can become unresponsive

MuJoCo environment collection is CPU-heavy. A large `algo.num_envs` can saturate all logical CPUs
and create memory pressure, while BLAS/OpenMP libraries may create additional worker threads. GPU
utilization can therefore coexist with an unresponsive SSH session.

The controls have different jobs:

- `taskset` reserves logical CPUs for SSH and system services;
- `nice` and `ionice` let interactive and system work pre-empt training;
- BLAS/OpenMP limits prevent nested thread oversubscription;
- `algo.num_envs` controls the dominant MuJoCo collection load.

Thread variables alone do not reserve CPU capacity. For a shared or modest server, use CPU
affinity as well.

## Preflight

Run these read-only checks before starting a new process:

```bash
cd /ssd1/cyx/ICE-Cal
pgrep -af 'scripts/train_offpolicy.py'
nvidia-smi
nproc
```

Do not start a second training process unless concurrent training was explicitly intended.

## Foreground privileged-Oracle command

This command intentionally does not use `nohup`. It reserves at least two, or roughly one eighth,
of the logical CPUs for SSH and starts from `1024` MuJoCo environments instead of the profile's
`2048`-environment maximum.

```bash
cd /ssd1/cyx/ICE-Cal

TOTAL_CPUS=$(nproc)
RESERVED_CPUS=$((TOTAL_CPUS / 8))
if [ "$RESERVED_CPUS" -lt 2 ]; then RESERVED_CPUS=2; fi
TRAIN_LAST_CPU=$((TOTAL_CPUS - RESERVED_CPUS - 1))
if [ "$TRAIN_LAST_CPU" -lt 0 ]; then
  echo "Not enough logical CPUs to reserve an SSH core set" >&2
  exit 1
fi

RUN_DIR=/ssd1/cyx/ICE-Cal/model/FADAPrivilegedOracle_v015_seed1
mkdir -p "$RUN_DIR"

taskset -c "0-${TRAIN_LAST_CPU}" \
nice -n 5 \
ionice -c2 -n5 \
env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=4 \
  OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 \
  NUMEXPR_NUM_THREADS=4 \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=/ssd1/cyx/ICE-Cal/src \
  ICE_CAL_ORACLE_LINEAGE_ID=fada-v015-seed1 \
uv run --no-sync python scripts/train_offpolicy.py \
  algo=sac \
  task=sac/g1_walk_flat/mujoco_fada_privileged_oracle \
  training.device=cuda:0 \
  training.no_play=true \
  training.log_dir="$RUN_DIR" \
  algo.num_envs=1024
```

The selected task profile still owns the algorithm, Reward, maximum iterations, save interval,
checkpoint lineage, and domain-randomization semantics. The command overrides only the operational
environment count and device/log identities.

## Resource adjustment rule

Keep the CPU reservation and thread limits fixed. Adjust only `algo.num_envs` first:

| Condition | Action |
|---|---|
| SSH is still delayed or memory pressure is high | reduce to `algo.num_envs=512` |
| SSH is responsive and the GPU is materially underutilized | try `algo.num_envs=1536` |
| The server remains responsive at 1536 and the operator accepts the risk | use the profile default `2048` by removing the override |

Change one resource control at a time. A higher environment count is throughput tuning, not a
method change, but it changes the effective runtime configuration and should be recorded with the
run.

## Stop and cleanup

Because the process is foreground, use `Ctrl-C` once and wait for the collector child to exit. In
another SSH session, verify cleanup with:

```bash
pgrep -af 'scripts/train_offpolicy.py'
nvidia-smi
```

Do not use an unconditional recursive kill. If a child remains, identify its exact PID and parent
before terminating it.

## Evidence boundary

A responsive launch proves only that the selected process can start inside this resource envelope.
It does not prove convergence, policy quality, checkpoint superiority, Planner-IDM readiness, or
deployment safety.

