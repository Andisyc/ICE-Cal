# Non-Nominal Walk-To-StandHeight DAgger Repair

Date: 2026-07-24

Active contracts: `DISTILL-METHOD-v002` and `DISTILL-TRAIN-v003`.

## Problem

The runtime-confirmed 99-D student checkpoint at
`20260724-110852_stand_height_walk_dagger_round2` failed the governed
Walk-to-StandHeight acceptance gate. Five of nine command-height cases passed.
All three forward cases and lateral-to-`0.754 m` terminated before completing
the recovery window. Command, target-height, and actor-observation roundtrip
checks were exact, so the current evidence points to a student data-distribution
gap rather than an evaluator input-routing defect.

The active method contract already requires nominal and non-nominal
Walk-to-StandHeight transitions. The current collector implementation does not
produce that distribution: `walk_to_stop` uses one global forward command and
the `g1_walk_height_nominal` environment keeps target height fixed at
`0.754 m` before and after the switch.

## Scope

- Keep one `walk_to_stop` scenario, two roles, two experts, and the existing
  cumulative DAgger/fork lineage.
- Add a config-owned Cartesian grid of active walk commands and post-switch
  StandHeight targets.
- Keep active walking at the nominal `0.754 m` target.
- Atomically write command and target height before one observation refresh at
  each transition boundary.
- Route the same collector semantics through legacy and `persistent_async`
  execution.
- Record per-case sample counts and post-switch horizon evidence in dataset
  metadata.

## Non-Scope

- No acceptance-threshold changes.
- No new role, expert, scenario label, router, checkpoint schema, or backend.
- No teacher/checkpoint mutation.
- No local MuJoCo run and no training launch.
- No claim that deterministic tests establish physical policy quality.

## Core Parameter Path

```text
Hydra transition_walk_commands + transition_post_switch_target_heights
-> legacy/persistent workflow connector
-> transition collector case rows
-> active command + nominal 0.754 target
-> zero command + case-specific recovery target
-> refreshed 99-D actor observation
-> matching teacher relabel
-> target_height/command_before/command_after dataset rows
-> cumulative scenario-balanced DAgger update
```

## Invariants

1. The configured production grid is exactly forward/lateral/yaw crossed with
   `0.650/0.702/0.754 m`.
2. Every active row uses its configured non-zero command and target
   `0.754 m`.
3. Every post-switch row uses zero velocity and its assigned recovery target.
4. `target_height` equals actor observation index 96 for the 99-D workflow.
5. Every configured case contributes post-switch rows and reaches the required
   minimum post-switch age, or collection fails closed.
6. The legacy singular `transition_walk_command` remains the fallback when no
   command-height grid is configured.
7. Persistent execution remains explicit opt-in and uses the existing resident
   worker/resource lifecycle.

## Owner Files

- `conf/distill/config.yaml`: backward-compatible grid defaults.
- `conf/distill/workflow/g1_stand_height_walk.yaml`: governed 3x3 grid.
- `src/unilab/algos/torch/distill/collector.py`: case assignment, atomic input
  update, relabel rows, and metadata evidence.
- `scripts/train_distill.py`: legacy connector forwarding.
- `src/unilab/algos/torch/distill/g1_persistent_worker.py`: resident connector
  forwarding.
- Focused tests: collector semantic fixture, workflow compose, and persistent
  connector propagation.

## Test Matrix

| Boundary | Class | Expected evidence |
| --- | --- | --- |
| Config compose | S0 / T-contract | exact 3x3 grid; legacy defaults unchanged |
| Collector | S1 / T-value+roundtrip | 9 cases, active `0.754`, post-switch requested height, zero command, obs index 96 equality |
| Negative collector contract | S1 / T-negative | missing target field or insufficient env rows fails closed |
| Legacy connector | S2 / T-connect | grid values reach collector callback |
| Persistent connector | S2/S3 / T-connect+lifecycle | identical grid values reach resident worker without lifecycle changes |
| Existing distill regressions | S1/S2 | old singular command and cumulative workflow tests remain PASS |

## Stop Condition

Focused tests, Ruff, and Atlas/document consistency checks pass; the exact SSH
fork command is derived from the existing round-2 manifest identity. Stop
before checkpoint access, MuJoCo, or training.

## Local Closure Status

- Implementation: PASS.
- Core collector and persistent differential tests:
  `10 passed, 95 deselected in 0.71s`.
- Workflow and persistent connector tests: `7 passed in 4.64s`.
- Focused Ruff lint and format: PASS.
- Atlas/document consistency: PASS; `npm.cmd run check` reported
  `runtime_modules=9 method_modules=11 concept_nodes=6`.
- Live boundary: not executed. The current round-2 parent remains immutable.

## SSH Fork Command

Run only after this local change is committed, pushed to
`codex/stand-height-walk-async`, and pulled on the SSH checkout. This creates a
new run and one new outer DAgger iteration; it does not resume or overwrite the
round-2 parent.

```bash
cd /ssd1/cyx/liujun/UniLab

PARENT_RUN_DIR=/ssd1/cyx/liujun/UniLab/logs/distill_workflow/20260724-110852_stand_height_walk_dagger_round2
RUN_ID="$(date +%Y%m%d-%H%M%S)_stand_height_walk_non_nominal_grid_r1"
RUN_DIR="/ssd1/cyx/liujun/UniLab/logs/distill_workflow/${RUN_ID}"

CUDA_VISIBLE_DEVICES=0 \
HYDRA_FULL_ERROR=1 \
UNILAB_G1_WALK_HEIGHT_TEACHER=/ssd1/cyx/liujun/UniLab/logs/G1WalkHeight/20260724-020039_g1_walk_height_nominal_0754/model_5000.pt \
UNILAB_G1_STAND_HEIGHT_TEACHER=/ssd1/cyx/liujun/UniLab/logs/G1StandHeight/20260724-013445_g1_stand_height_stage2_065_0754/model_5000.pt \
UNILAB_G1_WALK_HEIGHT_DATASET=/ssd1/cyx/liujun/UniLab/logs/distill_role_artifacts/20260724-022552_stand_height_walk_dagger/walk.pt \
UNILAB_G1_STAND_HEIGHT_DATASET=/ssd1/cyx/liujun/UniLab/logs/distill_role_artifacts/20260724-022552_stand_height_walk_dagger/stand_height.pt \
uv run --no-sync train \
  --algo distill \
  --task g1_walk_height_nominal \
  --sim mujoco \
  --render-mode none \
  workflow=g1_stand_height_walk \
  training.device=cuda:0 \
  training.workflow.mode=fork \
  training.workflow.parent_run_dir="$PARENT_RUN_DIR" \
  training.workflow.run_dir="$RUN_DIR" \
  training.workflow.execution_mode=persistent_async \
  training.workflow.collect_num_envs=64 \
  training.workflow.dagger_iterations=1
```

Expected terminal result:

```text
mode=fork
stage=DAGGER_ITERATION_1_COMPLETE
completed_dagger_iterations=1
```

The new checkpoint will be:

```text
$RUN_DIR/checkpoints/dagger_iteration_1.pt
```

After training, preserve `RUN_DIR`, checkpoint path, SHA-256, manifest path,
and the complete command output before running the unchanged nominal and
non-nominal physical acceptance gates.

## SSH Execution Result

- Date: 2026-07-27.
- Server checkout: `bef2a91b44854a61b8f86be2d6ba4d632ee77e5b` on
  `codex/stand-height-walk-async`, clean and synchronized.
- The package CLI already injects `training.workflow.enabled=true` for
  `--algo distill`; passing it again is rejected as a duplicate route override.
- The non-interactive SSH launcher used
  `/home/chengyuxuan/.local/bin/uv` because the server's default non-login
  `PATH` does not contain `~/.local/bin`.
- Immutable r1 run:
  `/ssd1/cyx/liujun/UniLab/logs/distill_workflow/20260727-104620_stand_height_walk_non_nominal_grid_r1`.
  It completed one persistent-async iteration with 1,114,112 cumulative rows,
  12,288 updates, and checkpoint SHA-256
  `13378dd0c7c7478307692b775bd72305fb1a4bfd2d2fffe7e1a96d1ca84844f9`.
- Collection evidence: nine command-height cases, 7,168-8,192 rows per case,
  7,024-8,000 post-switch rows per case, and maximum post-switch age 968.
- Unchanged seed-1 gate: FAIL. Nominal lateral stop-speed decay failed;
  non-nominal recovery passed 4/9 and failed forward at all three heights plus
  lateral at 0.650 and 0.754 m. All command/height/observation synchronization
  errors remained zero.
- A child-state r2 immutable fork collected successfully but exited before
  aggregate completion. Its exact 14-source aggregate passes in-memory at
  1,310,720 rows. A save/reload replay produced exit 139; the same replay
  passed under GDB, so the fault is native-symptom-confirmed and timing/layout
  sensitive, not owner-confirmed.

Stop before another training retry. The next step is a scoped repair or
first-invalid-operation capture at the CPU aggregate save/reload boundary,
followed by a new immutable fork and the same acceptance command.
