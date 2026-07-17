# HP-7c3 Bounded Persistent Workflow Freeze

Status: design frozen; Gate 0 blocked before server materialization by SSH authentication.
Date: 2026-07-17

## Decision

E95 closes the production cache-wiring and RNG-equivalence boundary with a
no-op trainer. One real bounded persistent workflow is the correct next gate
because forward, backward, optimizer, checkpoint, manifest lineage, memory,
and end-to-end timing remain unconfirmed. This is a live integration sentinel,
not a legacy/persistent A/B, promotion trial, or policy-quality experiment.

## Step Map

### Gate 0 / 2: No-training identity freeze

- Objective: materialize one immutable execution identity and oracle before the
  output directory exists.
- Scope: read-only server source/config/manifest/artifact hashing, Hydra compose,
  effective replay-update calculation, GPU identity, dependency identity, and
  oracle syntax/contract checks.
- Non-scope: env construction, collection, learner updates, output creation,
  source/config edits, dependency sync, or training.
- Stop: any mismatch, missing artifact, existing output, effective update count
  above the cap, or oracle preflight failure records `BLOCKED` and returns.

### Gate 1 / 2: One bounded persistent workflow

- Objective: execute exactly the Gate-0 identity and apply the frozen oracle
  immediately after command exit zero.
- Scope: one forked outer iteration through the formal persistent route.
- Non-scope: retry, resume after failure, second run, legacy A/B, source/config
  repair, physical acceptance, default-on, promotion, commit, or PR.
- Stop: first command/oracle failure stops. Oracle PASS records the remaining
  HP-7c live boundary; no later action begins automatically.

## Frozen Local Source And Config Identity

- Git commit: `d3c2bc77c34c081ae3861668b8a415cd55cc25d5`.
- Runtime source paths must be byte-identical to this identity:
  - `offline.py`: `24d2230e98673625bc3202e600692b6eafe67ef88f5c99e2f345d3c41301d76f`
  - `workflow.py`: `22896114219d5e08df9893f158c38c7470675ac6546feac9ae0d74351f86d47c`
  - `async_runtime.py`: `69cf2a5ebc516c718454a75a96745534504c927efdcfecf5c5c6f44756aad7ae`
  - `persistent_runtime.py`: `4e88ee8af7cf09fbc8b30cbee45cd354886ba34dec8ba84bf7950f5b5d23f442`
  - `persistent_resources.py`: `7f2c936bfbb7d84a6bc09505801917536aa67785a734e75060e03edfb8d1e463`
  - `g1_persistent_worker.py`: `77af161718248f7e046bfcb3717cf68d8ffeda304fa81b17c9c0fdd6ae37bd7f`
  - `scripts/train_distill.py`: `b0e3f1f6d5760a7318acd5ba694f52992397bcf4a8e2852448444e8441eb273b`
- Config paths:
  - `conf/distill/config.yaml`: `64de26d85ffa058e09cf0344b7545bf8153704cf5f76216402a7758e1c9234da`
  - `conf/distill/workflow/g1_walk_stand.yaml`:
    `8e64ab659f1eaae169ecb6dd8b4059e5cc172464e78956b03d07fd954539e4ba`
- E95 sentinel identity remains supporting evidence, not the training entry:
  `scripts/deploy/check_distill_label_pool_production_path.py` hash
  `4645c00066400aea36ea81723dd12da4427778e2152a6cd53847b9b140a40939`.
- Gate 0 requires `git diff --quiet` for `src/`, `scripts/train_distill.py`,
  `conf/distill/`, and the installed project metadata. Documentation-only
  worktree changes do not enter the runtime identity.
- Dependency execution is `uv run --no-sync`; Gate 0 records `uv --version`,
  Python/Torch/CUDA/MuJoCo versions, `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`,
  and the resolved import paths for `unilab`, `torch`, and `mujoco`.

## Frozen Server Artifact Identity

- Parent run:
  `/ssd1/cyx/UniLab/logs/distill_workflow/g1_walk_stand_persistent_test01`.
- Teacher paths:
  - walk: `/ssd1/cyx/UniLab/model/G1WalkFlat/model_5000.pt`
  - stand: `/ssd1/cyx/UniLab/model/G1StandStill/model_5000.pt`
- Reused role datasets:
  - walk: `/ssd1/cyx/UniLab/model/walk_flat_teacher_policy.pt`
  - stand: `/ssd1/cyx/UniLab/model/stand_teacher_policy.pt`
- Parent checkpoint is the latest completed entry in
  `parent/run_manifest.json::dagger_iterations[-1].checkpoint_path`.
- Gate 0 writes the exact absolute paths, SHA-256 values, sizes, parent manifest
  hash, latest checkpoint hash, role artifact hashes, aggregate dataset path/
  hash/row count, and teacher hashes into the freeze JSON. Every value must
  agree with the parent manifest where recorded. Missing or disagreeing values
  block execution; no filename-only reuse is accepted.

## Frozen Workload And Output Identity

- Hydra profile: `workflow=g1_walk_stand`.
- Workflow mode: `fork` from the frozen parent.
- Execution mode: `persistent_async` explicitly; repository default remains
  `legacy`.
- Outer iterations: `1`.
- New rows: exactly `512` for each ordered scenario
  `walk_flat`, `static_stand`, `walk_to_stop`; quotas remain
  `0.50/0.25/0.25`.
- Collection envs: `16`.
- Batch size: `512`.
- Configured update floor: `512`.
- Effective updates: Gate 0 computes the exact value with the production
  `required_balanced_replay_updates()` contract over the frozen parent aggregate
  plus the three exact 512-row scenario additions. It must equal
  `max(512, required_updates)` and must be at most `8192`; otherwise execution
  is blocked. The frozen value is written into the identity and oracle.
- Seed: `algo.seed=0`.
- Device identity: `CUDA_VISIBLE_DEVICES=0`, logical `training.device=cuda:0`.
  Gate 0 records physical GPU UUID/name/driver/total memory and rejects a
  different visible-device mapping at execution time.
- Transition contract remains `pre_switch_steps=8`,
  `min_post_switch_steps=20`, walk command `[0.4,0.0,0.0]`, scenario balance
  key/labels from `g1_walk_stand`, and minimum transition replay passes `8`.
- Output run directory:
  `/ssd1/cyx/UniLab/logs/distill_workflow/hp7c3_bounded_persistent_20260717_r1`.
- Freeze JSON:
  `/ssd1/cyx/UniLab/hp7c3_bounded_persistent_freeze_r1.json`.
- Oracle and oracle result:
  `/ssd1/cyx/UniLab/hp7c3_bounded_persistent_oracle_v1.py` and
  `/ssd1/cyx/UniLab/hp7c3_bounded_persistent_oracle_result_r1.json`.
- Console and external memory logs:
  `/ssd1/cyx/UniLab/hp7c3_bounded_persistent_r1.log`,
  `/ssd1/cyx/UniLab/hp7c3_bounded_persistent_r1.time`, and
  `/ssd1/cyx/UniLab/hp7c3_bounded_persistent_r1.nvidia.csv`.
- Gate 0 requires all output/result/log paths to be absent. No overwrite,
  resume, cleanup, or retry is permitted after execution starts.

## Frozen Command Shape

Gate 0 stores this exact training argv/environment array in the freeze JSON.
The configured floor remains 512; the separately frozen oracle expects the
production-derived effective count computed by Gate 0:

```bash
CUDA_VISIBLE_DEVICES=0 \
UNILAB_G1_WALK_TEACHER=/ssd1/cyx/UniLab/model/G1WalkFlat/model_5000.pt \
UNILAB_G1_STAND_TEACHER=/ssd1/cyx/UniLab/model/G1StandStill/model_5000.pt \
UNILAB_G1_WALK_DATASET=/ssd1/cyx/UniLab/model/walk_flat_teacher_policy.pt \
UNILAB_G1_STAND_DATASET=/ssd1/cyx/UniLab/model/stand_teacher_policy.pt \
HYDRA_FULL_ERROR=1 PYTHONWARNINGS=ignore \
uv run --no-sync train --algo distill --task g1_walk_flat --sim mujoco \
  workflow=g1_walk_stand \
  algo.seed=0 training.device=cuda:0 \
  training.workflow.enabled=true \
  training.workflow.mode=fork \
  training.workflow.parent_run_dir=/ssd1/cyx/UniLab/logs/distill_workflow/g1_walk_stand_persistent_test01 \
  training.workflow.run_dir=/ssd1/cyx/UniLab/logs/distill_workflow/hp7c3_bounded_persistent_20260717_r1 \
  training.workflow.execution_mode=persistent_async \
  training.workflow.collect_num_envs=16 \
  training.workflow.dagger_samples_per_role=512 \
  training.workflow.dagger_iterations=1 \
  training.workflow.dagger_batch_size=512 \
  training.workflow.dagger_updates_per_iteration=512
```

Gate 1 wraps this command with `/usr/bin/time -v -o` for the frozen `.time`
path and redirects stdout/stderr to the frozen `.log` path. Before launching
it, the supervisor starts exactly one `nvidia-smi` sampler at 250 ms intervals
with fields `timestamp,pid,gpu_uuid,used_gpu_memory`, writing the frozen CSV;
an EXIT trap stops only that sampler PID. Gate 0 stores the fully materialized
supervisor script bytes and SHA-256 in the freeze JSON. The command is design
text only and is not authorized for execution by this planning step.

## Frozen Acceptance Oracle V1

The oracle is generated and hashed during Gate 0, syntax-checked, and applied
only after command exit zero. `accepted=true` requires every hard gate below:

1. Freeze/source/config/dependency/GPU identities match immediately before and
   after execution; command argv/environment equal the frozen arrays.
2. Output was absent before execution and contains one new `run_manifest.json`;
   the manifest names the frozen parent and parent-manifest hash.
3. Stage is `DAGGER_ITERATION_1_COMPLETE`, completed iterations is `1`, and
   execution mode is `persistent_async`.
4. Scenario order is exactly the frozen three-scenario order; each artifact has
   512 rows, the expected scenario label, complete transition schema, and
   manifest-matching SHA-256.
5. All scenarios in outer iteration 1 use one non-null input checkpoint hash
   and one weight version; worker PID is stable and distinct from the workflow
   PID. No scenario observes a different student version inside the barrier.
6. Aggregate row count/hash equal the preflight parent rows plus 1536 new rows;
   aggregate sources preserve the parent and three scenario identities.
7. Real learner evidence exists: update count equals the frozen effective
   value; `learner_forward`, `learner_backward`, `optimizer_step`, and
   `checkpoint_save` metrics are present, successful, finite, and positive.
8. `learner_batch_staging` is present, successful, finite, and its normalized
   duration is at most `0.010 s/update`. This is a production-regression bound
   against E95's `0.004232 s/update`, not a formal A/B speedup claim.
9. The new checkpoint exists, is finite/loadable, and its SHA-256 equals the
   manifest. Input checkpoint, teacher, role artifact, and parent run bytes are
   unchanged after execution.
10. Cleanup-final metrics exist and report successful worker/resource cleanup;
    the metrics artifact reloads under the current schema with all required
    request/workflow/learner/checkpoint/cleanup stages.
11. `/usr/bin/time -v` records finite maximum resident set size. The GPU sidecar
    records timestamp/PID/UUID/used-memory samples for workflow and worker, no
    CUDA OOM appears, and observed memory never exceeds the frozen physical GPU
    capacity. These are bounded-run diagnostics, not comparative memory claims.

The oracle always records all observed values and failed clauses. It must not
delete, repair, resume, or reinterpret an artifact. End-to-end elapsed time,
staging share, and the E89/E92/E95 comparisons are reported as observations;
they cannot change `accepted` into a speedup, default-on, or promotion verdict.

## Stop And Human-Control Boundary

- This planning step stops after document/check consistency checks.
- Gate 0 requires separate authorization and returns control with the complete
  freeze/oracle hashes before any training.
- Gate 1 requires another explicit authorization after Gate 0 PASS.
- Any command failure or oracle rejection stops permanently at r1; do not retry
  or resume without a new human decision and new output identity.
- Even a full PASS closes only HP-7c's bounded integration evidence. Persistent
  remains OFF-default and promotion remains unauthorized.

## Gate 0 Attempt

E96 records the authorized first attempt. A read-only `BatchMode` SSH
discriminator to the configured `SUST_4090` alias reached the host but failed
authentication with `Permission denied (publickey,password)` before the remote
`cd`, HEAD read, artifact read, or any write executed. No freeze/oracle/output
path was created and Gate 1 remains closed. Resume Gate 0 only through a
user-authenticated SSH session or an explicitly provided non-interactive
connection identity; do not try passwords or alternate hosts automatically.
