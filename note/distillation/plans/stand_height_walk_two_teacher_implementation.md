# StandHeight And Walk Two-Teacher Implementation Plan

Date: 2026-07-23

Active method: `DISTILL-METHOD-v002`.

Objective: implement a new 99-D `G1StandHeight` teacher route and a separate
two-expert StandHeight/Walk distillation workflow while preserving every legacy
98-D task, config, checkpoint, and dataset.

## Current Plan Cursor

This table is the current control state. Detailed proof remains in the
checklist and evidence ledger rather than becoming a chronological plan log.

| Step | Status | Recorded evidence | Remaining boundary |
| --- | --- | --- | --- |
| Step 1: Govern Method Semantics | PASS | Active v002 contract, six mapped Concept Figure nodes, and Atlas contract check PASS | No semantic gap |
| Step 2: Implement G1StandHeight | PASS | E114: retained focused-suite record, `108 passed, 24 warnings in 19.46s` | No live or teacher-quality claim |
| Step 3: Implement Actor-Only Migration | PASS | E114: adapter/connector suite, `8 passed in 6.77s` | No real checkpoint was read or converted |
| Step 4: Implement Unified Workflow | PASS | E113: Ruff PASS and `27 passed in 20.56s` | No trained student or physical-quality claim |
| Step 5: Bounded Runtime Evidence | PASS | E115: one-env/one-step MuJoCo sentinel and training compose preflight PASS | No checkpoint, training, or policy-quality claim |

Post-closure runtime addendum: E116 confirms that StandHeight SAC dispatches to
the existing `AsyncRunner`-derived `DoubleBufferOffPolicyRunner`, and that the
99-D two-role workflow reaches the optional persistent DAgger runtime with all
three scenario contracts. This does not add a sixth method step or promote the
distillation default; E67 still forbids a stable-speedup claim.

## Step Map

### Step 1 / 5: Govern Method Semantics

Objective: activate the accepted two-teacher method and synchronize its human
control surfaces.

Scope: contract registry, Concept Figure, Method-to-Code status, task canvas,
and acceptance checklist.

Non-scope: environment or training code.

Owner files/modules: `note/distillation/` and `note/architecture/`.

Expected evidence: JSON/atlas contract checks and zero active `v001` mappings.

Stop condition: every Concept Figure design point maps to `DISTILL-METHOD-v002`.

### Step 2 / 5: Implement G1StandHeight

Objective: expose zero-velocity target-height tracking without changing
`G1StandStill` or `G1WalkFlat`.

Scope: owner config/registry, 99-D observation, dynamic target-aware standing
rewards, and deterministic config/reward tests.

Non-scope: teacher quality or long training.

Owner files/modules: G1 task configs, reward context/helpers, Hydra task YAML.

Expected evidence: S0 compose, S1 value/metamorphic, S2 legacy isolation.

Stop condition: low/mid/high targets select their own reward optimum and legacy
tasks remain byte-for-byte semantically unchanged at their public contract.

### Step 3 / 5: Implement Actor-Only 98-D To 99-D Migration

Objective: warm-start compatible SAC actors without dimension-tolerant loading.

Scope: actor and actor-normalizer conversion, immutable source/output hashes,
adapter metadata, fresh critic/replay/optimizer boundary, and tests.

Non-scope: critic, replay, or optimizer migration.

Owner files/modules: off-policy checkpoint migration owner and focused tests.

Expected evidence: S1 shape/persistence/metamorphic output equivalence within
`1e-6` and fail-closed incompatible payloads.

Stop condition: converted actors reproduce legacy outputs for matched inputs.

### Step 4 / 5: Implement Unified Two-Teacher Workflow

Objective: compose a new 99-D, two-expert workflow without mutating
`g1_walk_stand`.

Scope: `walk` and `stand_height` roles, expert mapping, role/scenario config,
dataset preflight, checkpoint roundtrip, and connector tests.

Non-scope: reuse of old 98-D datasets or promotion of any checkpoint.

Owner files/modules: Hydra distill profile and existing workflow/data/trainer
owners only where a general contract is missing.

Expected evidence: S0 compose, S1 labels/grad/roundtrip, S2 formal route.

Stop condition: both roles share 99-D and one bounded update reloads one finite
29-D action while only the selected expert changes.

### Step 5 / 5: Bounded Runtime Evidence

Objective: prove the new task route executes before any material training run.

Scope: one-env MuJoCo sentinel and evidence/checklist/atlas refresh.

Non-scope: teacher quality claims, long teacher training, final DAgger, policy
promotion, commit, push, or PR.

Owner files/modules: G1 live sentinel and evidence ledger.

Expected evidence: S3/S4 snapshot with target/measured height, observation
shape, support, tilt, termination, exact config, and command identity.

Stop condition: connector/live facts are recorded honestly and the remaining
teacher-checkpoint/training boundary is explicit.

Runtime probe contract:

- Owner: `scripts/deploy/check_unilab_g1_height_tracking_live_path.py`.
- Core path: `g1_stand_height` Hydra owner -> `BackendAdapter` -> one MuJoCo
  environment -> `height_commands[:, 0]` -> actor observation index 96 ->
  reward and structured runtime snapshot.
- Focused command: `uv run pytest
  tests/scripts/test_train_scripts.py::test_g1_height_tracking_live_path_stand_height_contract
  -q`.
- Live command: `uv run
  scripts/deploy/check_unilab_g1_height_tracking_live_path.py --task
  g1_stand_height --num-envs 1 --steps 1 --seed 7`.
- Expected facts: actor/critic dimensions `99/102`, zero velocity command,
  target/observation equality, finite target/measured/reward/tilt/support facts,
  and zero one-step terminations.
- Training preflight: compose the public `uv run train --algo sac --task
  g1_stand_height --sim mujoco` route with Hydra `--cfg job --resolve`; never
  enter `runner.learn()` in Step 5.

## Training-Ready Stop

Steps 1-5 now form the completed pre-training closure. E115 crosses the bounded
live-runtime boundary without reading a checkpoint or entering the learner.
E116 additionally closes the async connector boundary with synthetic fixtures.

The following SSH command is the preflighted first curriculum stage. It starts
fresh SAC training at fixed `0.754 m`; it intentionally does not set
`algo.actor_warm_start_checkpoint` because no source checkpoint has been
qualified. The repository-path assumption is `/ssd1/cyx/UniLab`.

This SAC entrypoint already constructs the repository's async collector and
CPU-pinned double-buffer replay pipeline with one-tick prefetch. Do not set
`training.no_sync_collection=true`; that option is rejected by this owner
because its collector handoff is synchronized even though collection runs in a
separate process.

```bash
cd /ssd1/cyx/UniLab
RUN_ID="$(date +%Y%m%d-%H%M%S)_g1_stand_height_stage1_fixed_0754"
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 uv run train \
  --algo sac \
  --task g1_stand_height \
  --sim mujoco \
  --render-mode none \
  training.no_play=true \
  "training.log_dir=/ssd1/cyx/UniLab/logs/G1StandHeight/${RUN_ID}" \
  'env.commands.height_range=[0.754,0.754]' \
  env.commands.default_height=0.754
```

Do not add a warm-start override unless the exact legacy source path, actor
shape, normalizer payload, and hash have first passed the Step 3 adapter gate.

## Authority And Continuation Boundary

Step 5 was a separate user-visible unit because it started MuJoCo and crossed
from deterministic evidence into a live-runtime boundary. E115 closes that
unit. Executing the command above starts material training and remains a human
action on SSH.

The previously agreed post-Step-5 plan is also recorded but inactive:

1. Qualify exact legacy teacher source identities and use only the explicit
   actor-only adapter for any 98-D to 99-D migration.
2. Train StandHeight first at fixed `0.754 m`, then over `[0.65, 0.754] m`, and
   expand toward `0.50 m` only after the preceding teacher-quality gate passes.
3. Start the two-teacher 99-D distillation and cumulative student-state DAgger
   workflow only after both teacher artifacts are qualified. Persistent
   execution remains explicit opt-in through
   `training.workflow.execution_mode=persistent_async`.
4. Evaluate repeated reset, both transition directions, non-nominal post-walk
   height recovery, and bounded support/tilt/termination before promotion.

Those units require real checkpoint access or material compute, persisted run
identity, and new live-acceptance evidence. E115 does not authorize or prove
their training or policy-quality outcomes.
