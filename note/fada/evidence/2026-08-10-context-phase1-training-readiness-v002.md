# Context Phase-1 v002 Training Readiness Evidence

Date: 2026-08-10
Branch: `codex/fada-quality-repair`
Contracts: `FADA-CONTEXT-PHASE1-METHOD-v002`, `FADA-CONTEXT-PHASE1-TRAIN-v002`

## Accepted authority

The human accepted the proposed formal teacher protocol before training:

- held-out seeds `[101, 102, 103, 104, 105]`;
- `256` environments per seed and `400` steps at command `(0.4, 0.0, 0.0)` m/s;
- equal arithmetic mean of per-seed, per-stratum summaries;
- independent left/right-knee gates: at least `10%` reduction in maximum lateral and yaw error,
  forward-velocity MAE degradation at most `2%`, fall rate no worse than nominal and at most `1%`,
  and clipping step rate at most `1%`;
- nominal-stratum trajectory degradation at most `2%`, fall rate no worse and at most `1%`, and
  clipping step rate at most `1%`.

The v001 contracts were moved to history. The active registry, plan, checklist, and task canvas now
point to v002. Existing v001 evidence retains its historical contract identity.

## Implemented owners

- `formal_protocol.py` owns the formal constants, full Hydra profile validator, evaluation protocol
  validator, and machine-readable conjunctive quality gate.
- `evaluate_context_teacher_phase1.py` defaults to the formal protocol. Shortened or changed runs are
  `unassessed`; an exact formal run emits `passed` or `failed` and exits nonzero on failure.
- `PrivilegedResidualSACRuntime.validate_training_config` is invoked by `train_offpolicy.build_runner`
  before environment creation. Direct launch cannot bypass formal profile validation.
- `preflight_context_teacher_phase1.py` builds and closes the real runner/learner without invoking
  `learn`, starting a collector, creating a run directory, or saving a checkpoint.
- The task YAML explicitly freezes the inherited G1 SAC training values and disables automatic
  post-training playback.

## Actual no-training preflight

Command:

```text
UV_CACHE_DIR=/private/tmp/fada-uv-cache uv run python \
  scripts/preflight_context_teacher_phase1.py \
  --device cpu \
  --planned-log-dir /private/tmp/fada-context-phase1-formal-not-started-20260810 \
  --output /private/tmp/fada-context-phase1-preflight-v002-launch-hook-20260810.json
```

Observed:

```text
status=passed
training_started=false
collector_started=false
runner=DoubleBufferOffPolicyRunner
learner=PrivilegedResidualSACLearner
actor=PrivilegedResidualSACActor
dimensions=(obs=98, critic=130, action=29, g=29)
sync_collection=true
env_steps_per_sync=1
nominal_sha256=db2f536f1391f7bdd92d22afe065170e32239834e398b685488bcb5ba5b63291
```

The planned training directory did not exist before preflight and remained absent afterward. The
preflight artifact SHA-256 is
`d30529cdebb272a1e0386f23d75d950a3a1e14573e5f6c97b821fc1d324c3bc7`.

## Gate behavior evidence

A shortened `64`-environment, `100`-step, seed-11 run emitted `quality_status=unassessed` with exact
protocol mismatches `num_envs_per_seed`, `seeds`, and `steps`.

The existing one-update checkpoint was then evaluated with the complete formal protocol. The run
covered `263` nominal, `509` left-knee, and `508` right-knee rows with exact pairing. It exited `2`
with `quality_status=failed`; the failed check was `left_knee.max_lateral_reduction`, observed
`0.094565` against required `0.10`. This is expected rejection evidence, not teacher-quality failure
analysis and not a training result.

Artifacts:

```text
/private/tmp/fada-context-phase1-paired-v002-nonformal-sentinel-20260810.json
sha256=269fadf9c0a9f8eb69f47f07adb66bd07af2dcaa1bd723f0b2489f36d141b494

/private/tmp/fada-context-phase1-formal-gate-one-update-20260810.json
sha256=de3365650d2cb940e460a363e3f6a7e5e79e9d0b826b35ecaf5691400e06c8dc
```

## Verification

- Formal protocol unit/contract suite: `9 passed`.
- Final focused runtime/config suite after launch-path enforcement: `19 passed`.
- Final related off-policy/HORA/G1 suite: `107 passed`.
- Expanded prior suite plus new v002 tests: `340 passed`; four permission-sensitive shared-memory and
  socket cases passed when rerun with those capabilities.
- Ruff: passed.
- Pyright: `0 errors, 0 warnings, 0 informations`.
- `git diff --check`: passed after final document update.

Gymnasium emitted its existing float-bound cast warnings during MuJoCo environment construction. No
preflight identity, dimension, quality metric, or JSON value became non-finite.

## Remaining live boundary

Formal training has not started. No collector process, formal run directory, formal log, or formal
teacher checkpoint was created. CPU preflight proves structural readiness only; the selected target
accelerator must pass the same preflight before the human separately authorizes `runner.learn`.
