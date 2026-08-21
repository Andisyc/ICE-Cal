# Gain Calibration Rollout Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `code-construction` with
> `superpowers:test-driven-development`; require an independent `code-review-expert`
> READY review before production edits and a final gate after the coherent diff.

**Status:** accepted gain-only smoke implementation plan.

**Goal:** Add the missing official MuJoCo collection path that records an
identity-bound gain-only `raw_rollouts.pt` and seals it into the existing v007/v006
calibration dataset schema.

**Architecture:** G1 owns the action-execution fault after recording the nominal
action. A focused calibration collector owns real H=30 history, frozen Planner-IDM
queries, rollout provenance and raw persistence. A thin Hydra entrypoint owns only
composition of the existing task, checkpoint, environment and collector.

**Tech stack:** Python 3.11, NumPy, PyTorch, Hydra/OmegaConf, pytest, Ruff, mypy,
`uv run`.

---

## Engineering boundary record

Requested behavior:

- collect only the approved gain smoke grid:
  `c_true={-1,0,+1} <-> gain={0.8,1.0,1.2}`;
- inject gain inside the G1 action-execution owner while preserving nominal
  `current_actions` and recording faulted `executed_actions`;
- record real 30-frame State/nominal-Action histories, current Command, Planner
  Intent, the complete K=6 nominal Action chunk, first-action execution and
  rollout/split/seed identity;
- atomically publish a typed raw artifact and let the existing dataset owner seal
  `calibration_dataset.pt`.

Preserved behavior:

- active `FADA-CONTEXT-METHOD-v007` and `FADA-CONTEXT-TRAIN-v006` semantics;
- Planner and Tracker remain frozen, K=6 remains complete, and only index zero is
  sent to the environment;
- existing G1 behavior is bitwise unchanged when the fault mode is disabled or
  gain is exactly 1.0;
- action authority is applied before the gain fault, so a gated zero action stays
  zero;
- actor observations and collector Action history expose nominal commanded
  Actions, never the hidden faulted Action;
- faulted first Actions are persisted only as diagnostic raw evidence and never
  enter `CalibrationRolloutBatch` or a deployable model input;
- train and validation identities are split by complete rollout/seed, never by
  individual windows;
- the existing three-axis Stage 1 admission remains fail-closed.

Authorized smoke protocol:

```text
task: G1WalkFlat / MuJoCo
command: [0.4, 0.0, 0.0]
points: (-1.0, 0.8), (0.0, 1.0), (1.0, 1.2)
train seed: 101
validation seed: 201
accepted rows per point and seed: 32
maximum environment steps per point and seed: 512
```

The protocol is configuration-owned and versioned independently from the unresolved
delay/offset physical mappings. The source checkpoint path and output path remain
runtime inputs and are bound by SHA256 in the raw artifact.

Forbidden shortcuts:

- no collector-side `env.step(faulted_action)` and no backend-private access;
- no reuse of Kp/Kd actuator-strength randomization as action execution gain;
- no synthetic zero-padded history row admitted as training data;
- no inference of delay, offset, held-out combination or unapproved gain values;
- no silent command change, episode crossing, non-finite policy output or partial
  artifact publication;
- no simulator, long training, Stage 1 publication, policy-quality claim or Git
  write in this offline construction unit.

Evidence boundary:

```text
offline construction:
  gain transform -> collector pseudo-env -> raw persistence -> dataset sealing

human-run server transition:
  real G1WalkFlat/MuJoCo -> raw_rollouts.pt -> calibration_dataset.pt

not claimed:
  three-axis Stage 1, learned correction quality, live stability, deployment
```

## Task 1: Pin action-execution ownership with RED tests

**Files:**

- Create: `tests/envs/locomotion/g1/test_calibration_action_fault.py`
- Create: `src/unilab/envs/locomotion/g1/calibration_fault.py`
- Modify: `src/unilab/envs/locomotion/g1/joystick.py`

- [ ] Add semantic pseudo-samples proving gain 0.8 and 1.2 scale only
  `executed_actions`, gain 1.0 is exact identity, and disabled mode preserves the
  old path.
- [ ] Prove `current_actions` remains the nominal input, action-authority zeroing
  happens first, and backend/private state is not consulted.
- [ ] Add fail-closed cases for non-finite/non-positive gain, unsupported mode,
  wrong rank and wrong environment batch.
- [ ] Run the focused tests and record RED caused only by the missing public owner.
- [ ] Implement immutable config validation and call the transform from
  `G1WalkEnv.apply_action` after `_actions_for_execution`.
- [ ] Re-run focused tests GREEN plus the neighboring G1 action/observation tests.

## Task 2: Pin collection and raw persistence with RED tests

**Files:**

- Create: `tests/algos/test_fada_calibration_collection.py`
- Create: `src/unilab/algos/torch/fada_context/calibration_collection.py`
- Modify: `src/unilab/algos/torch/fada_context/__init__.py`

- [ ] Add a deterministic pseudo-environment and frozen policy fixture with
  asymmetric State, Action, Intent and K=6 values. Independent oracles must prove
  the collector passes only nominal first Actions and accepts no row until 30 real
  nominal Actions have executed.
- [ ] Prove each accepted row contains `[B,30,O]`, `[B,30,A]`, `[B,6,O]`,
  `[B,6,A]`, gain-only `[B,3]` coefficients, exact point strength, and immutable
  rollout/seed/split identity. Persist the actual faulted first Action as a
  diagnostic-only `[B,A]` field and prove it never reaches the sealed training
  batch.
- [ ] Stage candidate rows in one episode-and-command transaction. Admit rows
  only after 30 real nominal Actions, and commit exactly the requested row quota
  only if the same episode and fixed Command remain valid throughout. A done,
  reset, command drift, malformed shape or non-finite policy output discards all
  pending rows, histories and candidates; the next reset receives a new
  `rollout_id` and repeats the complete warmup.
- [ ] Prove the exact three approved pairs are accepted while reordered,
  additional, rounded or mismatched pairs fail before environment creation.
- [ ] Prove train and validation rollout IDs are disjoint and row permutation does
  not alter rowwise meaning.
- [ ] Prove the raw writer/loader bind schema, Contracts, source checkpoint digest,
  architecture, active axis catalog, exact smoke protocol bytes, resolved
  task/backend configuration digest and fixed Command; corrupt identity or tensor
  trees reject before dataset sealing.
- [ ] Prove atomic save failure preserves an existing target and removes partial
  temporary files.
- [ ] Run the focused tests RED, implement the smallest owner API, then rerun GREEN.

## Task 3: Add the thin official collection composition root

**Files:**

- Create: `conf/fada_context/calibration_collection/gain_smoke_v1.yaml`
- Create: `scripts/collect_fada_calibration_rollouts.py`
- Modify: `scripts/prepare_fada_calibration_dataset.py`
- Modify: `tests/scripts/test_fada_calibration_entrypoints.py`

- [ ] Store the exact approved points, fixed command, split seeds, row quota and
  step limit in the smoke YAML; source/output paths remain explicit CLI inputs.
- [ ] Add a real argparse entrypoint that validates the source checkpoint digest,
  loads the frozen policy, applies the declared seed before each environment,
  constructs `G1WalkFlat/mujoco` through `BackendAdapter` and `create_env`, and
  closes every environment in `finally`.
- [ ] Build each environment with config-owned `action_execution_fault` and fixed
  command limits; do not mutate backend or env private fields.
- [ ] Change dataset preparation to use the typed raw loader and require its source
  digest to match the selected checkpoint before `prepare_calibration_rollout_batch`.
- [ ] Add parser/composition tests proving the script exposes only source,
  expected source SHA256, protocol, output and device inputs while the physical
  mapping remains YAML-owned. Canonicalize the resolved distill task plus base env
  override before adding the per-point gain, and bind its SHA256 into the raw
  artifact.
- [ ] Prove raw-to-dataset gain target equals `nominal_action_chunk / gain` and
  Stage 1 still rejects the gain-only dataset as incomplete for the active
  three-axis publication route.

## Task 4: Verification and independent closeout

- [ ] Run focused owner tests first, then the affected calibration, G1 env and
  entrypoint suites.
- [ ] Run Ruff check/format, mypy for changed production modules and
  `git diff --check`; preserve unrelated dirty files.
- [ ] Run impacted Module Alignment semantic pseudo-samples for the action fault,
  collection lifecycle and raw persistence owners.
- [ ] Run an independent `code-review-expert` final gate with `standard`,
  `module-boundary`, `repository-discipline`, and `research-ml` profiles.
- [ ] Hand back the exact server commands without executing simulator or training
  locally. Stop after `calibration_dataset.pt`; the next scientific decision is
  whether to add a non-publishable gain-only Direction probe or collect the full
  three-axis dataset.
