# Calibration Stage Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `code-construction` with
> `superpowers:test-driven-development`; run an independent `code-review-expert`
> plan gate before production edits and a final gate after the complete diff.

**Status:** current implementation plan; supersedes the execution boundary of
`2026-08-19-calibratable-tracker-three-stage.md` without changing its scientific semantics.

**Goal:** Make Stage 1, Stage 2, and Stage 3 independently invokable, strictly
ordered training transactions with stage-owned artifacts, while preserving the
serial three-stage method, frozen Planner/Tracker, three-axis catalog, H=30,
K=6, D=128, and first-action-only deployment.

**Architecture:** The training owner exposes three public use cases. Stage 1
creates and seals only a normalized Direction Bank. Stage 2 strictly loads the
Stage 1 artifact, creates and trains only the Coefficient Encoder, and seals the
two frozen owners. Stage 3 strictly loads the Stage 2 artifact plus typed scale
evidence, constructs no optimizer, and publishes the existing deployment
artifact. The existing serial CLI becomes a thin composition root that invokes
the same three public use cases through their persisted boundaries.

**Tech stack:** Python 3.11, PyTorch, argparse, pytest, Ruff, mypy, `uv run`.

---

## Engineering boundary record

Requested behavior:

- provide independent Stage 1, Stage 2, and Stage 3 commands;
- allow Stage 1 to run without Stage 2/3 inputs;
- prevent Stage 2 or Stage 3 from skipping their predecessor;
- preserve the existing serial entrypoint as a convenience composition route;
- run a maintainability review after the coherent diff.

Preserved behavior:

- active v007/v006 Contracts and gain/delay/offset axis order;
- exactly one mutable owner per stage;
- Stage 1 compensation ratio, Stage 2 coefficient error, and Stage 3 monotone
  fit gates;
- dataset, split, source Tracker, catalog, architecture and tensor-finiteness
  admission before optimizer construction or state mutation;
- atomic artifact publication, frozen Planner/Tracker, and final deployment
  artifact schema.

Forbidden shortcuts:

- no simulator collection, long training, playback, policy-quality claim or Git write;
- no test-only production hook or alternate loss/target/optimizer;
- no random Coefficient Encoder state in a Stage 1 artifact;
- no embedded duplicate Planner/Tracker state in stage artifacts;
- no in-memory handoff in the serial wrapper that bypasses strict save/load;
- no legacy fallback that silently loads the pre-isolation stage-checkpoint schema.

Artifact graph:

```text
source Tracker + labeled dataset
  -> Stage 1 -> stage1_direction_frozen.pt
  -> Stage 2 -> stage2_coefficient_frozen.pt
scale_evidence.pt + stage2_coefficient_frozen.pt
  -> Stage 3 -> calibration_artifact.pt
```

`stage1_direction_frozen.pt` owns Direction Bank state and per-axis validation
ratios. `stage2_coefficient_frozen.pt` owns the frozen Direction Bank,
Coefficient Encoder, worst validation error and the exact SHA256 of its Stage 1
parent. The final artifact binds the Stage 2 parent SHA256 and Scale Evidence
SHA256 and remains the only deployment input.

## Task 1: Pin the stage contracts with RED tests

**Files:**

- Modify: `tests/algos/test_fada_calibration_training.py`
- Modify: `tests/scripts/test_fada_calibration_entrypoints.py`

- [ ] Add a Stage 1 public-boundary test that calls the wished-for API without
  scale evidence, poisons `CoefficientEncoder` construction, and expects a
  `direction_frozen` artifact containing normalized directions and no encoder
  state.
- [ ] Add a Stage 2 test that supplies a real Stage 1 artifact, proves its
  Direction Bank and source policy remain bitwise frozen, and expects a
  `coefficient_frozen` artifact.
- [ ] Add a differential that executes Stage 2 from a copied Stage 1 artifact
  with identical bytes, then flips one byte in the parent and proves parent
  identity rejection before optimizer construction.
- [ ] Add Stage 2 negative cases for wrong schema, wrong stage, wrong source,
  wrong dataset/split/catalog, malformed/non-finite direction state and
  architecture mismatch; prove rejection occurs before `torch.optim.Adam`.
- [ ] Add a Stage 3 test that supplies a real Stage 2 artifact and typed scale
  evidence, poisons `torch.optim.Adam`, and expects the deployable calibration
  artifact.
- [ ] Add Stage 3 negative cases for a Stage 1 input, wrong identities,
  malformed/non-finite encoder state and wrong scale evidence; prove no output
  is published.
- [ ] Extend the parser characterization test to require these scripts:

```text
train_fada_calibration_stage1.py
train_fada_calibration_stage2.py
train_fada_calibration_stage3.py
```

- [ ] Run the new nodes and record RED caused only by missing public APIs and
  entrypoints:

```bash
uv run pytest tests/algos/test_fada_calibration_training.py \
  tests/scripts/test_fada_calibration_entrypoints.py -q
```

## Task 2: Introduce stage-owned typed persistence

**Files:**

- Modify: `src/unilab/algos/torch/fada_context/calibration_training.py`
- Modify: `src/unilab/algos/torch/fada_context/__init__.py`

- [ ] Add immutable `CalibrationStageIdentity` with
  `source_tracker_sha256`, `dataset_sha256`, `split_sha256`, and
  `axis_catalog_version`; validate all fields before reading mutable owners.
- [ ] Replace the previous general training-checkpoint writer/loader with one
  discriminated `unilab_fada_calibration_stage_artifact_v2` envelope. Every
  payload binds Method and Training Contract IDs, exact stage, H/K/D,
  architecture, axis catalog/version/order, transaction identity and gate
  metrics.
- [ ] Its `direction_frozen`
  payload contains only architecture, axis names, Direction Bank state,
  compensation ratios and identity.
- [ ] Its `coefficient_frozen` payload contains the admitted Direction Bank,
  Coefficient Encoder configuration/state, coefficient error, the same
  identity and the exact Stage 1 parent artifact SHA256.
- [ ] Keep stage writers private to their owning transaction. Public callers
  may load and inspect a typed frozen result, but cannot label arbitrary module
  state as `direction_frozen` or `coefficient_frozen`.
- [ ] Validate the complete tensor tree, exact stage, architecture, axis order,
  normalized directions, stored gate threshold/result, expected identity and
  parent digest before constructing an optimizer or mutating a caller-owned
  module.
- [ ] Publish through a same-directory uniquely named temporary file. Use
  `try/finally` to remove it after every exception and atomically replace the
  target only after the complete payload is sealed. A failed write must leave
  an existing target byte-for-byte unchanged and must not expose a new target.
- [ ] Remove the old writer/loader from `fada_context.__init__` and production
  callers. The v1 checkpoint schema remains only as an explicit fail-closed
  negative case; no compatibility coercion is allowed.

The intended public boundary is:

```python
@dataclass(frozen=True)
class CalibrationStageIdentity:
    source_tracker_sha256: str
    dataset_sha256: str
    split_sha256: str
    axis_catalog_version: str

@dataclass(frozen=True)
class DirectionStageConfig:
    steps_per_axis: int = 100
    learning_rate: float = 3e-4
    compensation_ratio_threshold: float = 0.1
    training_split_id: int = 0
    validation_split_id: int = 1

@dataclass(frozen=True)
class CoefficientStageConfig:
    steps: int = 1000
    learning_rate: float = 3e-4
    coefficient_error_threshold: float = 0.05
    training_split_id: int = 0
    validation_split_id: int = 1

@dataclass(frozen=True)
class DirectionStageResult:
    stage: Literal["direction_frozen"]
    artifact_path: Path
    artifact_sha256: str
    compensation_ratios: tuple[float, float, float]

@dataclass(frozen=True)
class CoefficientStageResult:
    stage: Literal["coefficient_frozen"]
    artifact_path: Path
    artifact_sha256: str
    parent_stage_sha256: str
    coefficient_error: float

@dataclass(frozen=True)
class ScaleStageResult:
    stage: Literal["complete"]
    artifact_path: Path
    artifact_sha256: str
    parent_stage_sha256: str
    scale_evidence_sha256: str

def run_direction_stage_training(
    policy: FADAPlannerIDMPolicy,
    batch: CalibrationRolloutBatch,
    *,
    output_path: str | Path,
    identity: CalibrationStageIdentity,
    config: DirectionStageConfig,
) -> DirectionStageResult

def run_coefficient_stage_training(
    policy: FADAPlannerIDMPolicy,
    batch: CalibrationRolloutBatch,
    *,
    direction_artifact_path: str | Path,
    output_path: str | Path,
    identity: CalibrationStageIdentity,
    config: CoefficientStageConfig,
) -> CoefficientStageResult

def run_scale_stage_fitting(
    policy: FADAPlannerIDMPolicy,
    *,
    coefficient_artifact_path: str | Path,
    scale_evidence_path: str | Path,
    output_path: str | Path,
    identity: CalibrationStageIdentity,
) -> ScaleStageResult
```

## Task 3: Extract the three owner-level transactions

**Files:**

- Modify: `src/unilab/algos/torch/fada_context/calibration_training.py`
- Modify: `scripts/prepare_fada_calibration_scale_evidence.py`
- Modify: `tests/algos/test_fada_calibration_training.py`
- Modify: `tests/scripts/test_fada_calibration_entrypoints.py`

- [ ] Move the existing Direction Bank loop into
  `run_direction_stage_training`. It validates only Stage 1 inputs, constructs
  no Coefficient Encoder, records all three ratios, freezes the bank and
  publishes `direction_frozen` only after every axis passes.
- [ ] Replace cross-stage parameter ownership with `DirectionStageConfig` and
  `CoefficientStageConfig`. `SerialCalibrationConfig` belongs only to the
  compatibility composition root and projects into those two immutable stage
  configs; no Stage 1 validation may read a Stage 2 field and vice versa.
- [ ] Move the existing Coefficient Encoder loop into
  `run_coefficient_stage_training`. It strictly loads Stage 1, constructs the
  encoder only after admission, preserves the exact loss and publishes only
  after worst normalized error passes.
- [ ] Move the scale fit into `run_scale_stage_fitting`. It strictly loads Stage
  2 and a typed scale-evidence path, computes SHA256 from the exact bytes it
  then deserializes and validates, constructs no optimizer, fits the unchanged
  PCHIP curves and publishes the existing deployment artifact. A caller cannot
  supply a separate claimed digest.
- [ ] Extend `CalibrationScaleEvidence` with the requested coefficient scan grid
  `[3,21]`; require every row to equal the declared `[-1,1]` 21-point grid,
  preserve measured Encoder readings separately as `[3,21,32]`, and bind the
  typed evidence artifact digest into Stage 3 output provenance.
- [ ] Refactor `run_serial_calibration_training` into a thin composition root
  that calls Stage 1, then reloads its artifact for Stage 2, then reloads Stage
  2 for Stage 3. Preserve its public arguments and result fields where they
  remain meaningful.
- [ ] Keep loss, oracle, normalization and frozen-owner helpers single-owned;
  do not duplicate stage semantics in scripts.

## Task 4: Add independent official CLIs

**Files:**

- Create: `scripts/train_fada_calibration_stage1.py`
- Create: `scripts/train_fada_calibration_stage2.py`
- Create: `scripts/train_fada_calibration_stage3.py`
- Modify: `scripts/train_fada_calibration.py`

- [ ] Stage 1 CLI accepts only source checkpoint, dataset, catalog, output,
  Stage 1 steps and learning rate.
- [ ] Stage 2 CLI additionally requires `--stage1-artifact`, accepts Stage 2
  steps, and has no scale-evidence argument.
- [ ] Stage 3 CLI requires `--stage2-artifact`, `--scale-evidence` and output;
  it exposes no learning rate or optimizer steps.
- [ ] Keep scripts as composition roots: load paths, compute file digests and
  call the public training owner. No loss, gate, state-dict interpretation or
  schema branching belongs in `scripts/`.
- [ ] Preserve `train_fada_calibration.py` as the all-stage convenience command.

Expected server-facing command shapes:

```bash
uv run python scripts/train_fada_calibration_stage1.py \
  --source-checkpoint SOURCE.pt --dataset DATASET.pt \
  --output STAGE1.pt --stage1-steps 100 --learning-rate 3e-4

uv run python scripts/train_fada_calibration_stage2.py \
  --source-checkpoint SOURCE.pt --dataset DATASET.pt \
  --stage1-artifact STAGE1.pt --output STAGE2.pt \
  --stage2-steps 1000 --learning-rate 3e-4

uv run python scripts/train_fada_calibration_stage3.py \
  --source-checkpoint SOURCE.pt --dataset DATASET.pt \
  --stage2-artifact STAGE2.pt --scale-evidence SCALE.pt \
  --output CALIBRATION_ARTIFACT.pt
```

## Task 5: Prove compatibility and lifecycle closure

**Files:**

- Modify: `tests/algos/test_fada_calibration_training.py`
- Modify: `tests/scripts/test_fada_calibration_entrypoints.py`

- [ ] Turn every new RED node GREEN without weakening its oracle.
- [ ] Preserve the existing real-owner serial transaction and assert its Stage
  1/2 artifacts can be loaded independently and its final artifact is accepted
  by deployment loading.
- [ ] Prove failed publication leaves no target file and no borrowed owner
  mutation.
- [ ] Inject a serialization failure with no pre-existing target and with a
  byte-labeled pre-existing target; in both cases require zero temporary files,
  and in the latter require the old bytes to remain exact.
- [ ] Prove Stage 1 does not inspect Stage 3 evidence, Stage 2 cannot create
  curves, and Stage 3 cannot construct an optimizer.
- [ ] Mutate the Scale Evidence file after one successful run and prove a later
  Stage 3 result either binds the new byte digest or rejects the changed typed
  content; it may never preserve the old digest while consuming new bytes.
- [ ] Prove serial execution and three independent processes produce equivalent
  stage schemas, identities and final Actions after fresh reload; direct
  in-memory owner forwarding is forbidden.
- [ ] Run focused, affected and static checks:

```bash
uv run pytest tests/algos/test_fada_calibration_training.py \
  tests/scripts/test_fada_calibration_entrypoints.py -q
uv run pytest tests/algos/test_fada_calibration.py \
  tests/algos/test_fada_calibration_training.py \
  tests/algos/test_fada_calibration_evaluation.py \
  tests/scripts/test_fada_calibration_entrypoints.py -q
uv run ruff check \
  src/unilab/algos/torch/fada_context/calibration_training.py \
  src/unilab/algos/torch/fada_context/__init__.py \
  scripts/train_fada_calibration.py \
  scripts/train_fada_calibration_stage1.py \
  scripts/train_fada_calibration_stage2.py \
  scripts/train_fada_calibration_stage3.py \
  tests/algos/test_fada_calibration_training.py \
  tests/scripts/test_fada_calibration_entrypoints.py
uv run ruff format --check \
  src/unilab/algos/torch/fada_context/calibration_training.py \
  src/unilab/algos/torch/fada_context/__init__.py \
  scripts/train_fada_calibration.py \
  scripts/train_fada_calibration_stage1.py \
  scripts/train_fada_calibration_stage2.py \
  scripts/train_fada_calibration_stage3.py \
  tests/algos/test_fada_calibration_training.py \
  tests/scripts/test_fada_calibration_entrypoints.py
uv run mypy src/unilab/algos/torch/fada_context/calibration_training.py
```

## Task 6: Independent review and governance closeout

**Files:**

- Create: `note/fada/reviews/2026-08-20-calibration-stage-isolation-plan.json`
- Create: `note/fada/reviews/2026-08-20-calibration-stage-isolation-final.json`
- Update: `note/testing/module_test_manifest.json`
- Update: `note/fada/task_canvas.md`
- Update: `note/fada/checklists/current.md`
- Update: `note/governance.json`
- Update only if source links changed: `note/architecture/atlas_manifest.json`

- [ ] Obtain a validated `code-review-expert` plan `READY` receipt before
  production edits using `standard`, `module-boundary`,
  `repository-discipline`, and `research-ml`.
- [ ] Run impacted-set Module Alignment for stage owner, persistence, negative,
  identity and composition cases; refresh checkout/content identity.
- [ ] Run an independent `code-review-expert` final gate over the coherent diff.
  Repair only same-scope P0/P1 findings under the current authorization, then
  re-review.
- [ ] Update governance to the accurate post-change state. Formal runtime,
  simulator execution and policy quality remain not run.

## Stop condition

Complete only when all three stage CLIs expose real parsers, every stage can run
without future-stage inputs, every later stage strictly consumes the preceding
artifact, the serial wrapper crosses the same persistence boundaries, affected
tests and static checks pass, Module Alignment is current, and the independent
final review has no open P0/P1. A simulator collector and actual server command
execution remain separate work.
