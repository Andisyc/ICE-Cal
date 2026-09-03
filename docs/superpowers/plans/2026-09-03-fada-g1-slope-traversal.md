# FADA G1 15-Degree Slope Traversal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic 15-degree narrow-slope target domain, collect target-only FADA rollouts across bounded episodes, adapt IDM LoRA, and compare zero-shot against adapted straight-line traversal under identical conditions.

**Architecture:** Hydra selects a typed target-domain owner. Existing paired actuator-gain collection moves behind a compatibility owner, while slope collection uses a target-only multi-episode workflow. A separate evaluator owns same-condition before/after videos and ramp-coordinate metrics.

**Tech Stack:** Python 3.13, Hydra/OmegaConf, PyTorch, NumPy, MuJoCo XML, pytest, Ruff, Pyright, `uv run`.

**Spec:** `docs/superpowers/specs/2026-09-03-fada-g1-slope-traversal-design.md`

## Global Constraints

- Always invoke Python tools through `uv run --frozen --no-sync`.
- Do not create or switch branches without explicit user approval.
- Preserve all pre-existing dirty worktree changes and artifact files.
- Do not run real MuJoCo collection, LoRA training, or policy evaluation without separate user authorization.
- Keep scripts as Hydra entrypoints; business rules stay under `src/unilab/algos/torch/distill/fada/`.
- Keep the keyframe in task-level XML and leave `g1.xml` unchanged.
- Keep checkpoint policy I/O at raw actor width 98 and projected student width 66.
- Do not add action clipping, clamping, min/max, tanh, or slope-specific control correction.
- Slope mode rejects randomization, observation noise, pushes, latency randomization, and actuator faults.
- LoRA freezes Planner and base IDM and trains only existing IDM Q/V attention adapters with rank 8, alpha 16, and dropout 0.05.
- Before editing a dirty file, capture its current diff. Stage only task-owned hunks and never stage experiment artifacts.

## File Map

New owner modules:

- `target_domain.py`: typed target conditions, legacy conversion, slope geometry.
- `target_actuator_workflow.py`: current nominal/faulty actuator collection, moved without behavior changes.
- `target_slope_workflow.py`: target-only slope collection and publication.
- `slope_metrics.py`: ramp-frame metrics.
- `target_evaluation.py`: identical-condition zero-shot/adapted evaluation.

New configuration and entrypoints:

- `src/unilab/assets/robots/g1/scene_slope_15.xml`
- `conf/offpolicy/target_domain/slope_15.yaml`
- `conf/offpolicy/task/sac/g1_walk_flat/mujoco_fada_slope_15.yaml`
- `conf/offpolicy/evaluation/fada_slope.yaml`
- `conf/offpolicy/fada_evaluate.yaml`
- `scripts/evaluate_fada_target.py`

Existing owners modified in place:

- `src/unilab/envs/locomotion/g1/base.py`
- `src/unilab/algos/torch/distill/fada/target_collector.py`
- `src/unilab/algos/torch/distill/fada/target_data.py`
- `src/unilab/algos/torch/distill/fada/target_workflow.py`
- `src/unilab/algos/torch/distill/fada_adaptation.py`
- `src/unilab/algos/torch/distill/fada/adaptation_checkpoint.py`
- Hydra roots and focused tests listed by each task below.

## Task 1: Typed Target-Domain Contract

**Files:** Create `target_domain.py` and `tests/algos/test_fada_target_domain.py`; modify the distill package facade.

**Public contract:**

```python
@dataclass(frozen=True)
class FADASlopeGeometry:
    angle_deg: float
    width_m: float
    approach_length_m: float
    surface_length_m: float
    entry_margin_m: float
    finish_margin_m: float

    def surface_coordinates(self, positions_w: np.ndarray) -> np.ndarray: ...
    def has_entered(self, base_pos_w: np.ndarray, feet_pos_w: np.ndarray) -> bool: ...
    def has_finished(self, base_pos_w: np.ndarray) -> bool: ...
    def foot_exited(self, feet_pos_w: np.ndarray) -> bool: ...

@dataclass(frozen=True)
class FADATargetDomainSpec:
    target_domain_id: str
    kind: Literal["slope", "actuator_gain"]
    task: str
    task_name: str
    backend: str
    command_sequence: tuple[tuple[float, float, float], ...]
    slope: FADASlopeGeometry | None = None
    actuator_index: int | None = None
    actuator_strength: float | None = None
    actuator_count: int | None = None

def resolve_fada_target_domain(cfg: DictConfig) -> FADATargetDomainSpec: ...
```

- [ ] Write failing tests for exact slope values and command sequence `0.75, 0.80, 0.85 m/s` with zero lateral/yaw command.
- [ ] Write failing tests for missing geometry, mixed slope/actuator fields, unsupported kinds, and simultaneous `target_domain` plus legacy `fault` mappings.
- [ ] Write failing geometry tests: entry requires pelvis `s>=0.25` and both feet `s>=0`; finish is `s>=7.5`; either foot at `abs(y)>0.4` exits.
- [ ] Run RED: `uv run --frozen --no-sync pytest -q tests/algos/test_fada_target_domain.py`.
- [ ] Implement immutable specs, finite/range validation, vectorized world-to-ramp coordinates, and an explicit legacy `cfg.fault -> actuator_gain` conversion only when `target_domain` is absent.
- [ ] Export the three public symbols and rerun the focused test plus Ruff.
- [ ] Commit the new owner, facade hunk, and test as `feat: add typed FADA target domains`.

## Task 2: Deterministic Slope Scene and Task State

**Files:** Create `scene_slope_15.xml` and `mujoco_fada_slope_15.yaml`; modify `src/unilab/envs/locomotion/g1/base.py`, `tests/envs/test_env_configs.py`, and config tests.

**Task boundary:**

```python
def get_foot_pos(self) -> np.ndarray:
    """Return world positions shaped (num_envs, 2, 3), left then right."""
    left = np.asarray(self._backend.get_sensor_data("left_foot_pos"))
    right = np.asarray(self._backend.get_sensor_data("right_foot_pos"))
    return np.stack((left, right), axis=1)
```

The scene retains `scene_flat.xml` assets, sensors, actuators, visuals, and `stand` keyframe, but replaces the infinite floor with:

```xml
<geom name="approach" type="box" pos="0.25 0 -0.05"
      size="1.25 0.4 0.05" material="groundplane" contype="1" conaffinity="1"/>
<geom name="slope_15" type="box" pos="5.376 0 0.987"
      size="4.0 0.4 0.05" euler="0 -15 0"
      material="groundplane" contype="1" conaffinity="1"/>
```

- [ ] Write a failing config test asserting `G1WalkFlat`, MuJoCo, slope scene path, 98-D raw observation compatibility, and 29 actions.
- [ ] Write a failing XML test that loads through MuJoCo and derives 15 degrees, 0.8 m width, 8.0 m surface length, a ramp entry at `x=1.5`, and flat support extending from `x=-1.0` to the entry.
- [ ] Write a failing fake-backend test for `get_foot_pos()` shape/order and malformed sensor shapes.
- [ ] Run RED with `uv run --frozen --no-sync pytest -q tests/envs/test_env_configs.py tests/scripts/test_collect_fada_target.py -k 'slope or foot_pos'`.
- [ ] Create the task-level XML without touching `g1.xml`; implement the task accessor using declared sensor data only.
- [ ] Create a Hydra task owner inheriting `/task/sac/g1_walk_flat/mujoco_fada_target`, overriding only the scene path, and preserving task/checkpoint identity.
- [ ] Rerun tests and cold-load XML with `uv run --frozen --no-sync python -c 'import mujoco; mujoco.MjModel.from_xml_path("src/unilab/assets/robots/g1/scene_slope_15.xml")'`.
- [ ] Commit Task 2 as `feat: add deterministic G1 slope scene`.

## Task 3: Slope Hydra Defaults and Preflight

**Files:** Create `conf/offpolicy/target_domain/slope_15.yaml`; modify `fada_target.yaml`, `fada_adapt.yaml`, collection/adaptation defaults, `target_workflow.py`, and their script tests.

**Target config:**

```yaml
defaults:
  - /task: sac/g1_walk_flat/mujoco_fada_slope_15
  - _self_
target_domain:
  target_domain_id: g1_slope_15_mujoco
  kind: slope
  task: sac/g1_walk_flat/mujoco_fada_slope_15
  task_name: G1WalkFlat
  backend: mujoco
  command_sequence: [[0.75, 0.0, 0.0], [0.80, 0.0, 0.0], [0.85, 0.0, 0.0]]
  slope:
    angle_deg: 15.0
    width_m: 0.8
    approach_length_m: 1.5
    surface_length_m: 8.0
    entry_margin_m: 0.25
    finish_margin_m: 0.5
```

Collection defaults are `control_steps=6000`, `max_env_steps=24000`, `ramp_steps=25`, `settle_steps=50`, `seed=1`, and `output_dir=artifacts/fada_target/${target_domain.target_domain_id}`.

- [ ] Write RED composition tests proving slope is the Stage C/D default, Stage D reads `target.pt`, and output names use the target-domain ID.
- [ ] Write RED preflight tests enabling noise, each randomization family, pushes, latency, and non-identity actuator multipliers one at a time. Assert failure occurs before environment construction and names the dotted field.
- [ ] Write RED identity tests for wrong task, backend, scene, or policy observation contract.
- [ ] Implement `assert_nominal_slope_environment(cfg, domain)` and call it before checkpoint loading/environment creation. Missing SHA means no digest comparison; a supplied SHA remains strict.
- [ ] Remove slope dependencies on `${fault.*}` while keeping the legacy conversion in the typed owner.
- [ ] Rerun the two script-test files and a Hydra compose smoke test.
- [ ] Commit Task 3 as `feat: configure slope target collection`, staging only new hunks from currently dirty config/workflow files.

## Task 4: Multi-Episode Target-Only Collector

**Files:** Modify `target_collector.py` and `tests/algos/test_fada_target_collector.py`.

**Collector protocol:**

```python
@dataclass(frozen=True)
class FADATargetStepDecision:
    accept: bool
    terminal_reason: str | None

class FADATargetEpisodePolicy(Protocol):
    def command_for_episode(self, episode_id: int) -> np.ndarray: ...
    def classify(self, *, base_pos_w: np.ndarray, feet_pos_w: np.ndarray,
                 done: bool) -> FADATargetStepDecision: ...
```

Extend the result with `accepted_steps`, `episode_count`, `rejected_pre_entry_steps`, `termination_counts`, and `representative_physics_states`.

- [ ] Write a scripted multi-episode RED test that enters, collects, terminates, resets, and re-enters. No causal window may cross episode ID/timestep boundaries.
- [ ] Add RED cases for pre-entry exclusion, command cycling, reset history clearing, partial-window discard, every terminal reason, longest-episode video selection, and bounded 24,000-step exhaustion diagnostics.
- [ ] Preserve characterization coverage for the legacy actuator single-trajectory lifecycle.
- [ ] Refactor the loop to reset controller/history at each boundary and classify before accepting a transition. Never join raw records across resets.
- [ ] Retain startup frames for representative video context while excluding them from `target.pt`.
- [ ] Run the full collector test, Ruff, and diff check.
- [ ] Commit Task 4 as `feat: collect slope rollouts across episodes`, staging only the collector/test hunks.

## Task 5: Target Artifact v3 and Stage D Migration

**Files:** Modify `target_data.py`, `fada_adaptation.py`, `adaptation_checkpoint.py`, and the three associated test files.

```python
FADA_TARGET_ARTIFACT_SCHEMA_VERSION = "fada-target-batch/v3"
FADA_LEGACY_TARGET_ARTIFACT_SCHEMA_VERSION = "fada-target-batch/v2"
```

V3 requires `target_domain_id`, `target_domain_kind`, task, source checkpoint digest, observation contract, command sequence, episode count, termination counts, and `randomization_disabled=True`. V2 continues to require `fault_profile`.

- [ ] Write RED round-trip tests for valid v3 and rejection of missing/unknown/mixed metadata.
- [ ] Write RED tests proving neither v2 nor v3 infers identity from a path.
- [ ] Write RED Stage D tests loading one v2 knee artifact and one v3 slope artifact through the same split/LoRA boundary.
- [ ] Write RED checkpoint tests requiring target schema, digest, and target-domain ID in newly adapted checkpoints.
- [ ] Implement version-dispatched metadata validation with shared tensor validation and explicit `source_schema_version` in the loaded representation.
- [ ] Compare v3 `target_domain_id` against config; compare v2 `fault_profile` only through the legacy actuator boundary.
- [ ] Record v3 lineage without changing policy tensor/checkpoint architecture schema.
- [ ] Run `test_fada_target_data.py`, `test_adapt_fada_target.py`, and `test_fada_adaptation_checkpoint.py` plus Ruff.
- [ ] Commit Task 5 as `feat: add target-domain artifact schema`, staging only task-owned hunks.

## Task 6: Separate Actuator and Slope Stage C Owners

**Files:** Create `target_actuator_workflow.py` and `target_slope_workflow.py`; reduce `target_workflow.py`; modify collection script tests.

```python
def run_fada_actuator_collection(cfg, *, preflight, dependencies) -> dict[str, Any]: ...
def run_fada_slope_collection(cfg, *, preflight, dependencies) -> dict[str, Any]: ...
```

- [ ] Pin current actuator outputs with characterization tests: `nominal.pt`, `faulty.pt`, `delta.pt`, both videos, and path metrics remain unchanged.
- [ ] Write RED slope tests requiring exactly `target.pt`, `collection.mp4`, `collection_summary.json`, and `manifest.json`; forbid nominal/faulty/delta/excess outputs.
- [ ] Inject collection, save, render, summary, and manifest failures separately; each RED test must prove the final bundle is unpublished.
- [ ] Move current paired behavior into the actuator owner without semantic edits.
- [ ] Implement the slope owner with multi-episode collection, v3 save, representative-video rendering through the existing MuJoCo playback owner, and transactional publication.
- [ ] Leave `target_workflow.py` owning shared path/checkpoint/dependency preflight and strict two-kind dispatch only.
- [ ] Run the complete Stage C script tests, Ruff, Pyright on these three owners, and diff check.
- [ ] Commit Task 6 as `refactor: separate FADA target workflows`.

## Task 7: Ramp Metrics and Same-Condition Evaluation

**Files:** Create `slope_metrics.py`, `target_evaluation.py`, evaluation configs/CLI, and their two test files; export the owner API.

```python
@dataclass(frozen=True)
class FADASlopeTrajectory:
    base_pos_w: np.ndarray
    base_yaw_rad: np.ndarray
    feet_pos_w: np.ndarray
    forward_velocity_mps: np.ndarray
    command_forward_mps: np.ndarray
    physics_states: tuple[np.ndarray, ...]
    terminal_reason: str

def summarize_slope_trajectory(trajectory, geometry) -> dict[str, float | bool | str]: ...
def compare_slope_summaries(zero_shot, adapted) -> dict[str, float]: ...
def run_fada_target_evaluation(cfg: DictConfig) -> dict[str, Any]: ...
```

Evaluation defaults: source `planner_idm_v022_cpu_limited.pt`, adapted `artifacts/fada_adaptation/g1_slope_15_mujoco_v3.pt`, command `[0.8,0,0]`, seed 1, flat regression enabled, videos enabled.

- [ ] Write RED metric tests for straight, left-drift, right-drift, yaw drift, uphill progress, velocity error, fall, finish, and foot exit.
- [ ] Assert error improvement is `zero_shot - adapted`; progress improvement is `adapted - zero_shot`; preserve signed lateral/yaw traces.
- [ ] Write RED workflow tests proving task, seed, reset physics snapshot, command, and horizon are identical for both checkpoints.
- [ ] Write RED publication tests for slope videos/metrics/manifest, optional flat-regression outputs, and refusal to overwrite a run directory.
- [ ] Implement one shared rollout function. Restore the same reset snapshot and reset controller history before each checkpoint.
- [ ] Load both schemas with `load_fada_deployable_policy_checkpoint()` and validate architecture before environment creation.
- [ ] Keep flat regression in a separate result object and out of artifact/adaptation owners.
- [ ] Add a thin Hydra CLI of at most 19 lines.
- [ ] Run both new test files, Ruff, Pyright, and diff check.
- [ ] Commit Task 7 as `feat: evaluate FADA slope adaptation`.

## Task 8: Integrated Verification and Runbook

**Files:** Modify `note/fada/README.md` and `note/fada/contracts/README.md`; create `docs/runbooks/fada-slope-traversal.md`.

- [ ] Document the four distinct boundaries: source training, target collection, LoRA adaptation, and post-adaptation evaluation.
- [ ] Add exactly three complete `uv run --frozen --no-sync` commands for collection, adaptation, and evaluation. Default commands omit hashes.
- [ ] State that automated tests do not constitute simulation, training, or policy-quality evidence.
- [ ] Run focused integration:

```bash
uv run --frozen --no-sync pytest -q \
  tests/algos/test_fada_target_domain.py \
  tests/algos/test_fada_target_collector.py \
  tests/algos/test_fada_target_data.py \
  tests/algos/test_fada_slope_metrics.py \
  tests/algos/test_fada_adaptation.py \
  tests/algos/test_fada_adaptation_checkpoint.py \
  tests/scripts/test_collect_fada_target.py \
  tests/scripts/test_adapt_fada_target.py \
  tests/scripts/test_evaluate_fada_target.py \
  tests/envs/test_env_configs.py
```

- [ ] Run affected FADA regression:

```bash
uv run --frozen --no-sync pytest -q \
  tests/algos/test_fada_*.py \
  tests/scripts/test_collect_fada_target.py \
  tests/scripts/test_adapt_fada_target.py \
  tests/scripts/test_evaluate_fada_target.py
```

- [ ] Run Ruff over changed source/tests, Pyright over the seven new or changed owner modules, and `git diff --check`.
- [ ] Inspect `git status --short`; no experiment artifact, unrelated note, or pre-existing dirty file may be staged.
- [ ] Commit documentation as `docs: add FADA slope traversal runbook`.

## Execution Stop Boundary

Stop after automated verification and report changed owners, exact checks, new versus pre-existing static-analysis failures, the three runbook commands, and untouched user-owned files.

Do not launch Stage C, Stage D, MuJoCo evaluation, upload, push, or deployment. Each is a separate human-authorized runtime gate.
