# FADA Paper-Aligned Source Training v012 Implementation Plan

> Status: Unit A implemented and locally verified; Oracle training, server, Git, and Unit B remain unauthorized.

**Goal:** Replace the v011 distilled-teacher source route with an ICE-Cal-trained privileged Oracle lineage, then train a fresh Planner–IDM source policy whose inputs and causal targets match the FADA paper.

**Architecture:** Two serial engineering units. Unit A owns privileged Oracle observation, reward, domain randomization, and checkpoint lineage. Unit B consumes an admitted Unit A lineage and owns the 98→(66 state, 29 previous action, 3 command) split, action-free future prediction, causal IDM training, and Planner-through-frozen-IDM training. Unit B cannot begin until Unit A has passed formal runtime and policy-quality admission.

**Stack:** Hydra, MuJoCo, UniLab-derived off-policy SAC runtime, PyTorch Planner–IDM modules, pytest. Every Python command uses `uv run`.

## Non-negotiable contract

- One task only: `G1WalkFlat` on MuJoCo. No separate standing task, walk-to-stand scenario, or scenario quota.
- The Oracle actor directly receives the typed privileged bundle defined by `FADA-METHOD-v012`.
- No Gait/feet-phase Reward. Gait phase may remain in observation; every phase-conditioned reward scale must be zero or absent. Preflight fails closed on any non-zero alias.
- One Oracle campaign: iterations `240, 480, …, 4800` are the 20 intermediate checkpoints; iteration `5000` is final; all 21 artifacts share one `oracle_lineage_id`.
- Planner history token is `66 state + 29 previous action = 95`; command is a separate 3-vector; Planner future is `K×66` and contains no action.
- Intermediate Oracle checkpoints supply IDM suboptimal trajectory coverage only. The final Oracle supplies all source action labels and Oracle-shadow pairs.
- Preserve Oracle-shadow, causal future–action pairing, per-round IDM-before-Planner order, Planner gradients through a frozen IDM, first-action supervision, and receding-horizon first-action execution.
- v011 checkpoints and datasets are incompatible historical artifacts; there is no silent migration or fallback.

## Task 0 — Lock failing contract probes

**Owner files:**

- Modify `tests/algos/test_fada_input_contract.py`
- Add `tests/algos/test_fada_v012_oracle_contract.py`
- Modify `tests/algos/test_fada_unified_oracle.py`

**Work:**

1. Add a negative test showing the existing 66-only Planner history violates the 95-dimensional v012 contract.
2. Add a leakage test that changes the supervised future action while keeping the action-free future fixed and proves the Planner input is unchanged.
3. Add a preflight test that rejects `feet_phase != 0` and accepts zero/absent phase reward.
4. Add lineage tests rejecting mixed `oracle_lineage_id`, missing intermediate indices, a non-5000 final checkpoint, and intermediate checkpoints used as Planner labels.
5. Run the focused tests and retain the expected RED output as the implementation baseline.

**Command:**

```bash
uv run --no-sync pytest -q \
  tests/algos/test_fada_input_contract.py \
  tests/algos/test_fada_v012_oracle_contract.py \
  tests/algos/test_fada_unified_oracle.py
```

## Unit A — ICE-Cal privileged Oracle lineage

### Task A1 — Define the typed privileged observation at its owner boundary

**Owner files:**

- Add `src/unilab/algos/torch/distill/fada_privileged_oracle.py`
- Modify `src/unilab/envs/locomotion/g1/joystick.py`
- Modify `src/unilab/base/backend/base.py` only if a required privilege is absent from `SimBackend`
- Modify the MuJoCo backend adapter only for a newly declared abstract capability

**Work:**

1. Introduce immutable field names, dimensions, ordering, and version for the deployable 98-vector plus privileged base linear velocity, binary foot contacts, terrain/root clearance, actuator state, and domain-randomization parameters.
2. Materialize asset-independent metadata during initialization; do not inspect XML/assets or backend-private attributes in `step`/`reset`.
3. Make the Oracle actor consume this full bundle directly. Do not hide privileges in a critic-only branch.
4. Validate exact shape, dtype, finiteness, and field order at the environment→replay and replay→learner boundaries.

### Task A2 — Express the Oracle task, reward, and domain distribution in config

**Owner files:**

- Add `conf/offpolicy/task/sac/g1_walk_flat/mujoco_fada_privileged_oracle.yaml`
- Modify `src/unilab/envs/locomotion/common/domain_rand.py`
- Modify `src/unilab/envs/locomotion/common/dr_provider.py`
- Add focused environment/config tests under `tests/envs/locomotion/g1/`

**Work:**

1. Inherit the single locomotion task and override only Oracle-owned observation, reward, and DR settings.
2. Set gait/feet-phase reward to zero and add an owner-layer alias scan that rejects any non-zero phase-conditioned reward before environment construction.
3. Implement the agreed FADA DR family: friction, base CoM, added base mass, link mass, DoF bias, gains, torque RFI, one-step latency, and external pushes.
4. If a DR operation needs a backend-specific capability, declare it in `SimBackend` first and keep implementation in the adapter.

### Task A3 — Seal one 20+1 checkpoint lineage

**Owner files:**

- Modify `src/unilab/algos/torch/distill/fada_oracle.py`
- Modify `src/unilab/algos/torch/distill/fada_source_plan.py`
- Modify `src/unilab/algos/torch/distill/fada_artifact_admission.py`
- Keep `scripts/train_offpolicy.py` assembly-only

**Work:**

1. Persist `oracle_lineage_id`, task/config hashes, privilege schema, reward fingerprint, DR fingerprint, seed, iteration, and role (`intermediate` or `final`) with each checkpoint.
2. Require exactly 20 intermediate iterations `240…4800` plus final iteration `5000` from one lineage.
3. Reject a distilled policy, a UniLab-external artifact without v012 provenance, mixed runs, or a checkpoint whose reward fingerprint contains non-zero gait reward.
4. Resolve paths only from the admitted manifest; do not infer ordering from filenames alone.

### Task A4 — Prove Unit A before any long run

**Focused verification:**

```bash
uv run --no-sync pytest -q \
  tests/algos/test_fada_v012_oracle_contract.py \
  tests/algos/test_fada_unified_oracle.py \
  tests/envs/locomotion/g1
```

Then perform a separately authorized formal runtime audit proving observation identity, actor privilege consumption, reward rejection, DR activation, CUDA/device identity, checkpoint provenance, and resumability. Long Oracle training requires a new explicit authorization. Its final checkpoint must then pass a separately authorized policy-quality audit before Unit B opens.

## Unit B — Fresh Planner–IDM source campaign

### Task B1 — Replace the observation contract without compatibility ambiguity

**Owner files:**

- Modify `src/unilab/algos/torch/distill/fada_observation.py`
- Modify `src/unilab/algos/torch/distill/fada.py`
- Modify `tests/algos/test_fada_input_contract.py`
- Modify `tests/algos/test_fada_planner_idm.py`

**Work:**

1. Add a v012 observation contract that splits raw 98 as state 66, previous action 29, and command 3.
2. Make Planner accept `H×95` history plus a separate command and anchor its residual output to the latest 66-dimensional state slice only.
3. Keep IDM inputs `H×66`, `H×29`, and `K×66`; output remains `K×29`.
4. Prohibit action fields in the predicted future by construction and by leakage regression test.

### Task B2 — Rebuild collection roles around one Oracle lineage

**Owner files:**

- Modify `src/unilab/algos/torch/distill/fada_collection_contract.py`
- Modify `src/unilab/algos/torch/distill/fada_collector.py`
- Modify `src/unilab/algos/torch/distill/fada_windows.py`
- Modify `src/unilab/algos/torch/distill/fada_source_artifact.py`
- Modify `src/unilab/algos/torch/distill/fada_replay.py`

**Work:**

1. Remove v011 standing/transition source roles and scenario quotas from the active route.
2. Collect intermediate-policy realized trajectories for IDM-only suboptimal coverage; collect final-Oracle trajectory and Oracle-shadow causal pairs.
3. Apply the FADA source balance of two suboptimal rows per optimal row at sampler ownership, not in the top-level script.
4. Persist checkpoint role and lineage per row; reject rows whose Planner-label owner is not the final Oracle.

### Task B3 — Preserve the serial optimizer contract

**Owner files:**

- Modify `src/unilab/algos/torch/distill/fada_trainer.py`
- Modify `src/unilab/algos/torch/distill/fada_training.py`
- Modify `src/unilab/algos/torch/distill/fada_checkpoint.py`
- Modify `tests/algos/test_fada_alternating_training.py`
- Modify `tests/algos/test_fada_persistence.py`

**Work:**

1. In every round, update IDM first, then freeze IDM parameters while retaining gradients through IDM inputs, then update Planner using final-Oracle first-action labels.
2. Assert no IDM parameter changes during Planner optimization and no Planner gradient is detached at the IDM boundary.
3. Write a new checkpoint schema binding dimensions `66/95/29/3`, action-free future, Oracle lineage manifest, dataset fingerprints, and optimizer ownership.
4. Reject v011 schema and partial resume rather than guessing a migration.

### Task B4 — Verification and stop gate

```bash
uv run --no-sync pytest -q \
  tests/algos/test_fada_input_contract.py \
  tests/algos/test_fada_planner_idm.py \
  tests/algos/test_fada_source_collection.py \
  tests/algos/test_fada_replay_and_admission.py \
  tests/algos/test_fada_alternating_training.py \
  tests/algos/test_fada_persistence.py \
  tests/algos/test_fada_workflows.py
```

After focused tests, run the repository-required test gate and a separately authorized formal runtime audit. Planner–IDM training remains a separate authorization after runtime admission.

## Completion boundary

Unit A now has local module evidence but no trained Oracle or policy-quality claim. The next gate is a formal runtime audit, followed by explicit authorization for the privileged Oracle long run. Unit B remains blocked by a real admitted Unit A lineage and policy-quality evidence.
