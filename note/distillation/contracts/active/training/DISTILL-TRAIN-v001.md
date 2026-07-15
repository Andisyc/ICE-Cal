---
contract_id: DISTILL-TRAIN-v001
status: active
effective_date: 2026-07-15
updated_date: 2026-07-15
supersedes: null
method_contract: DISTILL-METHOD-v001
concept_figure: note/architecture/concept/03_g1_multiteacher_distillation_method.data.json
---

# Single-Entry Multi-Role Distillation Training Contract

## Public Contract

Normal training is one command over one run directory. The human supplies role
owners and teacher checkpoint identities; the workflow owns collection,
assembly, bootstrap training, iterative DAgger, checkpoint continuation, and
artifact identity. Low-level collection and offline-update commands remain
diagnostic tools, not the formal completion path.

The new path is opt-in through one owner flag. With the flag disabled, all
existing `scripts/train_distill.py` branches retain their current behavior.

## Role Artifact Preflight

Every required role is resolved independently before training:

| Decision | Meaning |
| --- | --- |
| `REUSE` | An immutable dataset and manifest satisfy the current role contract. |
| `COLLECT` | No dataset exists; collect only this role. |
| `STALE` | The artifact exists but its teacher/config fingerprint no longer matches. |
| `INCOMPATIBLE` | Observation, action, projection, schema, or intent semantics cannot be reused. |

Reuse is never inferred from filename existence. The manifest must bind role,
task owner, teacher checkpoint hash, dataset hash, schema version, observation
and action dimensions, projections, command filter, thresholds, and sample
count. `STALE` and `INCOMPATIBLE` fail closed unless the run explicitly requests
replacement collection or a declared schema migration.

An unmanifested legacy dataset may be adopted only through the explicit
`training.workflow.adopt_legacy_artifacts=true` migration path. Adoption loads
and validates tensor dimensions, cached teacher actions, task, teacher path,
projections, command filter, and thresholds before writing the first manifest.
It is never a filename-existence shortcut.

Adding a height role must not invalidate compatible standing and walking role
datasets. Only the missing height artifact and downstream merged/student stages
are rebuilt. If height changes the common student observation schema, the
workflow must apply a named semantics-preserving migration to legacy rows or
stop as `INCOMPATIBLE`; silent padding is forbidden.

## Internal Stage Machine

The single entry owns these resumable machine stages:

```text
PREFLIGHT
-> BOOTSTRAP_DATA (reuse or collect each role, then assemble)
-> BOOTSTRAP_UPDATE
-> DAGGER_ITERATION_1..N
-> CANDIDATE
-> ACCEPTANCE
```

Each completed stage is committed to `run_manifest.json` before the next stage
starts. Intermediate datasets and checkpoints remain inspectable, but generated
paths are passed internally rather than manually retyped.

## Iterative DAgger Contract

DAgger is an outer loop, not one rollout followed by one final finetune:

```text
student_k
-> role-conditioned Rollout_k
-> corresponding teacher relabel
-> cumulative aggregate bootstrap + rounds 1..k
-> balanced optimizer updates
-> student_(k+1)
```

`dagger_iterations` counts outer rollout/relabel/update cycles.
`dagger_updates_per_iteration` counts minibatch updates inside one cycle.
Every new rollout must load the checkpoint produced by the previous cycle.
The update inside each cycle is the finetune; no separate final `Finetune`
method stage exists.

Each configured role participates according to its role owner. Role labels,
command intent, rollout expert, teacher target, behavior-loss expert, and
deployment expert must remain aligned. The cumulative dataset must preserve
bootstrap evidence and all completed DAgger rounds.

## Resume And Fork

- `resume`: continue the same run from its last committed stage. Immutable
  identities must match; increasing the target DAgger iteration count is
  allowed.
- `fork`: create a new run from a declared parent stage/checkpoint while
  referencing compatible immutable role datasets. The parent is never mutated.
- `fresh`: create a new manifest and collect only roles whose preflight returns
  `COLLECT`.

An interrupted stage is rerun from its beginning. A completed stage is never
silently repeated. Parent checkpoint hash, teacher hashes, dataset hashes,
resolved config fingerprint, completed outer iterations, sample counts, update
counts, and candidate hash are mandatory manifest fields.

## Ownership

- Hydra config owns workflow enablement, run/resume/fork identity, roles,
  budgets, migrations, and replacement policy.
- `scripts/train_distill.py` only dispatches into the workflow owner and adapts
  existing env/teacher/collector APIs.
- `src/unilab/algos/torch/distill/workflow.py` owns preflight, manifests, stage
  sequencing, cumulative multi-role DAgger, and fail-closed transitions.
- `collector.py`, `data.py`, `trainer.py`, and `dagger.py` retain local tensor,
  collection, loss, and single-role primitive contracts.

## Required Evidence

1. S1 contract tests for role decisions, hashes, schema mismatch, stage
   transitions, resume, and fork.
2. S2 connectivity tests proving the default-off branch is unchanged and the
   enabled branch reaches the workflow owner.
3. A tiny multi-role runtime probe proving round `k+1` rolls out the checkpoint
   updated in round `k`, cumulative sample counts grow, and roles stay balanced.
4. Formal-run evidence recording exact role reuse/collection decisions and the
   produced checkpoint identity.
5. Physical acceptance remains separate: repeated reset, low-speed walk,
   walk-to-stop recovery, base height, tilt, and termination.

## Non-Scope And Stop Conditions

- This contract does not qualify the absent height teacher.
- It does not treat checkpoint size, finite actions, or low offline MSE as
  physical acceptance.
- Transition-conditioned walk-to-stop collection remains a required downstream
  role/scenario contract; generic stand and walk DAgger alone cannot close it.
- Do not recommend another long run until the single-entry owner, manifest,
  cumulative multi-role DAgger probe, and exact checkpoint identity pass.
