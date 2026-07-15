# Engineering Plan: Single-Entry Multi-Teacher Distillation Workflow

Status: accepted semantics in `DISTILL-TRAIN-v001`; implementation active

## Problem

The current human workflow exposes seven implementation stages as seven manual
commands:

```text
collect walking
collect standing
merge teacher datasets
bootstrap distillation
DAgger rollout
merge teacher and DAgger datasets
finetune student
```

Collection, dataset assembly, aggregation, checkpoint continuation, and
finetuning are internal runtime responsibilities. Requiring the human to name
and reconnect every intermediate `.pt` file creates path drift, accidental
deletion, stale-parent training, and checkpoint-identity ambiguity.

## Accepted Human Contract

The public workflow should be one command and one run directory:

```text
teacher checkpoint identities + task owner config
  -> one distillation run command
  -> one resumable run directory
  -> candidate checkpoint + manifest + acceptance report
```

The exact CLI syntax remains unconfirmed until the owner audit, but the user
must not manually collect, merge, or reconnect intermediate datasets in the
normal path.

## Role-Aware Artifact Reuse Preflight

Before collection, the workflow must resolve every required teacher role
independently and report one decision:

```text
REUSE         compatible immutable dataset already exists
COLLECT       no compatible dataset exists for this role
STALE         dataset exists but its teacher/config identity changed
INCOMPATIBLE  dataset cannot enter the current student schema
```

Reuse must be proven from manifest content, not inferred from a filename. For
each role, validate:

- dataset schema version and file hash;
- role and command-intent labels;
- teacher checkpoint path and immutable hash;
- owner task/backend config fingerprint;
- student/teacher observation dimensions and projections;
- action dimension;
- command filter and thresholds;
- required metadata and minimum accepted sample count.

The collection plan is then role-local. For example, when adding height
tracking:

```text
stand   -> REUSE
walk    -> REUSE
height  -> COLLECT
```

Only height collection runs. Bootstrap/retraining may still read the reused
stand and walk datasets to preserve prior experts and router boundaries, but it
must not regenerate them.

If height tracking changes the common student observation schema, old stand
and walk data are not silently rejected or recollected. The workflow must first
choose an explicit schema migration, such as adding the declared default-height
feature to legacy role rows. If no semantics-preserving migration exists, the
preflight returns `INCOMPATIBLE` and stops for a design decision.

## Proposed Internal Runtime

The single run may retain internal stages because they have different evidence
boundaries, but the orchestrator owns all transitions:

```text
Stage A  Bootstrap
  role-aware dataset preflight
  -> collect only missing or explicitly stale roles
  -> role-preserving dataset assembly
  -> initial MoE behavior distillation

Stage B  Closed-Loop Adaptation
  for k = 1..N:
    rollout updated student_k
    -> teacher relabel newly visited states
    -> aggregate datasets 1..k
    -> run M optimizer updates
    -> produce student_(k+1)

Stage C  Candidate Gate
  immutable manifest
  -> repeated reset and transition sentinels
  -> candidate acceptance report
```

These are resumable machine stages, not seven human-operated steps.

## DAgger Loop Semantics

Two counters must remain explicit:

```text
dagger_iterations
  = number of outer Rollout -> Relabel -> Aggregate -> Update cycles

dagger_updates_per_iteration
  = number of minibatch optimizer updates inside one outer cycle
```

`20000` optimizer updates on one saved DAgger dataset still represent one outer
DAgger iteration because the updated student has not generated a new state
distribution. The low-level manual route:

```text
collect student-policy dataset
-> merge saved datasets
-> offline finetune
```

is therefore a single outer iteration unless the entire sequence is repeated
with the newly updated checkpoint.

The repository already contains a code-confirmed iterative route:

```text
training.online_dagger=true
-> run_online_dagger_update()
-> run_iterative_dagger_updates()
-> for iteration in range(dagger_iterations)
```

The workflow problem is not the absence of an iterative DAgger primitive. It is
that the default human procedure exposed the one-iteration diagnostic branches
instead of composing the existing iterative owner into the formal multi-role
training entry.

## Concept Figure Alignment Gap

The current `Student-State DAgger` block shows a feedback connection but does
not distinguish the outer DAgger iteration from inner optimizer updates. This
is a `figure-mismatch`, not a new top-level method contribution. The next
Concept Figure synchronization should label the feedback semantics as:

```text
Rollout_k -> Teacher Relabel -> Cumulative Aggregate -> Update -> Rollout_(k+1)
```

There must not be an independent `Finetune` method block. Finetuning is the
update operation inside each DAgger iteration.

## Proposed Ownership

- Hydra owner config: teacher identities, role mapping, sample/update budgets,
  transition schedule, output root, resume policy, and acceptance thresholds.
- `scripts/train_distill.py`: compose config and invoke the workflow owner only.
- Proposed `src/unilab/algos/torch/distill/workflow.py`: own stage sequencing,
  resume state, artifact identities, and fail-closed transitions.
- Existing `collector.py`, `data.py`, `trainer.py`, and `dagger.py`: retain
  their current local contracts; the workflow composes them rather than
  duplicating their logic.
- Checkpoint/manifest owner: record parent checkpoint, teacher hashes, config,
  datasets, stage, sample/update counts, and output hash.
- Dataset artifact registry: resolve immutable role datasets, validate reuse
  compatibility, and produce the per-role collection plan.
- Acceptance owner: repeated reset, command transitions, walk-to-stop recovery,
  base height, tilt, termination, and teacher-action differential.

## Proposed Run Directory

```text
run_dir/
  run_manifest.json
  datasets/
    role_refs.json
    bootstrap_merged.pt
    dagger_iteration_*.pt
  checkpoints/
    bootstrap_student.pt
    dagger_iteration_*.pt
    candidate.pt
  evidence/
    reset_acceptance.json
    transition_acceptance.json
```

Intermediate files remain inspectable but their paths are generated and
consumed by the workflow. Resume reads `run_manifest.json`; it does not rely on
the human remembering which checkpoint name belongs to which stage.
`role_refs.json` may reference immutable datasets produced by an earlier run;
the new run does not need to copy or regenerate those files.

## Required Integration Of Current Root Causes

1. Fix playback initial-command/reset ordering before using playback as an
   acceptance surface.
2. Replace stand-only and walk-only adaptation as the final route with an
   explicit walk-to-zero transition collection stage.
3. Keep cumulative DAgger aggregation internal; never ask the user to merge
   teacher and DAgger `.pt` files manually.
4. Produce a candidate, not a promoted policy, until repeated physical gates
   pass.

## Migration Boundary

- Existing low-level collection, dataset, offline update, and DAgger commands
  remain available as diagnostic tools.
- They are removed from the default human workflow, not deleted from the code.
- A one-iteration manual collect/finetune path must be labelled diagnostic and
  must not be reported as completion of iterative DAgger.
- Existing candidate checkpoints remain evidence artifacts and are not silently
  treated as parents for the new run.
- Height control remains outside the first implementation; the workflow schema
  must support another teacher role without changing the public command shape.
- Adding a role invalidates only that role's missing/stale data plus downstream
  student/DAgger/candidate stages. It does not invalidate compatible datasets
  owned by existing roles.

## Implementation Step Map

The work is split because artifact identity, orchestration, iterative learning,
and formal CLI integration have independent owner and evidence boundaries.

### Step 1 / 5: Contract And Figure Alignment

- Objective: activate the public workflow semantics and remove the one-shot
  DAgger ambiguity.
- Scope: training contract, Concept Figure, plan, checklist.
- Non-scope: code and live training.
- Owner files: `note/distillation/` and Concept Figure data.
- Expected evidence: registered active contract and synchronized feedback loop.
- Stop condition: role reuse and outer-loop semantics are unambiguous.

### Step 2 / 5: Artifact Preflight And Manifest

- Objective: resolve each role as `REUSE`, `COLLECT`, `STALE`, or
  `INCOMPATIBLE` from content identity.
- Scope: pure workflow owner, hashing, manifest persistence, semantic fixtures.
- Non-scope: environment rollout.
- Owner files: `workflow.py`, workflow contract tests.
- Expected evidence: tiny files exercise every decision and fail-closed path.
- Stop condition: adding one role does not invalidate compatible existing roles.

### Step 3 / 5: Single-Entry Bootstrap

- Objective: let one enabled branch collect only missing roles, assemble role
  data, train/bootstrap, and persist stage identity.
- Scope: callback orchestration over existing collector/data/trainer owners.
- Non-scope: DAgger and physical acceptance.
- Owner files: `workflow.py`, `train_distill.py`, config, connectivity tests.
- Expected evidence: default-off regression plus an enabled two-role toy chain.
- Stop condition: no manual merge or checkpoint-path handoff is needed.

### Step 4 / 5: Multi-Role Iterative DAgger And Resume/Fork

- Objective: run cumulative outer iterations over all configured roles and
  resume/fork without replaying completed stages.
- Scope: stage loop, checkpoint lineage, cumulative aggregate, interruption.
- Non-scope: height teacher qualification and GUI acceptance.
- Owner files: `workflow.py`, DAgger/data primitives, runtime probe tests.
- Expected evidence: round `k+1` consumes student `k`; cumulative counts and
  role labels are proven at each round.
- Stop condition: repeated optimizer updates on fixed data cannot masquerade as
  multiple DAgger iterations.

### Step 5 / 5: Formal Entry And Architecture Closeout

- Objective: expose one normal command and refresh current-state architecture.
- Scope: repository CLI/config routing, Method-to-Code Atlas, impact suite,
  persistent evidence.
- Non-scope: declaring policy quality accepted without live gates.
- Owner files: CLI, Atlas, checklist, evidence.
- Expected evidence: exact formal route, focused tests, stale-search gate.
- Stop condition: the seven-command route is documented only as diagnostics.

## Step End Reports

### Step 1 End Report

- Status: PASS.
- Changed: activated `DISTILL-TRAIN-v001`, registered it, synchronized the
  Concept Figure outer-loop labels, and replaced proposal questions with this
  implementation map.
- Evidence: JSON parser and `check_distillation_atlas.mjs` both exit 0.
- Remaining: no workflow code existed at this boundary.
- Next safe step: pure artifact preflight and manifest owner.

### Step 2 End Report

- Status: PASS.
- Changed: added cold-path role artifact hashing, sidecar manifests, canonical
  owner-config fingerprints, and fail-closed preflight decisions.
- Evidence: `uv run pytest tests/algos/test_distill_workflow.py -q` reports
  `4 passed`; focused Ruff passes after import formatting.
- Runtime facts: compatible stand/walk fixtures report `REUSE`, absent height
  reports `COLLECT`, changed teacher/dataset bytes report `STALE`, changed
  student observation dimension reports `INCOMPATIBLE`, and a dataset without
  a manifest is not reused.
- Remaining: collection/bootstrap dispatch is not implemented.
- Next safe step: single-entry Bootstrap orchestration over these decisions.

### Step 3 End Report

- Status: PASS.
- Changed: the enabled script branch now composes role task owners, collects
  only `COLLECT` roles, assembles role-labelled data, writes the Bootstrap
  checkpoint, and commits generated paths to `run_manifest.json`.
- Evidence: callback owner tests and script connectivity/profile tests pass;
  the default `training.workflow.enabled=false` leaves old branches intact.
- Remaining: outer DAgger, resume, and fork were outside the Step 3 gate.
- Next safe step: multi-role iterative DAgger lineage.

### Step 4 End Report

- Status: PASS for generic role-conditioned DAgger; transition scenario remains
  BLOCKED by its separate method contract.
- Changed: every outer iteration collects every configured role from the latest
  checkpoint, aggregates Bootstrap plus all prior rounds, updates, and commits
  lineage atomically. Resume skips completed rounds; fork references immutable
  parent sources and checkpoint.
- Evidence: round 1 reads `bootstrap_student.pt`, round 2 reads
  `dagger_iteration_1.pt`, cumulative source counts grow from 4 to 6, resume
  executes only the missing round, and fork leaves the parent manifest bytes
  unchanged.
- Remaining: no accepted walk-to-zero command schedule or physical threshold.
- Next safe step: formal CLI and Architecture closeout.

### Step 5 End Report

- Status: PASS for formal-route integration.
- Changed: `uv run train --algo distill` routes to the opt-in workflow owner;
  `workflow=g1_walk_stand` supplies the two-role config; explicit legacy
  adoption validates existing stand/walk datasets before adding manifests.
- Evidence: Atlas validator passes; final impact suite reports `302 passed`;
  Ruff passes on all touched Python files.
- Remaining: no live MuJoCo run was performed in this implementation task;
  reset ordering, transition-conditioned collection, and promotion remain open.
- Next safe step: fix playback reset ordering and define the walk-to-stop
  transition scenario before another policy-quality claim.
