# G1 Multi-Teacher Distillation Control Room

This directory is the default human/LLM entrypoint for the G1 multi-teacher
distillation work. The historical migration compendium is evidence and history,
not the current design authority.

## Default Read Path

1. [Concept Figure](../architecture/concept/03_g1_multiteacher_distillation_method.data.json)
   - Human-controlled method intent.
2. [Active Method Contract](contracts/active/method/DISTILL-METHOD-v001.md)
   - Accepted semantics and design-point details.
3. [Current Task Canvas](task_canvas.md)
   - Current problem, verified facts, active proposal, and next decision.

Stop after these three unless a concrete question requires deeper evidence.

## Drill Down By Question

- Current workflow implementation plan:
  [Single-Entry Training Workflow](plans/single_entry_training_workflow_proposal.md)
- Current acceptance state: [Checklist](checklists/current.md)
- Verified facts: [Current Evidence](evidence/current.md)
- Archived reset/stop diagnosis:
  [Playback Reset And Stop-Transition Root Causes](evidence/2026-07-15-playback-reset-and-stop-transition-root-causes.md)
- Code owners: [Method-to-Code Atlas](../architecture/architecture/02_g1_distillation_method_to_code.data.json)
- Repository runtime: [UniLab Runtime Atlas](../architecture/runtime/01_unilab_runtime_atlas.data.json)
- Superseded/completed plans and migration history: [History](history/README.md)

## Human/LLM Interaction Loop

```text
Human edits or confirms Concept Figure intent
  -> LLM proposes/versions the matching contract detail
  -> Human confirms semantic changes
  -> LLM refreshes Concept Figure + Method-to-Code + Runtime Atlas
  -> Engineering plan and checklist define bounded work
  -> Code/tests/live probes produce evidence
  -> Checklist and atlas are refreshed before another training run
```

The Concept Figure is the human method control surface. Markdown contracts are
the detailed machine-readable source of accepted semantics. Architecture maps
are derived current-state views. Plans and checklists are replaceable. Evidence
is retained. Raw logs are restricted inputs and must not become design truth by
accumulation.

## Current Method In One Sentence

Use command intent to select standing, walking, and later height-control teacher
evidence, train role-specialized experts inside one MoE student, and deploy one
checkpoint whose routing follows the same intent contract.

## Current Boundary

- Standing and walking teachers exist.
- Height control remains an accepted future teacher role, but no qualified
  height teacher checkpoint is currently part of the active training route.
- Existing local student checkpoints and the RT-8 bounded candidate are
  candidate artifacts only. None is a promoted or accepted policy.
- Playback reset/command ordering is repaired and contract/live-reset tested;
  transition policy quality remains a separate unresolved boundary.
- Stop-transition collection and the scenario workflow are implemented; the
  RT-8 candidate still fails the physical walk-to-stop acceptance gate.
- The seven-command training workflow is intended only for diagnostics, but
  its explicit diagnostic-only labeling remains an open checklist item. The
  accepted public path is the single-entry workflow owned by
`DISTILL-TRAIN-v003`.
- The persistent runtime is integrated but not promoted. It remains explicit
  opt-in and the default execution mode remains `legacy` because E67 found no
  stable end-to-end speedup.
- Iterative DAgger means repeated outer student rollouts followed by relabel,
  cumulative aggregation, and update. Repeating optimizer updates on one fixed
  dataset is not an additional DAgger iteration.

## Formal Walk/Stand Command

The same command handles fresh collection, reuse, Bootstrap, iterative DAgger,
and resume. Existing legacy datasets can be adopted once only after their
tensor and metadata contracts pass; the flag does not permit filename-only
reuse.

```bash
CUDA_VISIBLE_DEVICES=0 \
UNILAB_G1_WALK_TEACHER=/ssd1/cyx/UniLab/model/G1WalkFlat/model_5000.pt \
UNILAB_G1_STAND_TEACHER=/ssd1/cyx/UniLab/model/G1StandStill/model_5000.pt \
UNILAB_G1_WALK_DATASET=/ssd1/cyx/UniLab/model/teacher/walk_flat_teacher_policy.pt \
UNILAB_G1_STAND_DATASET=/ssd1/cyx/UniLab/model/teacher/stand_teacher_policy.pt \
HYDRA_FULL_ERROR=1 PYTHONWARNINGS="ignore" \
uv run train \
  --algo distill \
  --task g1_walk_flat \
  --sim mujoco \
  workflow=g1_walk_stand \
  training.workflow.run_dir=/ssd1/cyx/UniLab/logs/distill_workflow/g1_walk_stand \
  training.workflow.adopt_legacy_artifacts=true
```

On the first run, each role reports `REUSE` or `COLLECT` independently. Reusing
the same `run_dir` resumes only unfinished DAgger iterations. Increase
`training.workflow.dagger_iterations=N` to continue the same run to a larger
outer-loop target. Use `training.workflow.mode=fork` plus
`training.workflow.parent_run_dir=...` when downstream logic changes but
compatible role artifacts and the parent checkpoint should be preserved.

The resulting candidate is
`<run_dir>/checkpoints/dagger_iteration_<N>.pt`; exact lineage and hashes are in
`<run_dir>/run_manifest.json`. Physical acceptance is still a separate gate.
