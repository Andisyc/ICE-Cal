# ICE-Cal FADA Repository Simplification

Status: accepted for one-shot offline implementation on 2026-09-07.

## Boundary record

- Requested behavior: make the active G1/FADA configuration and execution flow small,
  explicit, and aligned with UniLab ownership boundaries.
- Preserved behavior: SAC/TD3/FlashSAC construction, FADA model tensors and updates,
  checkpoint schemas, target fault/slope behavior, backend selection, and existing
  historical evidence.
- New explicit behavior: `mujoco_clean_baseline` is gait-free, privilege-free, and has
  all domain randomization disabled.
- Forbidden behavior: no training, simulation, branch creation, commit, push,
  checkpoint rewrite, reward redesign, optimizer change, or backend-private access.
- Compatibility: old checkpoint bytes remain loadable through the canonical profile
  matching their recorded behavior. Historical task names cease to be training
  entrypoints and are documented in a migration table instead of remaining aliases.

## Current production trace

`scripts/train_offpolicy.py` owns Hydra, runner construction, training lifecycle,
playback, export, and failure cleanup. FADA task identities are assembled through an
experiment-to-experiment inheritance chain up to eight files deep. FADA commands import
implementation modules directly from one flat package.

The characterization pinch points are Hydra composition, `build_runner`,
`play_offpolicy`, checkpoint contract validation, and the four FADA command scripts.

## Target ownership

1. `scripts/train_offpolicy.py` is a thin Hydra composition root and compatibility
   export surface only.
2. `unilab.training.offpolicy` owns runner construction and train/play lifecycle.
3. Hydra canonical task files own complete experiment values. A specialization may inherit
   exactly one canonical owner when it changes only one semantic dimension; historical
   experiment-to-experiment chains are forbidden.
4. `fada.source`, `fada.planner_idm`, and `fada.target` are the public method-stage
   namespaces. Existing implementation modules remain internal to avoid a no-value
   mass file move.
5. Runtime validators enforce structural invariants and cross-field consistency. They
   do not select experimental actuator indices, ranges, or probabilities.

## Canonical active profiles

| Profile | Purpose | Gait phase | Actor privilege | Domain randomization |
| --- | --- | --- | --- | --- |
| `mujoco_clean_baseline` | true comparison baseline | off | off | all off |
| `mujoco_fada_source` | v022 phase-neutral source teacher | off | on | broad physical DR; no targeted actuator faults |
| `mujoco_fada_phase_contact` | v024 command-gated source Oracle | on | on | broad physical DR; no targeted actuator faults |
| `mujoco_fada_phase` | v023 phase locomotion source | on | on | explicit grouped DR |
| `mujoco_fada_target` | actuator target rollout | profile-owned | off | selected fault only |
| `target_domain=slope_10/15` | slope target rollout | target-owned | off | all random DR off |
| `mujoco_context_teacher` | full-action Context teacher | inherited | runtime-owned | fixed left knee |

The distinct residual `mujoco_context_teacher_phase1` runtime remains separate. The
full-action v005-v007 chain is behavior-preservingly flattened into
`mujoco_context_teacher`.

## Migration map

- `mujoco_no_gait_single_reward` -> `mujoco_clean_baseline`
- `mujoco_fada_privileged_oracle_grouped_dr_lineage` -> `mujoco_fada_source`
- `mujoco_fada_privileged_oracle_phase_locomotion_grouped_dr_lineage` ->
  `mujoco_fada_phase`
- fixed/live/nominal/DR curriculum profiles -> historical diagnostics; no active
  training alias
- simple/phase-height/command/original-height profiles -> historical failed or
  superseded experiments; no active training alias

Old FADA source checkpoints use `mujoco_fada_source`; v023 checkpoints use
`mujoco_fada_phase`. The checkpoint's own FADA metadata remains the admission owner.

## Implementation batches and proof

1. Add RED architecture/config tests for canonical profiles, shallow defaults, thin
   script ownership, public FADA stage namespaces, and retired launchers.
2. Add the three canonical leaf configs with explicit values and compose tests.
3. Migrate current distill configs, tests, runbooks, and contracts to canonical names.
4. Remove superseded FADA task YAML and misleading experiment launchers from active
   code; preserve the mapping in documentation.
5. Extract off-policy runner/lifecycle/playback into `unilab.training.offpolicy`, with
   the script re-exporting established helper names during source compatibility.
6. Add public FADA stage namespaces and route production commands through them.
7. Remove targeted-fault selection checks from the privileged Oracle runtime, preserving
   structural and observation/checkpoint validation; fault identity belongs to downstream rollout
   collection.
8. Move slope scene/DR ownership into the existing `target_domain` group and remove
   redundant slope task files. Flatten the full-action Context teacher chain.
9. Run narrow RED/GREEN tests, affected FADA/off-policy/config tests, Ruff, and
   `make test-all`. Do not run training or simulation.

## Acceptance

- Canonical profile composition proves exact gait, privilege, reward, and DR semantics.
- Active profiles use at most one canonical semantic specialization layer and no
  historical experiment chain.
- `scripts/train_offpolicy.py` contains no runner or playback implementation.
- Official FADA scripts import stage namespaces instead of flat private modules.
- No historical reward experiment can be selected from the active task group.
- Existing checkpoint schemas and model/update code are byte-for-byte untouched.
- Focused and repository test gates pass, or the first genuine blocker is reported with
  the worktree preserved.
