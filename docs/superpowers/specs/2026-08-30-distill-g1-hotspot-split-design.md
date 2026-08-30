# Distillation and G1 Hotspot Split Design

## Goal

Remove the remaining confirmed Divergent Change hotspots without changing
training, dataset, collector, workflow, environment, observation, reward,
action, domain-randomization, checkpoint, Hydra, reset, or playback behavior.

## Accepted boundaries

- `data.py`, `collector.py`, and `workflow.py` remain compatibility facades.
- Dataset validation, values, diagnostics, merge, and persistence each have one production
  owner below `src/unilab/algos/torch/distill/`.
- Standard and transition collection have separate owners and share only
  stateless projection/reset helpers.
- Workflow contracts, artifact persistence, bootstrap, and DAgger iteration
  lifecycle have separate owners.
- `entry_workflow.py` remains the resolved-Hydra composition root; it does not
  own dataset, collector, update, or artifact rules.
- `G1WalkEnv` retains simulator/backend lifecycle, mutable rollout state,
  reset, and framework hooks. Observation, reward, command, and control
  decisions move to G1 sibling modules that never import `joystick`.

## Dependency direction

```text
entry_workflow -> workflow_bootstrap/workflow_dagger -> workflow_artifacts
               -> entry_collection/entry_training

data facade -> dataset/dataset_contract/dataset_diagnostics/dataset_merge/dataset_io
collector facade -> collection_standard/collection_transition -> collection_common
workflow facade -> workflow_contracts/workflow_artifacts/workflow_bootstrap/workflow_dagger

joystick -> walk_observation/walk_reward/walk_commands/walk_control
         -> walk_config/walk_math/action_trace
```

No production owner imports a script. Distillation owner modules do not import
their compatibility facade. G1 sibling modules do not import `joystick`.

## State and persistence

- Dataset values are immutable-by-convention tensors plus validated metadata;
  dataset IO alone owns serialization.
- Workflow DAgger owns iteration state transitions; workflow artifacts alone
  own hashes, manifests, atomic writes, resume, and legacy adoption.
- Environment mutable arrays, backend handles, command/curriculum state, and
  reset snapshots remain owned by `G1WalkEnv`.
- Extracted G1 decision functions receive explicit arrays/config/context and
  cannot inspect backend-private capabilities.
- Diagnostic modules are read-only projections and cannot feed training state.

## Compatibility and proof

- Existing imports and public callable signatures remain available through
  direct re-exports, not duplicate wrappers.
- Existing private Env lifecycle seams `_compute_obs`, `_compute_reward`,
  `_update_commands`, and `apply_action` remain on `G1WalkEnv`.
- Characterization tests cover dataset fields, collector reset/filtering,
  workflow resume/cleanup, G1 observation/reward/action, and legacy imports.
- Each new owner boundary is introduced by a RED import/behavior test before
  production movement, followed by focused GREEN tests and static checks.

## Non-scope and evidence boundary

- No algorithm, reward, observation, action, curriculum, normalization,
  checkpoint schema, Hydra default, registry, or simulator behavior changes.
- Playback hotspots are deferred.
- No training, simulation, server, deployment, commit, push, branch creation,
  or destructive cleanup is authorized.
- Passing offline tests proves only behavior-preserving code organization, not
  simulator reachability or policy quality.
