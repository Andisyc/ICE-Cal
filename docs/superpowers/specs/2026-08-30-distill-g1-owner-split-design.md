# Distillation Entrypoint and G1 Owner Split Design

## Goal

Reduce the two confirmed Divergent Change hotspots without changing training,
environment, checkpoint, observation, reward, domain-randomization, or playback
behavior.

## Accepted boundaries

- `scripts/train_distill.py` remains the Hydra/CLI composition root and keeps
  its existing import-level callable surface for repository callers.
- Distillation model construction and update routes, dataset collection, and
  the single-entry workflow move to production owners below
  `src/unilab/algos/torch/distill/`.
- `src/unilab/envs/locomotion/g1/joystick.py` remains the G1 walk environment
  and registry owner.
- G1 configuration/value objects, deterministic gait calculations, and the G1
  domain-randomization provider move to focused sibling modules.
- Existing imports from `unilab.envs.locomotion.g1.joystick` remain valid via
  explicit re-exports.

## Dependency direction

```text
scripts/train_distill.py
  -> distill.entry_training
  -> distill.entry_collection -> distill.entry_training
  -> distill.entry_workflow   -> entry_collection + entry_training

g1.joystick
  -> g1.walk_config
  -> g1.walk_math
  -> g1.walk_domain_randomization -> walk_config + walk_math
```

The production modules never import `scripts.train_distill`. The G1 sibling
modules never import `joystick`, preventing cycles and duplicated owners.

## State and compatibility

- Optimizer, replay, checkpoint, normalizer, collector, logger, environment,
  backend, reset, and curriculum lifecycle remain unchanged.
- No compatibility fallback or second implementation is retained: moved
  definitions have one production owner and the old modules only import them.
- Test monkeypatches move to the production owner when they target internal
  implementation seams; public script routing tests continue to patch the
  composition root.

## Proof

1. Characterization tests pin public imports, registries, route selection,
   workflow cleanup, reset plans, observations, rewards, and actions.
2. Each extraction first makes an owner-location test fail, then moves the
   existing implementation without semantic edits.
3. Focused script, workflow, G1 environment, reward, and DR tests run after
   each extraction.
4. Ruff, import-cycle probes, `git diff --check`, line recount, and final
   maintainability review close the offline engineering boundary.

## Non-claims

Offline refactoring tests do not prove simulator behavior, training quality,
checkpoint quality, deployment, or real-robot behavior. No live run, Git
commit, push, or destructive cleanup is authorized.
