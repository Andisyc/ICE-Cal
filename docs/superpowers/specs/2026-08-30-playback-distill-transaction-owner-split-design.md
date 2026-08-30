# Playback and Distillation Transaction Owner Split Design

## Goal

Reduce the remaining ICE-Cal playback and distillation maintenance hotspots without changing
policy IO, checkpoint/config contracts, simulator behavior, collection row order, reset/RNG
lifecycle, DAgger persistence order, or public entrypoints.

## Scope

The implementation covers two sequential engineering units in one human-visible transaction:

1. playback composition and diagnostics currently concentrated in
   `scripts/play_interactive.py` and `src/unilab/visualization/interactive_playback.py`;
2. transition collection, FADA source collection, multitask merge, and DAgger iteration
   transactions under `src/unilab/algos/torch/distill/`.

`G1WalkEnv`, Trainer math/update ownership, Reward semantics, Hydra values, checkpoint schemas,
and live execution are outside scope.

## Ownership design

### Playback

- `scripts/play_interactive.py` remains the Composition Root and owns CLI-to-viewer orchestration.
- `playback_overlay.py` owns MuJoCo debug geometry and viewer-model resolution.
- `playback_controls.py` owns keyboard/height command projection and observation probing.
- `playback_trace.py` owns read-only distillation trace formatting and diagnostics.
- `interactive_playback.py` remains the compatibility facade and factory entrypoint.
- focused playback modules may own session contracts or distillation routing only when the move
  preserves existing imports and dependency-injection seams.

### Distillation transactions

- public collection and workflow functions remain unchanged.
- one transaction owns mutable rollout or persistence state; helpers receive explicit typed state
  and never duplicate environment, RNG, history, optimizer, dataset, or manifest ownership.
- dataset merge helpers remain pure and local to `dataset_merge.py`.
- DAgger iteration work is separated from outer resume/iteration scheduling, while durable commit
  remains last.

## Invariants

- observation shapes, dtype/device, command intent, scenario label, transition age, and row order
  remain identical;
- same-state Oracle labeling precedes rollout action exactly as before;
- reset/done handling, window acceptance, compaction, and maximum-step failure remain unchanged;
- checkpoint resolution/load fallback, policy routing, keyboard behavior, MuJoCo overlays, and
  trace output remain compatible;
- no backend-private feature moves into environment or training code;
- no generic manager, mixin, callback framework, schema, or new external dependency is added.

## Evidence boundary

Characterization and focused offline tests may prove import compatibility and behavior
preservation. They do not prove simulator correctness, official-route connectivity, training
convergence, checkpoint quality, or policy quality. No live or long-running work is authorized.
