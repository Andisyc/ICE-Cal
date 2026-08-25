# FADA Two-Stage Planner-IDM Training Design

## Status and authority

Status: human-confirmed on 2026-08-25.

This design replaces the interleaved per-iteration IDM/Planner optimizer schedule. It preserves the
66-D observation contract, 29-D action, 3-D command, `H=30`, `K=6`, Planner-to-IDM Decoder
interface, Oracle identities, source-role semantics, replay quotas, and persistent-async runner.

## Design Inspector

| Decision | Accepted behavior | Forbidden behavior |
|---|---|---|
| DP-TS-01 phase boundary | One invocation owns exactly one phase: `idm_pretrain` or `planner` | Updating IDM and Planner in the same invocation |
| DP-TS-02 IDM source | `idm_pretrain` repeatedly collects final/intermediate Oracle trajectories and updates only IDM | Student rollout or Planner optimizer steps during IDM pretraining |
| DP-TS-03 freeze identity | Planner phase requires a completed IDM-pretrain checkpoint, loads only IDM weights, hashes them, and keeps them byte-identical | Loading an incomplete/legacy/Planner checkpoint as pretrained IDM or changing IDM after admission |
| DP-TS-04 Planner source | Planner phase collects only main DAgger data: iteration zero Oracle rollout, later iterations current Planner-IDM rollout | Using DAgger collection to update IDM or collecting intermediate Oracle rows that only IDM can consume |
| DP-TS-05 gradient path | Planner action loss differentiates through the frozen IDM computation into Planner outputs, while IDM parameters have `requires_grad=false` and receive no optimizer step | Detaching IDM output or accumulating/updating IDM gradients |
| DP-TS-06 persistence | Schema-4 checkpoints bind phase, phase completion, one phase-owned optimizer, and pretrained IDM identity; inference remains able to read schema 1-3 | Training resume, multiple optimizer owners, output/pretrained path aliasing, or silent cross-phase compatibility |

## Components and ownership

- `fada_training_phase.py` owns legal phase names, config validation, source families, and canonical
  IDM tensor identity.
- `fada_trainer.py` owns phase-specific optimizer and gradient authority.
- `fada_workflow.py` is the composition root that creates one phase-specific trainer and admits
  pretrained IDM state before mutable runtime work.
- `fada_persistent_workflow.py` and `fada_legacy_workflow.py` orchestrate phase-specific collection,
  update, and save without defining phase semantics.
- `fada_async_runtime.py` owns rollout-policy selection inside the resident collector.
- `fada_checkpoint.py` owns schema-4 phase persistence and old/new reader policy.

## Data flow

`idm_pretrain`:

```text
final/intermediate Oracle rollout -> source replay -> IDM loss -> IDM optimizer
-> schema-4 completed IDM checkpoint
```

`planner`:

```text
completed IDM checkpoint -> strict IDM-only load + permanent freeze
-> Oracle bootstrap / Planner-IDM DAgger rollout -> Oracle action label
-> Planner future -> frozen IDM Decoder -> action loss -> Planner optimizer
-> schema-4 Planner checkpoint
```

## Failure and compatibility policy

- Missing phase, unknown phase, missing pretrained-IDM path, incomplete IDM checkpoint,
  architecture mismatch, or wrong checkpoint phase fails before environment/runtime creation.
- v010 does not support training resume. `resume_path` and `initial_weights_path` must remain null;
  each phase is a fresh campaign.
- Planner output `checkpoint_path` must not resolve to `pretrained_idm_path`.
- Both phases require a fresh, non-existent output checkpoint path; admission rejects before the
  first save, environment creation, replay mutation, or runtime construction.
- Schema 4 stores exactly one optimizer payload named by its phase; missing or extra optimizer
  owners reject before policy mutation.
- The IDM identity digest is canonical over sorted state keys, dtype, shape, and contiguous CPU
  tensor bytes; it never hashes device-dependent serialization bytes.
- Planner admission recomputes the canonical digest from the stored IDM tensors and compares it
  with checkpoint metadata before mutating the target policy.
- Planner phase holds IDM in eval mode as well as `requires_grad=false`.
- Playback accepts schema 1-4 so existing negative-evidence checkpoints remain inspectable.
- `initial_weights_path` is retired from the active 66-D source route; stage transfer uses only
  `pretrained_idm_path` and copies only IDM weights.

## Executable acceptance

1. A trainer in `idm_pretrain` changes IDM and never Planner.
2. A trainer in `planner` changes Planner and keeps every IDM tensor byte-identical.
3. The async collector always uses Oracle rollout and intermediate Oracles in `idm_pretrain`;
   Planner phase preserves Oracle-at-zero then student rollout and never loads intermediate Oracles.
   Exact role spans, configured intermediate-Oracle coverage, scenario/cold-start quotas, and the
   Planner replay's absence of intermediate/planner-ineligible rows are asserted.
4. Schema-4 round-trip persists phase, one optimizer owner, and IDM identity; resume and malformed
   mixed-owner payloads reject.
5. The official persistent workflow produces phase-specific result metadata and never calls the
   inactive optimizer.
6. Both phase configurations pass the production setup admission boundary, while pre-existing
   outputs and tampered IDM identity metadata fail before mutable runtime work.

No offline test or formal audit proves learned walking quality. Long training remains a separate
human-authorized action.
