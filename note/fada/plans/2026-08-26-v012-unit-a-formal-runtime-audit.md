# FADA v012 Unit A formal runtime audit

Status: `HOLD — OFFLINE REPAIR REQUIRED`

Authority: read-only audit and local evidence writing only. This unit does not authorize simulator
execution, Oracle training, server work, checkpoint publication, or Git operations.

## Admitted identity

- Design: `FADA-METHOD-v012:FADA-TRAIN-v012`
- Checkout: `main@e0835756230e66acfb7026e04eaa9fa7002372b2` plus the current uncommitted Unit A implementation
- Official entrypoint: `scripts/train_offpolicy.py`
- Runtime selection: `privileged_locomotion_sac`
- Task/backend: `G1WalkFlat/MuJoCo`
- Checkpoint source: cold start

## Critical design-point matrix

| ID | Design point | Owner boundary | Minimum faithful proof | Current evidence | Disposition |
|---|---|---|---|---|---|
| DP-A01 | Actor directly consumes ordinary 98-D observation plus the sealed privilege bundle | actor factory + off-policy worker | official route constructs the production Actor with the resolved width | focused module tests only | not yet run |
| DP-A02 | Gait/phase Reward is fail-closed and task is exactly G1WalkFlat/MuJoCo | Hydra config + privileged SAC preflight | official entry resolves config and passes pre-env preflight | static and module evidence | offline-proven only |
| DP-A03 | Actual DR values reach the privilege observation without hot-path asset inspection | G1 env + SimBackend public interface | at least one reset and transition through the official route | module tests only | not yet run |
| DP-A04 | Replay and optimizer consume the privileged Actor/critic tensors | double-buffer runner + SAC learner | one real replay insert and optimizer update through the official route | no formal route evidence | not yet run |
| DP-A05 | Each saved artifact is self-identifying as one of 20 intermediates or the final Oracle | learner checkpoint schema + runner persistence owner | save/load a production checkpoint and validate iteration, role, lineage, config hashes, task/backend/action scale and dimensions before mutation | implementation omits required identity fields | blocked |
| DP-A06 | Exactly 20 compatible intermediates plus one final artifact form one lineage | lineage admission owner | validate the 20+1 production artifacts, not synthetic records | validator exists but has no production consumer | blocked |
| DP-A07 | Admitted module evidence matches the active design and checkout | module-alignment manifest | validator accepts the current manifest with exact design and checkout | current receipt is not a module-alignment manifest | blocked |

## Offline counterexamples

1. `2026-08-26-v012-unit-a-module-receipt.json` is rejected by the module manifest validator:
   it has no supported schema, design identity, checkout identity, or module records.
2. `FADAPrivilegedSACLearner.get_state_dict()` stores only schema version, lineage id,
   privilege schema, task name, and the direct-privilege flag. It does not store the Contract's
   canonical configuration hashes, checkpoint iteration/role, backend, action scale, seed,
   resolved layout/order, Reward/DR identity, or dimensions.
3. `FlashSACLearner.load_state_dict()` mutates model and optimizer state immediately and does not
   validate the FADA metadata first.
4. `validate_fada_oracle_lineage()` is exercised by tests but has no production caller. The generic
   runner derives iteration only from the filename and never seals it into the payload.

## Formal-test decision

The proposed smallest faithful official transaction would use the production entrypoint, actual
MuJoCo env, replay, optimizer, and the first scheduled checkpoint. It is not admitted while DP-A05,
DP-A06, and DP-A07 remain offline-fixable. Executing it now could show tensor compatibility but
could not prove the persistence and lineage claim that defines Unit A.

## Required repair before rerun

1. Add one owner-layer checkpoint identity object that seals iteration/role, lineage, canonical
   config hashes, task/backend/action scale/seed, dimensions, and resolved privilege order.
2. Validate that identity before any model, optimizer, env, or collector mutation on every Unit A
   load/admission path.
3. Connect the production 20+1 admission path to `validate_fada_oracle_lineage`; do not infer roles
   solely from filenames.
4. Refresh a validator-compliant module-alignment manifest tied to the exact design and checkout.
5. Re-review this same formal test. Only a `READY` review may authorize the bounded official-route
   transaction; long training remains a separate human decision.

## Negative scope

This HOLD does not claim runtime failure, CUDA failure, simulator incompatibility, convergence, or
policy quality. No simulator or training process was started.
