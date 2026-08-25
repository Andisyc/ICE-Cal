# ICE-Cal Calibration Owner Split Plan

**Status:** human-authorized one-shot local refactor; no commit or push.

**Checkout:** `main@5e955870b651da578c34be0ed98c82d01fd5ce3f`

## Required outcome

Split the two calibration hotspots by reason to change while preserving every existing public import,
runtime value, tensor contract, persistence envelope, fail-closed check, legacy Gateway, and atomic
write behavior. The existing `calibration.py` and `calibration_collection.py` remain compatibility
facades.

## Owner map

`calibration.py` becomes a facade over:

- `calibration_types.py`: constants, axis catalog/spec, rollout batch.
- `calibration_models.py`: DirectionBank and CoefficientEncoder.
- `calibration_readout.py`: PCHIP curves and stateful online readout.
- `calibration_artifact.py`: final artifact validation and atomic persistence.
- `calibration_policy.py`: calibrated frozen-policy aggregate.

`calibration_collection.py` becomes a facade over:

- `gain_collection_types.py`: gain protocol and transaction value objects.
- `gain_collection_provenance.py`: canonical protocol/config identity and digests.
- `gain_collection_artifact.py`: raw artifact build, validation, legacy Gateway, and atomic IO.
- `gain_collection_runtime.py`: environment transaction and full-grid collection.

## Dependency rules

- Types depend only on stable FADA tensor/config primitives.
- Models and readout depend on types or finite-tensor validation, never persistence or runtime.
- Artifact owners may depend on types/models/readout/provenance; runtime may call artifact builders.
- Facades only re-export. No implementation module imports a facade.
- The scenario transaction and calibrated policy remain intact aggregates.
- Private validation helpers are no longer imported from the facade.

## Execution

1. Add a structural test requiring the new owner modules, facade-only definitions, identity-preserving
   re-exports, and an acyclic implementation import graph; verify RED before production edits.
2. Extract `calibration.py` in dependency order and run core/training/runtime tests.
3. Extract `calibration_collection.py` in dependency order and run collection/CLI tests.
4. Run all calibration tests, Ruff, Pyright, compile/import checks, stale private-import checks, and
   diff review.
5. Leave the coherent refactor uncommitted and unpushed.

## Evidence boundary

Offline tests can prove behavior preservation and module boundaries only. They do not prove simulator
reachability, training efficacy, deployment, or policy quality.
