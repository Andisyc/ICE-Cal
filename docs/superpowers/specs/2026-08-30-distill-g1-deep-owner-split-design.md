# Distill and G1 Deep Owner Split Design

## Accepted outcome

Reduce the remaining confirmed Divergent Change and long-transaction hotspots in
the distillation and G1 walk paths without changing training, tensor, reward,
action, domain-randomization, reset, persistence, Hydra, simulator, or checkpoint
behavior.

## Immutable behavior

- `BehaviorDistillationTrainer` remains the sole owner of model references,
  optimizer execution, gradient mutation, update count, and teacher evaluation.
- Teacher inference remains detached and no-grad. Loss signs, reductions,
  coefficients, optimizer ordering, gradient clipping, and update cardinality do
  not change.
- Dataset fields, row order, role/scenario labels, tensor shape/dtype/device,
  metadata, serialization, and checkpoint schemas do not change.
- DAgger preflight, resume, collection order, aggregation order, checkpoint
  activation, metrics, artifact validation, and manifest commit order do not
  change.
- `G1WalkEnv` remains the only mutable owner of backend, rollout, reward-log,
  command, curriculum, reset, and episode state.
- `G1WalkDomainRandomizationProvider` remains the public DR provider and preserves
  validation, random-number consumption order, reset payloads, and curriculum
  persistence.
- Existing import paths and private compatibility seams used by repository tests
  continue to resolve.

## Owner boundaries

### Trainer

`trainer.py` keeps forward/backward/optimizer orchestration. Pure label-to-expert
routing moves to `trainer_routing.py`; read-only trace construction moves to
`trainer_diagnostics.py`. These modules do not own models, optimizer state, or
update count.

### Dataset, collection, and workflow transactions

Each long transaction is decomposed into named phases inside its existing owner
module unless an intermediate object has a real invariant:

- dataset merge: validated source loading, compatibility accumulation, tensor
  concatenation, final dataset construction;
- transition collection: assignment, row capture, buffer finalization;
- DAgger: preflight/resume, one-iteration collection/update, durable commit.

No generic transaction framework or repository abstraction is admitted.

### G1 rewards

`walk_reward.py` owns only stateless numerical reward terms. `G1WalkEnv` builds
the existing `RewardContext`, supplies explicit configuration/projections, and
retains thin compatibility methods for the existing reward registry and tests.
Any reward term that needs mutable environment or backend lifecycle state stays
in `G1WalkEnv`.

### G1 domain randomization

`walk_actuator_randomization.py` owns stateless actuator-strength validation,
range scaling, and multiplier sampling. `walk_reset_randomization.py` owns
stateless command/gait reset decisions. The provider composes those helpers and
remains the only stateful/public lifecycle owner.

## Dependency direction

- compatibility facades -> production owners;
- trainer -> routing/diagnostics, never routing/diagnostics -> trainer;
- transaction entry functions -> local phase helpers and existing contract/IO
  owners;
- `joystick.py` -> pure reward helpers;
- DR provider -> pure actuator/reset helpers;
- pure G1 helpers never import `joystick.py` or backend subclasses;
- production modules never import `scripts/` or tests.

## Verification and stop conditions

New owner-boundary tests must first fail because the new production owners do not
exist. After construction, run narrow tests, all directly affected tests, Ruff,
compileall, reverse-import/cycle checks, and the established focused suite.

Stop and leave the corresponding behavior in its current owner if extraction
requires duplicated mutable state, a new schema, a changed RNG call/order, a
changed gradient/update sequence, backend-private access, a simulator/training
run, or a semantic choice. Offline evidence proves only behavior preservation
at tested boundaries, not runtime reachability or policy quality.

