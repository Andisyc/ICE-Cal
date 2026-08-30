# Off-policy Lifecycle Owners Design

## Goal

Remove the two confirmed off-policy `Divergent Change` hotspots without
changing training, collection, replay, privilege, IPC, logging, checkpoint, or
cleanup behavior.

## Accepted behavior and non-scope

- Preserve the public `DoubleBufferOffPolicyRunner.learn(...)` and
  `off_policy_collector_fn(...)` signatures.
- Preserve actor construction, privileged Actor inputs, action sampling,
  terminal observation handling, replay row order and dtype, weight versions,
  queue messages, timing keys, logger fields, checkpoint filenames, save
  cadence, final save, and collector-death cleanup.
- Keep the existing Async Runner and Replay Pipeline protocols. No second
  collector/learner synchronization mechanism is admitted.
- Do not change Hydra configuration, model/checkpoint schemas, optimizer or
  gradient behavior, simulator/backend behavior, or research semantics.
- No training, simulation, playback, server operation, branch, commit, or push
  is part of this unit.

## Current problem

`DoubleBufferOffPolicyRunner.learn` owns both persistent runner configuration
and the mutable state/resources of one run. Its 563-line body creates replay
resources, queues, tracing and logging, starts the collector, waits for data,
executes learner updates, synchronizes weights, saves checkpoints, and performs
success/failure cleanup.

`worker._run_collector` receives a primitive data clump of roughly thirty
arguments and owns environment creation, weight synchronization, inference,
replay writes, episode accounting, queue synchronization, telemetry, and
cleanup in one 435-line function.

Both are on the source-teacher production route selected by
`scripts/train_offpolicy.py`.

## Architecture

### Collector owner

Create `offpolicy/collector_session.py` with:

- a frozen `OffPolicyCollectorSpec` containing the current entrypoint values;
- a frozen `OffPolicyCollectorDependencies` containing existing factories and
  decision functions, preserving current monkeypatch/test seams;
- an `OffPolicyCollectorSession` that exclusively owns the environment,
  connected weight-sync handle, Actor, observations, episode counters, timing
  accumulators, pending pack request, and cleanup for one child process.

`off_policy_collector_fn` keeps its current signature, constructs the spec and
dependencies, and delegates. `_run_collector` becomes the private compatibility
adapter. The session separates initialization, weight refresh and action
selection, environment/replay step, collector synchronization, telemetry, and
close phases. It borrows the replay buffer and queues and must not close or
replace them; it closes only the weight-sync handle exactly where the legacy
function did.

### Learner run owner

Create `offpolicy/double_buffer_session.py` with:

- frozen run options and dependency records;
- mutable per-run state owned by `DoubleBufferTrainingSession`;
- explicit preparation, resource construction, collector start, readiness,
  replay sampling/update, metrics/checkpoint, success finalization, and
  collector-death paths.

`DoubleBufferOffPolicyRunner` remains the persistent configuration and learner
owner. Its public `learn` method constructs and runs one session. Existing
runner helpers remain the authority for liveness, metric draining, checkpoint
saving, summaries, and shared-resource cleanup. The session may invoke those
narrow operations but must not duplicate them.

## Effect sketch

Normal learner flow:

`train_offpolicy -> DoubleBufferOffPolicyRunner.learn ->
DoubleBufferTrainingSession -> ReplayPipeline/Learner/Logger/Checkpoint`

Collector flow:

`AsyncRunner -> off_policy_collector_fn -> OffPolicyCollectorSession ->
Env/Actor/ReplayBuffer/IPC metrics`

Failure flow:

`queue timeout or dead collector -> _CollectorDiedError -> existing
_fail_collector_died -> logger close + replay pipeline close + failed summary`

No producer, tensor field, queue payload, or persistence owner changes.

## Verification

1. Architecture tests must first fail because both session owners are absent.
2. Existing privileged-input, terminal replay, collector death, checkpoint,
   trace, sync/async, and configuration tests must remain green.
3. Scoped Ruff, compileall, import checks, and the affected off-policy suite
   must pass.
4. Repository pytest is run offline; unrelated pre-existing missing artifacts
   are classified, not fabricated.

## Stop conditions

Stop rather than alter semantics if extraction requires changing the Async
Runner call contract, ReplayBuffer row contract, privilege observation source,
normalization formula, checkpoint schema/name, or collector/learner ordering.
