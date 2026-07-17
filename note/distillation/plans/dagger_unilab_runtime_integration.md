# DAgger UniLab Runtime Integration Plan

Status: completed execution record. `DISTILL-TRAIN-v003` is active with
integration complete, promotion deferred, and `legacy` default. This file is
not the current next-action surface; dated evidence preserves step history.

Status: approved for staged execution on `codex/dagger-mainline-runtime`.

Date: 2026-07-16

## Outcome

Connect multi-role DAgger to UniLab's runtime mainline without copying runner,
IPC, or backend code into the distillation package. DAgger continues to own
rollout/relabel/aggregate/update semantics. UniLab continues to own process
lifecycle, weight publication, failure propagation, and cleanup.

This plan does not modify or instrument the currently running server job.

## Decision

Use interface integration, not code copying.

- Reuse `AsyncRunner` for a persistent collector process and lifecycle.
- Reuse `SharedWeightSync` for versioned student publication.
- Keep DAgger-specific requests, dataset schema, teacher relabeling, manifests,
  and the outer iteration barrier in `src/unilab/algos/torch/distill/`.
- Keep `scripts/train_distill.py` as Hydra composition and dependency wiring.
- Do not force DAgger rows into APPO's fixed rollout ring or off-policy replay
  schema. A schema-aware shared batch channel is a later, separately gated
  change.
- Do not add Motrix in this change. Backend owner configs, registrations, and
  sim2sim proof are a separate gate.

## Architectural Invariants

1. All scenario collections in outer iteration `k` consume exactly
   `student_k`; `student_(k+1)` is published only after collection,
   aggregation, and update complete.
2. A collector result records the requested and observed weight version,
   checkpoint identity, scenario identity, row count, and artifact path.
3. Worker errors reach the parent through the existing `AsyncRunner` error
   channel; shutdown reaps the process and all shared resources.
4. Legacy execution remains the default until the new route reaches its
   acceptance gate.
5. No task/backend business rule is added to `scripts/train_distill.py`.
6. Timing evidence is emitted as structured request/result metrics or an
   optional trace artifact. No live-process `print` instrumentation is needed.

## Current Bottlenecks and Evidence Boundary

Static dataflow proves that the current workflow:

- creates and closes role/scenario model and environment state repeatedly;
- collects scenarios sequentially before updating;
- converts and accumulates per-step tensor chunks in the collector;
- rematerializes the cumulative aggregate on each outer iteration;
- runs the offline optimizer in a single Python update loop.

Static evidence does not prove which item dominates the active server run.
Therefore this plan first removes repeated lifecycle cost and adds structured
stage metrics. Aggregate-copy and learner throughput changes require measured
evidence from the new route.

## Step Map

### HP-0: Governance Baseline

- Scope: this plan, proposed runtime contract, checklist, task canvas, isolated
  worktree, and focused baseline tests.
- Evidence: existing focused baseline passes (`315 passed`).
- Stop condition: owner boundary and OFF-default are explicit before code.

### HP-1: Persistent Collector Protocol

- Scope: top-level picklable collect request/result records and a concrete
  `AsyncRunner`-owned request service with one long-lived worker.
- Non-scope: real G1 env creation, teacher loading, shared dataset storage.
- Tests first: request roundtrip, sequential request identity, worker exception
  propagation, cooperative cleanup, and observed weight-version reporting.
- Stop condition: an S2 spawned fake worker proves lifecycle and protocol.

### HP-2: DAgger Barrier Adapter

- Scope: adapt `run_multirole_dagger_workflow` to an explicit collector service
  interface while retaining the legacy callback route.
- Tests first: every scenario in iteration `k` sees one version/checkpoint;
  publish occurs only after the updater creates `student_(k+1)`; resume keeps
  manifest identity; default route remains byte-for-byte semantically legacy.
- Stop condition: S1/S2 workflow tests pass with no change in aggregation or
  checkpoint lineage.

#### HP-2 Execution Contract

- Scope: add one workflow-owner switch, `execution_mode`, and connect the
  scenario loop to a service interface that activates one checkpoint/version
  before collecting all scenarios in an outer iteration.
- Non-scope: Hydra/script wiring, real G1 factories, `SharedWeightSync`
  attachment, multiprocessing performance, aggregation changes, and training.
- Modify: `src/unilab/algos/torch/distill/workflow.py` owns mode validation,
  request construction, barrier ordering, and manifest evidence.
- Shared interface: `src/unilab/algos/torch/distill/async_runtime.py` owns
  request/result validation used at the connector boundary.
- Test: `tests/algos/test_distill_workflow.py` owns OFF regression, ON event
  trace, illegal mixed-state rejection, and manifest assertions.
- Core parameter path: current checkpoint -> service activation -> expected
  weight version -> every scenario request/result -> iteration manifest ->
  updater output -> next iteration activation.
- Test class: core parameter path plus S2 offline connectivity. Structured
  event assertions are sufficient; the server process is not instrumented.
- Command: `uv run pytest tests/algos/test_distill_workflow.py -q -k
  'execution_mode or scenario_manifest_and_quota_sources'`.
- Expected result: legacy callback ordering remains unchanged and emits no new
  persistent fields; persistent mode emits one activation before all scenario
  collects, then update, then the next activation.
- Stop condition: all scenario requests in iteration `k` carry one checkpoint
  and one version, and no activation of `student_(k+1)` occurs before update
  creates it.

#### HP-2 Parameter Inventory

| Parameter | Owner | Old/OFF value | ON value | Consumers | Persistence risk |
|---|---|---|---|---|---|
| `execution_mode` | workflow owner | `legacy` | `persistent_async` | scenario dispatch | new ON-only manifest field |
| `collect_scenario` | legacy workflow adapter | existing callback | forbidden | legacy scenario collection | unchanged when OFF |
| `scenario_collector` | persistent workflow adapter | forbidden/`None` | required service | checkpoint activation and collection | ON-only request/result evidence |
| `expected_weight_version` | persistent service barrier | absent | one non-negative integer per outer iteration | every scenario request/result | ON-only iteration/scenario record |

Forbidden half-open states: `legacy` with `scenario_collector`,
`persistent_async` with `collect_scenario`, persistent mode without scenarios,
or persistent mode without a service.

### HP-3: Real Distillation Runtime Owner

- Scope: move reusable role/scenario runtime construction out of the script
  into a distillation owner module; keep teachers and environments resident;
  attach the student to `SharedWeightSync`; write existing immutable dataset
  artifacts and return metadata only.
- Config: add `training.workflow.execution_mode` with `legacy` default and
  explicit `persistent_async` opt-in.
- Tests first: Hydra OFF/ON composition, fake owner connector, checkpoint and
  version mismatch fail-closed, one bounded real MuJoCo connectivity run.
- Stop condition: persistent mode produces a valid manifest-compatible
  artifact and cleanly exits. No policy-quality claim.

#### HP-3 Split Rationale

HP-3 crosses two independent owners and is therefore executed as two bounded
steps:

- HP-3a owns Hydra and entrypoint routing only.
- HP-3b owns real environment/model persistence and shared-weight publication.

HP-3a can be accepted with an injected semantic fake runtime; default mainline
ON execution remains fail-closed until HP-3b supplies the production factory.

#### HP-3a Execution Contract

- Scope: add `training.workflow.execution_mode=legacy`, trace it through
  `run_single_entry_workflow`, select exactly one legacy callback or injected
  persistent service, and close the service after workflow exit.
- Non-scope: real env/model construction, `SharedWeightSync`, performance,
  dataset schema, aggregation, optimizer, and live MuJoCo.
- Modify: `conf/distill/config.yaml` owns the OFF default;
  `scripts/train_distill.py` owns assembly only.
- Test: `tests/scripts/test_train_scripts.py` owns OFF/ON dispatch and cleanup;
  `tests/config/test_config_system.py` owns compose/default evidence.
- Core parameter path: Hydra value -> entrypoint parse -> factory requirement ->
  `run_multirole_dagger_workflow(execution_mode, collect_scenario,
  scenario_collector)` -> result summary.
- Test class: core parameter path plus S2 offline connectivity. Assertions are
  sufficient because no simulator/runtime semantics are exercised.
- Command: `uv run pytest tests/scripts/test_train_scripts.py
  tests/config/test_config_system.py -q -k 'distill and execution_mode'`.
- Expected result: OFF supplies only the legacy callback; ON supplies only the
  injected service, passes all role/scenario owner inputs to its factory, and
  closes it; ON without a production/injected factory fails closed.
- Stop condition: no half-open combination reaches workflow and old config
  composes with `legacy` by default.

#### HP-3a Parameter Inventory

| Parameter | Owner | Old default | OFF | ON | Consumers | Risk |
|---|---|---|---|---|---|---|
| `training.workflow.execution_mode` | distill Hydra root | absent/legacy behavior | `legacy` | `persistent_async` | script and workflow owner | run config only; checkpoint schema unchanged |
| `persistent_scenario_collector_factory` | entrypoint injection boundary | absent | forbidden/unused | required until HP-3b production factory | service construction | no persistence/playback effect |
| `collect_scenario` | legacy adapter | callable | callable | `None` | workflow owner | existing behavior preserved |
| `scenario_collector` | persistent adapter | absent | `None` | factory result | workflow owner | closed in entrypoint `finally` |

Forbidden half-open states: unknown mode; ON without scenarios; ON without a
factory; OFF with an injected factory; or forwarding both collector routes.

#### HP-3b Split Rationale

HP-3b crosses shared-weight publication and simulator/model resource caching.
Execute them separately:

- HP-3b1: parent checkpoint -> `SharedWeightSync` -> resident worker student.
- HP-3b2: resident role teachers/envs and real role/transition collection.

#### HP-3b1 Execution Contract

- Scope: create a distillation runtime owner that lazily initializes one
  `SharedWeightSync`, validates checkpoint state keys/shapes, publishes one
  version per activation, and drives the existing persistent runner.
- Non-scope: Hydra default factory, G1 envs, SAC teachers, dataset collection,
  aggregation, optimization, and MuJoCo.
- Create: `src/unilab/algos/torch/distill/persistent_runtime.py` owns shared
  student lifecycle; `tests/algos/test_distill_persistent_runtime.py` owns a
  tiny linear-model worker probe.
- Shared owner: `async_runtime.py` remains process/request lifecycle only.
- Core parameter path: checkpoint state_dict -> key/shape validation -> shared
  buffer write/version -> worker read into resident model -> request result
  observed version and compact weight facts.
- Test class: core parameter path plus S2 spawned connectivity; the tiny model
  has hand-checkable weights and crosses actual shared memory.
- Command: `uv run pytest tests/algos/test_distill_persistent_runtime.py -q`.
- Expected result: two checkpoints publish versions 1 and 2; the same worker
  PID observes different exact weight sums without reconstruction; shape/key
  mismatch fails before publication.
- Stop condition: real `SharedWeightSync` version and worker-observed weights
  match both checkpoints, and cleanup releases process/shared memory.

#### HP-3b2 Human-Control Decision

Date: 2026-07-16.

The user delegates code-level lifecycle review to Codex because the migration
is otherwise a black box, while retaining the final decision on whether the
persistent route may enter bounded real training. Codex must stop and report if
teacher identity, reset isolation, dataset semantics, checkpoint/version, or
resource cleanup differs from the legacy route.

Approved technical defaults:

- cache teachers and envs by complete owner identity, never only by a
  `walk`/`stand` string;
- let `walk_to_stop` reuse the walking env only when backend, task owner,
  resolved env config, and `num_envs` are identical;
- explicitly reset each request and clear command, done/episode, and
  transition-age state before collection;
- require a legacy-versus-persistent dataset differential before live
  acceptance;
- keep `execution_mode=legacy` as the default until all HP-3b2 gates pass;
- do not touch the active server job, add Motrix, or introduce a new shared
  dataset format in HP-3b2.

#### HP-3b2 Execution Contract

- Objective: keep walking/standing teachers and compatible G1 envs resident in
  the persistent worker without changing DAgger row semantics.
- Scope: exact resource identity keys, lazy teacher/env caches, per-request
  reset isolation, role and `walk_to_stop` collection, structured lifecycle
  counters, production factory wiring, and bounded MuJoCo connectivity.
- Non-scope: aggregation/learner optimization, shared dataset transport,
  backend migration, long training, promotion, and policy-quality tuning.
- Owner modules: create a distillation worker/resource owner under
  `src/unilab/algos/torch/distill/`; keep `scripts/train_distill.py` limited to
  config composition and factory selection; reuse collector/data owners for
  row construction and persistence.
- Core identity path: role/scenario -> task owner + backend + resolved env cfg
  + `num_envs` + teacher checkpoint/spec -> cache key -> resident resource ->
  reset -> collection -> dataset/manifest identity.
- State-isolation path: previous request command/done/transition age -> explicit
  reset and command schedule -> next request first observation/row.
- Test class: S1/S2 fake-resource lifecycle and differential first; S4 bounded
  MuJoCo only after both offline gates pass.
- Fake lifecycle facts: one student init, one walking teacher init, one standing
  teacher init, one env init per exact owner key, no counter growth on repeated
  requests, reset count increments per request, and every resource closes once
  on normal and exceptional exit.
- Differential facts: legacy and persistent outputs must match schema/dims,
  role and command-intent counts, scenario and transition-age schedule, teacher
  checkpoint identity, and checkpoint/version lineage for the same bounded
  request. Tensor values need not be bit-identical when simulator sampling is
  nondeterministic, but any semantic mismatch blocks the live gate.
- Bounded MuJoCo sequence: `walk_flat -> static_stand -> walk_to_stop ->
  walk_flat` in one worker, with worker PID, cache keys, init/reuse/reset/close
  counters, checkpoint/version, teacher hashes, sample counts, done counts,
  collection time, and artifact-write time persisted as structured evidence.
- Stop condition: fake lifecycle, dataset differential, and bounded MuJoCo
  gates all pass; default remains legacy; the user receives the evidence and
  decides whether persistent mode may enter bounded training.

Forbidden implementation shortcuts:

- caching only by semantic role name;
- calling script-private business logic from the worker;
- skipping reset because an env is cached;
- recreating student, teacher, or env per request while reporting the process
  itself as persistent;
- weakening teacher/role/transition guards to make the differential pass;
- claiming speedup before HP-4 structured timing.

#### HP-3b2b Dataset Differential Execution Contract

- Scope: add an optional pre-reset handoff to the existing role and transition
  collectors, then compare legacy self-reset and persistent cache-reset outputs
  with deterministic semantic env fixtures.
- Non-scope: production env/teacher factories, subprocess spawn, MuJoCo,
  aggregation, update, or timing claims.
- Modify: `collector.py` owns `initial_reset`; `persistent_resources.py` owns
  the reset result passed to it; tests own role/intent/scenario/age comparison.
- Core path: env reset -> `(obs, info)` -> role/transition collector -> dataset
  fields and metadata. Legacy and persistent must enter row construction with
  equivalent reset state.
- Test class: S1 semantic differential with actual collector/data owners.
- Command: `uv run pytest tests/algos/test_distill_persistent_differential.py -q`.
- Expected result: schema/dims, role and intent counts, scenario labels,
  transition ages, teacher identity, and sample counts match; persistent path
  performs one reset rather than a cache reset plus collector reset.
- Stop condition: any semantic mismatch blocks production factory work.

### HP-4: Structured Performance Evidence

HP-4 is split because metrics ownership/schema, a bounded live A/B run, and the
bottleneck decision cross three independent evidence and user-control gates.
Do not combine instrumentation, live execution, and optimization selection in
one step.

#### HP-4 Gate 0A: Read-Only Identity Preflight

- Objective: confirm the isolated owner diff and identify every value that must
  be frozen after HP-4a, without pretending the pre-instrumentation tree is the
  final A/B code identity.
- Scope: review the HP-3b2 owner diff; inventory branch/diff, Hydra command,
  device, seeds, scenario/sample counts, student checkpoint, both teachers, and
  route-specific output directories.
- Non-scope: code changes, server sync, training, performance claims, or
  optimization.
- Owner files/modules: this plan, runtime checklist, and task canvas; source
  files remain read-only.
- Expected evidence: S0/T-persist inventory. The current dirty worktree is
  explicitly not accepted as the formal HP-4b identity.
- Stop condition: the identity fields and final-freeze owner are explicit, so
  HP-4a schema work may proceed. HP-4b remains blocked.

#### HP-4a Step 1/3: Metrics Contract

- Objective: define the stage metric semantics, owner boundaries, units, and
  run-local artifact schema before adding instrumentation.
- Scope: define names and semantics for cold start, weight sync, env
  init/reset/step, teacher inference, student inference, tensor packing,
  artifact write, cumulative aggregation, learner batch/device staging,
  forward/backward/optimizer, checkpoint save, total elapsed time, sample
  count, env-step count, and rows/second; implement only the pure schema,
  recorder, validation, and persistence owner in this step.
- Non-scope: running MuJoCo, changing collector/dataset semantics, optimizing a
  stage, adding ad-hoc console prints, touching the active server job, Motrix,
  or changing the default from `legacy`.
- Owner files/modules:
  - `g1_persistent_worker.py` will own request lifecycle/resource observations;
  - `collector.py` will own env-step, inference, and tensor-pack observations;
  - `workflow.py` will own aggregate/update/checkpoint observations;
  - `performance.py` owns schema validation, fake-clock assembly, derived
    throughput, identity consistency, and atomic persistence;
  - `scripts/train_distill.py` may later wire only the configured output path.
- Required metric identity: schema version, execution mode, outer iteration,
  scenario, worker PID, request ID, checkpoint path/hash, weight version,
  teacher hashes, config hash, seed, device, `num_envs`, row count, stage name,
  duration in seconds, and success/error/cleanup state.
- Expected evidence: S0 schema/config checks and S1 deterministic metric
  assembly/persistence tests with an injected/fake clock. Missing stages,
  negative durations, identity drift, or duplicate incompatible records fail
  closed.
- Stop condition: the artifact schema and owner map are inspectable, tests
  prove roundtrip/identity validation, and a Step 1/3 End Report returns control
  for user approval before HP-4b live execution.

##### HP-4a Execution Contract

- Scope: create `src/unilab/algos/torch/distill/performance.py` and its S1
  tests; refresh plan/checklist/testing/atlas ownership views.
- Non-scope: no collector/workflow/worker instrumentation, Hydra connector,
  MuJoCo, training, performance comparison, server changes, or default change.
- Core parameter path: immutable metric identity + stage/counts -> injected
  clock duration -> validated record -> derived rows/second -> atomic JSON
  artifact -> reload and revalidation.
- Test class: secondary contract and S3 persistence. A semantic fake identity
  and fake clock are sufficient because no simulator or tensor path is used.
- Test-first command: `uv run pytest
  tests/algos/test_distill_performance.py -q`.
- Expected result: schema roundtrip and fake-clock duration pass; invalid hash,
  negative duration/count, mode/version mismatch, identity drift, missing
  required stage, and duplicate incompatible record fail closed.
- Stop condition: the pure metrics owner passes focused tests and no timing
  call is added to a runtime owner. Return control before instrumentation or
  HP-4b.

##### HP-4a2 Connector Split

Runtime wiring crosses three independent owners and must remain split:

- HP-4a2a: persistent worker emits identity-free request stage observations;
- HP-4a2b: collector owns env/inference/tensor-pack observations;
- HP-4a2c: parent workflow enriches observations with immutable identity and
  writes the run-local artifact.

HP-4a2a must not reconstruct parent identity or hash checkpoint files inside
the measured request. `config_sha256`, formal checkpoint lineage, cross-route
run identity, and artifact completeness remain HP-4a2c responsibilities.

###### HP-4a2a Execution Contract

- Objective: connect existing persistent-worker request timings to validated,
  identity-free HP-4 stage observations without changing collection semantics.
- Scope: add an observation data object/conversion in `performance.py`; update
  `g1_persistent_worker.py` to emit `weight_sync`, `artifact_write`, and
  request-wide `total_elapsed` observations in result metadata; preserve the
  existing flat metrics mapping.
- Non-scope: collector internal timing, parent identity enrichment, JSON
  persistence, Hydra/script wiring, checkpoint hashing, cleanup-final records,
  MuJoCo, training, A/B, or optimization.
- Core path: request start -> weight sync -> scenario collection -> artifact
  write -> request end -> stage observations -> `DaggerCollectResult.metadata`.
- Count semantics: `weight_sync` carries zero rows/steps; `artifact_write` and
  `total_elapsed` carry dataset rows; `total_elapsed` carries the collector's
  existing `env_steps`; cleanup remains `pending` until worker shutdown.
- Test class: secondary connector with deterministic fake clock plus the
  existing semantic fake G1 worker. No live simulator fact is claimed.
- Test-first command: `uv run pytest
  tests/algos/test_distill_performance.py
  tests/algos/test_distill_g1_persistent_worker.py -q`.
- Expected result: observations roundtrip through the schema; worker metadata
  has exactly the three stages, exact fake-clock seconds/counts, same request
  output semantics, and unchanged flat metrics.
- Stop condition: focused and impact tests pass; no collector/workflow/script
  change, no JSON artifact, and no HP-4a2b/2c work begins automatically.

###### HP-4a2b Execution Contract

- Objective: measure only collector-owned inference, env-step, and tensor-pack
  boundaries and preserve them through dataset and worker metadata.
- Scope: add an opt-in identity-free accumulator in `performance.py`; add
  `performance_clock=None` to both collectors; emit ordered
  `teacher_inference`, `student_inference`, `env_step`, and `tensor_pack`
  observations into a replaced dataset metadata dict; let the persistent
  worker pass validated collector observations between `weight_sync` and
  `artifact_write`.
- Non-scope: env reset/resource init timing, parent identity, JSON persistence,
  Hydra/script wiring, legacy default instrumentation, learner/workflow stages,
  MuJoCo, training, A/B, or optimization.
- Stage ownership: teacher inference counts every teacher-policy input row;
  student inference counts every rollout-policy input row; env step counts
  actual `env.step` calls; tensor pack counts accepted dataset rows and includes
  per-loop tensor conversion plus final dataset build.
- Core path: fake clock -> local stage accumulator -> validated observations ->
  copied dataset metadata -> worker validation/pass-through -> result metadata.
- Test class: core parameter path with semantic role/transition env fixtures.
  The default `None` path must emit no performance metadata.
- Test-first command: `uv run pytest
  tests/algos/test_distill_persistent_differential.py
  tests/algos/test_distill_g1_persistent_worker.py
  tests/algos/test_distill_performance.py -q`.
- Expected role oracle: durations `2/2/1/3`, inference rows `4/4`, env steps
  `1`, packed rows `4`.
- Expected transition oracle: durations `4/4/3/5`, inference rows `16/8`, env
  steps `3`, packed rows `8`.
- Stop condition: both collector variants and worker pass-through match exact
  stage order/counts/durations; legacy/default metadata stays unchanged; no
  HP-4a2c work begins automatically.

###### HP-4a2c Execution Contract

HP-4a2c is user-authorized and remains split across three independently
verifiable owner/connector boundaries:

1. HP-4a2c1 pure identity enrichment owner.
   - Objective: define one immutable run context and enrich the exact persistent
     request observation sequence without reading files in the worker hot path.
   - Scope: `performance.py` context validation, canonical required-stage order,
     full-run teacher hash set, request identity construction, and tiny golden
     tests.
   - Non-scope: workflow mutation, JSON writing, script wiring, live collection,
     Gate 0B, or HP-4b.
   - Core path: run context + request/checkpoint facts + seven observations ->
     seven full `DistillationStageMetric` records.
   - Test class: core parameter path with deterministic semantic records.
   - Command: `uv run pytest tests/algos/test_distill_performance.py -q`.
   - Expected result: exact identity/stage/count propagation passes; stage order,
     schema version, empty teacher set, and request/checkpoint drift fail closed.
   - Stop condition: the pure owner passes before workflow persistence begins.
2. HP-4a2c2 parent workflow persistence connector.
   - Objective: load/revalidate prior run-local metrics, enrich each successful
     persistent request, and atomically persist after every scenario.
   - Scope: `workflow.py` only plus workflow contract tests; the artifact path is
     fixed at `<run_dir>/distillation_metrics.json` and recorded in the manifest.
   - Non-scope: learner/aggregation/checkpoint timing, legacy metrics, script
     config construction, simulator execution, or cleanup-final observations.
   - Core path: worker metadata -> parent identity -> recorder -> atomic JSON ->
     reload/resume -> manifest path/hash/count.
   - Test class: offline connectivity and S3 persistence/resume contract.
   - Command: `uv run pytest tests/algos/test_distill_workflow.py
     tests/algos/test_distill_performance.py -q`.
   - Expected result: all scenarios in one outer iteration share checkpoint and
     version identity; resume is idempotent and identity drift fails closed;
     legacy creates no metrics artifact.
   - Stop condition: workflow artifact and resume proof pass before script wiring.
3. HP-4a2c3 formal entrypoint assembly connector.
   - Objective: construct the full-run context only for `persistent_async` from
     the resolved Hydra config, all role teacher checkpoints, seed, device, and
     configured `num_envs`.
   - Scope: `scripts/train_distill.py` assembly and script tests, followed by
     evidence/checklist/atlas refresh.
   - Non-scope: new Hydra fields, default changes, live training, server process,
     Gate 0B, HP-4b, Motrix, or optimization.
   - Core path: resolved cfg + role specs -> immutable context -> workflow arg;
     legacy -> `None` and no artifact.
   - Test class: S2 formal-route connectivity plus OFF-path isolation.
   - Command: `uv run pytest tests/scripts/test_train_scripts.py
     tests/algos/test_distill_workflow.py tests/algos/test_distill_performance.py -q`.
   - Expected result: persistent route passes the exact context, legacy passes
     none, and the affected suite remains green.
   - Stop condition: E44 and current derived views agree with code; return control
     before Gate 0B or HP-4b.

The run identity uses the unique canonical hashes of every configured role
teacher, not only the teacher selected by one scenario. The resolved workflow
configuration hash preserves role-to-teacher mapping. Scenario-specific teacher
metadata remains request evidence and cannot redefine the run signature.

#### HP-4 Gate 0B: Final Immutable A/B Identity

- Objective: freeze the actual post-HP-4a/connector code and complete A/B
  commands/assets before any HP-4b run.
- Scope: require a clean commit or immutable diff bundle, exact config/command,
  checkpoint and teacher hashes, seeds, device, workload, run order, and
  separate output directories for both routes.
- Non-scope: further code edits, training, timing execution, or optimization.
- Expected evidence: S0/T-persist identity artifact and human approval.
- Stop condition: any mutable code/config/asset or command asymmetry blocks
  HP-4b; otherwise emit a Gate 0B report and return control for approval.

Gate 0B execution result: `BLOCKED` by E45. RT-10 assets and a symmetric fork
workload compose are verified, but legacy has no structured metrics artifact,
workflow/learner/checkpoint stages are schema-only, cleanup-final evidence is
not persisted, and the dirty worktree has no valid immutable bundle. Do not
freeze or run HP-4b commands. Any repair is a new HP-4a2d proposal requiring
separate authorization and owner-by-owner Step End Reports.

Gate 0B rerun execution contract (authorized after E46):

- Scope: recompute asset hashes, compose both formal routes, freeze the current
  source snapshot, workload, balanced run order, commands, and output identity.
- Non-scope: executing either formal command, MuJoCo, training, server mutation,
  HP-4b, or any performance conclusion.
- Owner: governance evidence and run-artifact identity; Hydra is read-only.
- Core identity path: HEAD/worktree -> content manifest -> deterministic source
  archive -> SHA-256; canonical asset -> SHA-256; shared overrides -> two
  resolved configs -> allowed two-field diff.
- Evidence class: S0/S3 T-persist/T-oracle secondary contract assertions.
- Stop condition: any asset, compose, command, source, or output collision
  yields `BLOCKED`; complete immutable identities yield `PASS` and return
  control before HP-4b.

Gate 0B rerun result: `PASS` by E47. Seven canonical asset hashes match, both
resolved routes differ only by execution mode and run directory, a deterministic
740-file source bundle reproduces the same SHA-256 twice, and eight balanced
unique-output commands are frozen in the raw identity manifest. No command ran.
Control returns to the user before HP-4b.

#### HP-4a2d Measurement Symmetry Repair

HP-4a2d is user-authorized after E45 and is split into three owner-bounded
steps. It does not rerun Gate 0B or HP-4b automatically.

1. HP-4a2d1 legacy request metrics.
   - Objective: give the formal legacy scenario route a validated request
     artifact while preserving integer callbacks and workflow-disabled OFF behavior.
   - Scope: mode-specific request stages in `performance.py`; a rich legacy
     scenario result in `workflow.py`; opt-in fake-clock spans in
     `run_collect_dataset()` and the transition callback; formal context assembly.
   - Non-scope: workflow aggregation, learner/update stages, cleanup-final,
     MuJoCo, training, Gate 0B, or A/B.
   - Core path: legacy cold start -> collector stages -> artifact write -> total
     request -> parent identity -> atomic JSON/reload.
   - Test class: core parameter path plus OFF-path contract.
   - Command: `uv run pytest tests/algos/test_distill_performance.py
     tests/algos/test_distill_workflow.py tests/scripts/test_train_scripts.py -q`.
   - Stop condition: a semantic legacy fixture writes exact mode-specific
     records; old integer callbacks without context remain artifact-free.
2. HP-4a2d2 workflow and learner stages.
   - Objective: record aggregation at the workflow owner and batch staging,
     forward, backward, optimizer, and checkpoint save at their real owners.
   - Scope: injected-clock observations in `workflow.py`, `offline.py`, and
     `trainer.py`; update callback returns validated observations to the parent.
   - Non-scope: changing loss/update math, batch semantics, replay budgets,
     cleanup-final, simulator execution, Gate 0B, or optimization.
   - Core path: aggregate callback -> aggregate observation; dataset batch ->
     staging -> forward/loss -> backward/grad -> optimizer -> checkpoint save ->
     workflow-level identity -> artifact.
   - Test class: tiny golden learner fixture plus offline connectivity contract.
   - Command: `uv run pytest tests/algos/test_distill_performance.py
     tests/algos/test_g1_distillation_contract.py tests/algos/test_distill_workflow.py
     tests/scripts/test_train_scripts.py -q`.
   - Stop condition: exact fake-clock durations/counts pass and no algorithm
     output, update count, checkpoint, or OFF behavior changes.
3. HP-4a2d3 cleanup-final persistence.
   - Objective: persist route cleanup duration/state and the persistent worker
     close counters after service close.
   - Scope: one cleanup stage and finalizer owned by metrics/workflow; formal
     script closes then atomically updates metrics and manifest; legacy records
     its per-request cleanup ownership without a resident service.
   - Non-scope: changing process lifecycle, adding retries/fallbacks, live run,
     Gate 0B, HP-4b, or speedup claims.
   - Core path: workflow complete -> service close -> close report -> cleanup
     record + manifest counters -> atomic reload/hash/count validation.
   - Test class: lifecycle fake and S3 persistence/resume contract.
   - Command: `uv run pytest tests/algos/test_distill_persistent_runtime.py
     tests/algos/test_distill_workflow.py tests/scripts/test_train_scripts.py -q`.
   - Stop condition: complete cleanup is persisted for both modes, missing or
     malformed persistent close reports fail closed, and E46/current views agree.

HP-4a2d completion result: `PASS` by E46. The legacy request route, workflow
aggregation, learner owners, checkpoint save, and post-close cleanup now emit
validated records into the same atomic run-local artifact. The persistent
finalizer requires worker identity and resource counters. No simulator run,
Gate 0B rerun, HP-4b command, or speedup claim is included. The next action is
a separately authorized Gate 0B rerun that also freezes an immutable source
bundle.

#### HP-4b Step 2/3: Bounded Legacy/Persistent A/B

- Objective: compare `legacy` and `persistent_async` under one controlled,
  bounded workload without changing method semantics or training quality.
- Scope: one cold-start measurement and at least three steady-state repetitions
  per route using the same immutable code identity, checkpoint/teacher hashes,
  Hydra owner config, seed set, device, `num_envs`, scenario order, sample
  quotas, outer-iteration count, and output schema. Persist raw per-stage
  records and one run manifest per repetition.
- Non-scope: long training, default-on promotion, HP-5 optimization, server
  process instrumentation, cross-backend comparison, or checkpoint quality
  claims.
- Owner files/modules: formal `train_distill.py` entrypoint, workflow/runtime
  owners, HP-4a metrics artifact owner, and separate run-local output dirs.
- Required comparisons: dataset schema/dims, role/intent/scenario counts,
  transition-age schedule, teacher/checkpoint identity, outer barrier version,
  cleanup counters, total elapsed time, stage durations, and rows/second.
- Expected evidence: S2 formal-route connectivity plus S4 bounded local
  MuJoCo timing artifacts. Report raw values, median, range, and run order; do
  not report only a speedup ratio.
- Stop conditions: identity, dataset semantics, checkpoint/version, reset, or
  cleanup mismatch stops the A/B immediately. High run-to-run variance or
  missing stage records yields `partial`, not a speedup conclusion. Control
  returns through a Step 2/3 End Report before HP-4c.

HP-4b execution is user-authorized against E47's exact frozen manifest and is
split at live acceptance boundaries:

1. HP-4b1 frozen live preflight.
   - Scope: extract the bundle into the required absent cwd; verify embedded
     hashes, G1 XML/assets, imports, Hydra compose, parent/assets, and absent
     outputs before constructing an env.
   - Non-scope: training, timing verdict, source repair, or fallback to the
     mutable worktree.
   - Test class: core identity path plus live-sentinel readiness.
   - Stop: any missing resource or hash mismatch blocks all eight runs.
2. HP-4b2 bounded formal A/B execution.
   - Scope: run the eight manifest entries in exact order, one at a time, from
     the frozen cwd; after every run verify exit status, manifest, metrics,
     cleanup, checkpoint/version, and scenario artifacts before continuing.
   - Non-scope: rerun-on-failure, code edits, workload changes, or HP-4c.
   - Test class: S2/S4 live sentinel and formal-route integration.
   - Stop: first failed run or semantic/lifecycle mismatch stops the sequence.
3. HP-4b3 raw differential acceptance.
   - Scope: compare exact identities, dataset schema/role/intent/scenario/age,
     lifecycle counters, stage completeness, raw durations, median, and range.
   - Non-scope: source optimization or HP-5 proposal.
   - Test class: S3 T-diff/T-scale over raw run-local artifacts.
   - Stop: emit a new HP-4b evidence artifact and a Step 2/3 End Report, then
     return control before HP-4c.

HP-4b execution result: `BLOCKED` at HP-4b1 by E48. The deterministic E47
bundle omits `README.md`, which `pyproject.toml` requires to build UniLab.
Frozen-cwd `uv run` exits before import, env construction, or run 1. Do not copy
the mutable README into the bundle or continue to HP-4b2. Gate 0B must repair
and refreeze its complete package-build input set under separate authorization.

Gate 0B bundle repair is user-authorized after E48:

- Scope: replace the fragile allowlist with the complete Git-visible source
  inventory, including all root build metadata; create a new versioned bundle
  and identity manifest; remove the failed temporary cwd; extract into a new
  absent cwd; rerun full hash/build/import/G1 XML/compose preflight.
- Non-scope: source-code behavior changes, mutable-file injection into the
  extracted tree, A/B run 1, training, timing, HP-4c, or overwriting E47 raw
  artifacts.
- Owner: Gate 0B raw artifact builder and governance evidence only.
- Core path: Git-visible file -> embedded size/hash -> deterministic archive ->
  extracted file -> `uv_build` inputs (`README.md`, `LICENSE`, `pyproject.toml`)
  -> UniLab import -> package-owned G1 assets -> symmetric compose.
- Evidence class: S0/S2/S3 T-persist/T-oracle plus frozen-cwd live readiness.
- Stop: two identical bundle generations and one successful frozen preflight
  restore Gate 0B; return control before HP-4b execution.

Gate 0B bundle repair result: `PASS` by E49. The r2 bundle includes all 1241
Git-visible files and explicit package-build inputs, reproduces its SHA-256
twice, builds UniLab from the extracted cwd, verifies every source/asset hash,
loads the G1 MuJoCo XML, and reproduces the symmetric config hashes. The r2
output root remains absent. Control returns before HP-4b run 1.

HP-4b r2 execution is now user-authorized. Execute manifest orders 1 through 8
sequentially from `/private/tmp/unilab-hp4b-f7d87a15`. Validate each completed
run before starting the next; do not retry or change workload after a failure.

HP-4b r2 execution result: `BLOCKED` at order 1 by E50. All three legacy
scenario requests complete, but cumulative aggregation rejects mixed
`scenario_labels` presence across role and transition datasets. The partial run
has 21 request records and no aggregate, learner, checkpoint, or cleanup-final
evidence. Orders 2-8 did not start. Do not repair or rerun inside this step.

HP-4b schema-owner repair is user-authorized after E50:

- Scope: preserve `scenario` and `preserve_row_role_labels` when the workflow
  owner serializes parent cumulative sources into a fork manifest.
- Non-scope: changing `data.py` fail-closed validation, synthesizing fields in
  saved legacy artifacts, collector changes, partial-run reuse, refreeze, or
  HP-4b rerun.
- Owner: `workflow.py::fork_workflow_run`; existing data-owner annotation
  remains the only in-memory legacy-to-active-scenario adapter.
- Core path: parent source identity -> fork `bootstrap_sources` -> explicit
  scenario annotation -> complete transition-aware merge.
- Evidence: tiny fork identity regression plus transition-aware aggregate
  connectivity and affected workflow/data suites.
- Stop: source fields survive fork, legacy files remain byte-identical, mixed
  aggregation no longer occurs in the fixture, then return before Gate 0B.

Repair result: `PASS` by E51. `fork_workflow_run()` preserves the source
scenario and row-role flags; the focused workflow/data owner chain passes and
the affected suite reports `288 passed, 8 skipped`. Parent source hashes remain
unchanged. This source edit invalidates E49's frozen identity, so control
returns before a separately authorized Gate 0B refreeze; HP-4b is not rerun.

Gate 0B refreeze result: `PASS` by E52. The r3 deterministic bundle contains
1244 Git-visible files, has SHA-256 `f66ab818...7191`, and is extracted at
`/private/tmp/unilab-hp4b-f66ab818`. Frozen source import/hash, seven assets,
G1 XML, allowed-only compose differential, output absence, and the 312+8
affected suite pass. Identity remains `execution_authorized=false`; return
control before any HP-4b run.

HP-4b r3 execution result: `BLOCKED` at order 1 by E53. The exact nested
`uv run` exits 2 while initializing the default user cache because r3 does not
freeze `UV_CACHE_DIR` or the dependency-provider environment. Python/Hydra,
MuJoCo, collection, aggregation, learner, metrics, and cleanup do not start;
the formal output root remains absent and orders 2-8 are not attempted. Do not
retry or change commands inside HP-4b. The next bounded action is a separately
authorized Gate 0B execution-env repair and exact nested no-training preflight.

Gate 0B execution-env repair is user-authorized after E53:

- Scope: create a new r4 executable identity that reuses the immutable r3
  source bundle/cwd while freezing uv cache, dependency provider, no-sync,
  frozen-source import path, and progress environment.
- Non-scope: source/config/workload edits, r3 mutation, E53 log deletion,
  formal output creation, HP-4b retry, MuJoCo, collection, or training.
- Owner: Gate 0B command/environment identity and raw preflight artifact owner.
- Core path: identity environment -> nested uv -> provider Python -> frozen
  source import -> `scripts/train_distill.py --help`.
- Evidence: dependency-provider package snapshot/hash, exact nested import
  oracle, exact nested no-training entrypoint exit, output absence, and identity
  verifier.
- Stop: persist E54 and return control before HP-4b.

Execution-env repair result: `PASS` by E54. r6 freezes the absolute uv engine,
provider venv/Python/package snapshot, cache, no-sync, frozen PYTHONPATH,
progress flag, and a new absent output root. Exact nested import and
`scripts/train_distill.py --help` both exit 0; training does not start. r4/r5
remain rejected candidates. Return control before HP-4b.

HP-4b r6 execution result: `BLOCKED` at order 1 by E55. The r6 environment and
E51 aggregate route work, but learner preflight rejects the 98-D role teacher
checkpoint against the generic composed `teacher.obs_dim=99`. The workflow YAML
owns a 98-D student override but no teacher override. Do not bypass the guard,
override the frozen command, retry, or run orders 2-8. A separately authorized
Hydra workflow teacher config-owner repair and subsequent refreeze are required.

Workflow teacher config-owner repair is user-authorized after E55:

- Scope: declare the 98-D top-level teacher contract in
  `conf/distill/workflow/g1_walk_stand.yaml`, add compose isolation coverage,
  validate both role specs/checkpoints, and refreeze source/execution identity.
- Non-scope: Python fallback, checkpoint-guard weakening, task YAML changes,
  r6 partial reuse/deletion, HP-4b, simulator, collection, or learner update.
- Owner: Hydra workflow configuration.
- Core path: generic 99-D teacher -> workflow 98-D overlay -> composed teacher
  spec -> both 98-D role checkpoint contracts.
- Evidence: RED/GREEN compose regression, generic-default isolation, two role
  task/checkpoint probes, deterministic source bundle, frozen nested preflight.
- Stop: persist E56 and return before HP-4b.

Teacher config-owner repair result: `PASS` by E56. The workflow now owns a
98-D teacher beside its 98-D student, generic 99-D behavior remains isolated,
and both real role checkpoint guards pass. r7 deterministic source and
executable identity passes frozen build/hash/XML/compose/teacher/nested and
313+8 tests. Output is absent and execution remains unauthorized; return before
HP-4b.

HP-4b r7 execution result: `BLOCKED` at order-1 acceptance by E57. The formal
legacy command exits 0 and persists complete aggregate/update/checkpoint/
metrics/cleanup evidence, but the external oracle wrongly requires scenario
labels in raw legacy role artifacts. Do not rerun order 1 or start orders 2-8.
A separately authorized oracle-owner repair must validate raw role semantics,
native transition fields, and aggregate scenario identity, then freeze/hash the
oracle and apply it to existing order-1 artifacts before any resume decision.

Acceptance-oracle owner repair is user-authorized after E57:

- Scope: implement and hash a kind-aware external oracle, bind it to r7
  identity, and apply it to existing order-1 artifacts without rerunning.
- Non-scope: UniLab source/config edits, r7 artifact mutation, order-1 rerun,
  order-2 execution, or A/B conclusions.
- Owner: HP-4b experiment acceptance owner.
- Core path: artifact kind -> raw role/native transition checks -> aggregate
  scenario checks -> checkpoint/metrics/cleanup -> acceptance JSON.
- Evidence: frozen oracle hash, acceptance contract hash, existing artifact
  before/after hashes, successful order-1 acceptance, order-2 absence.
- Stop: persist E58 and return before order 2.

Acceptance-oracle repair result: `PASS` by E58. Oracle v2 and its contract are
frozen and bound to r7 identity. It accepts existing order 1 while proving all
seven tracked training artifacts unchanged and `training_rerun=false`. Order 2
remains absent. Return control before any resume.

#### HP-4b Resume Result: r7 Order 2 Blocked

- Objective: resume the frozen sequence at order 2 and require oracle v2 after
  every successful run; never rerun order 1.
- Executed scope: fresh immutable preflight, persistent order 2, and immediate
  stop at the first nonzero training exit. Orders 3-8 and oracle invocation for
  the failed run are outside the executed scope.
- Runtime fact: shared-memory-enabled execution reaches the spawned G1 worker
  and first scenario collection, then fails because the workflow-owned
  `datasets/dagger_iteration_1` parent does not exist before artifact save.
- Owner boundary: `workflow.py` owns iteration output materialization. Do not
  repair this in `scripts/train_distill.py` or as a persistent-worker fallback.
- Required next evidence: a focused regression proving the parent exists before
  persistent dispatch, affected workflow/runtime tests, a new immutable source
  and output identity, and a no-training frozen preflight.
- Stop: E59 is `BLOCKED`; return control before source modification, refreeze,
  order-2 retry, HP-4c, or HP-5.

#### HP-4b Workflow Materialization Repair (Authorized)

Step 1 / 2:

- Objective: prove and repair the workflow-owned iteration-directory boundary.
- Scope: change the spawned persistent workflow fake so it does not create its
  own parent; prove RED; materialize `iteration_dir` in `workflow.py`; prove
  focused GREEN and run the affected workflow/runtime/config suite.
- Non-scope: script, persistent-worker, dataset-owner, method-contract, teacher,
  checkpoint, workload, oracle, or formal-run changes.
- Owner files/modules: `src/unilab/algos/torch/distill/workflow.py` and
  `tests/algos/test_distill_workflow.py`.
- Expected evidence: the focused test fails on the absent parent before the
  source fix, then passes after one workflow-owner materialization change;
  affected tests and Ruff pass.
- Stop condition: source/test diff is bounded and reviewed before refreeze.

Step 2 / 2:

- Objective: freeze the repaired source and a new empty formal output identity.
- Scope: deterministic source bundle, r8 identity, frozen cwd, exact provider/
  import/Hydra/teacher/compose/test preflight, and frozen oracle v2 binding.
- Non-scope: order-2 retry or any A/B execution.
- Owner files/modules: external Gate 0B run artifacts plus current governance
  evidence/checklists/Architecture.
- Expected evidence: two deterministic source inventories/bundle hashes match;
  build/import/config/assets/teacher/tests pass from frozen cwd; new output root
  is absent.
- Stop condition: persist the new identity and return control before execution.

Repair/refreeze result: `PASS` by E60. The no-mkdir spawned regression proves
RED then GREEN at the workflow owner, the affected suite reports 493 passed,
and deterministic r8 source/output/oracle identity passes the complete
no-training preflight. The r8 formal output root remains absent. Return control
before any A/B command; a future execution starts at r8 order 1.

#### HP-4b r8 Formal A/B Execution (Authorized)

- Objective: execute the frozen eight-run r8 sequence from order 1 and require
  frozen oracle v2 acceptance after every successful run.
- Scope: exact r8 cwd/argv/environment/order/output identity, per-run raw log,
  per-run oracle log and acceptance, then bounded timing analysis only if all
  eight runs pass.
- Non-scope: source/config/workload/oracle mutation, r7 artifact reuse, failed
  run continuation, HP-4c bottleneck selection, or HP-5 optimization.
- Owner files/modules: frozen formal entrypoint and workflow/runtime owners;
  r8 output and execution-log artifacts; governance evidence is derived after
  runtime completion.
- Expected evidence: eight exit-zero runs, eight `accepted=true` oracle files,
  exact route/repetition order, complete metrics/cleanup/checkpoint/artifact
  identities, and four comparable repetitions per route.
- Stop condition: first command/oracle failure stops later orders; otherwise
  persist the HP-4b A/B result and return before HP-4c/HP-5.

r8 execution result: `PASS` for eight-run execution, oracle, semantics,
lifecycle, and timing-artifact completeness; `PARTIAL` for stable end-to-end
speedup by E61. Persistent request collection median is lower, but complete
e2e median is higher and paired ratios cross 1. Return control before HP-4c or
HP-5; do not infer a bottleneck owner from the residual alone.

#### HP-4c Bottleneck Verdict (Authorized)

- Objective: determine whether cleanup or persistent request residual is a
  stable, recurring, owner-attributable bottleneck that justifies one HP-5
  change.
- Scope: read-only decomposition of E61 raw records by run/scenario/stage;
  absolute time, e2e share, range, stdev/CV; source timer and lifecycle trace;
  one-time versus recurring classification.
- Non-scope: new run, timer instrumentation, source change, optimization,
  server-scale claim, or HP-5 implementation.
- Owner files/modules: E61 analysis/metrics plus `g1_persistent_worker.py`,
  `persistent_resources.py`, `persistent_runtime.py`, `async_runtime.py`,
  `workflow.py`, and the formal entrypoint lifecycle finalizer.
- Expected evidence: name one stable recurring owner only if the raw interval
  and source boundary agree; otherwise record one smallest additional
  discriminator and keep HP-5 closed.
- Stop condition: persist the HP-4c verdict and return control before any new
  measurement or implementation.

HP-4c result: `PASS`, verdict `NO_HP5_OWNER` by E62. Cleanup and resource
construction are stable but once-per-invocation/worker cold costs; the confirmed
warm cache-hit residual is about 2.25 ms. The smallest next discriminator is
one newly frozen paired two-outer-iteration run with iteration-aware acceptance.
Return control before that experiment or any HP-5 change.

#### HP-4c Two-Iteration Amortization Discriminator (Authorized)

The work is split at the governance freeze, oracle/preflight, live execution,
and read-only verdict boundaries because each has an independent stop condition
and evidence class. The r8 source bundle and frozen cwd remain the exact source
owner; r9 is a new workload/output identity only.

Step 1 / 4:
- Objective: freeze one r9 legacy/persistent pair whose only workload change
  from r8 is `training.workflow.dagger_iterations=2`.
- Scope: r8 source bundle/cwd/provider/assets; two-run order
  `legacy_r1 -> persistent_r1`; new empty output root; new identity and hashes.
- Non-scope: source/config edits, new timers, HP-5, policy-quality acceptance,
  or reuse of the already executed r8 output identity.
- Owner files/modules: governance plan/checklists/task canvas and external r9
  immutable experiment artifacts.
- Expected evidence: compose diff limited to execution mode/run dir, all r8
  source/provider/asset hashes match, output root absent, identity frozen.
- Stop condition: any unapproved drift blocks before oracle or training;
  otherwise emit the Step 1 End Report and start Step 2.

Step 2 / 4:
- Objective: freeze an iteration-aware acceptance oracle and execute the exact
  no-training preflight.
- Scope: two checkpoint lineages, two ordered scenario sets, cumulative
  aggregate sizes, per-iteration timing identities, persistent weight versions
  `1 -> 2`, six requests/resets, and final cleanup.
- Non-scope: accepting a one-iteration manifest, changing formal source, or
  inferring timing performance from preflight.
- Owner files/modules: external r9 oracle/contract/preflight artifacts; frozen
  UniLab dataset loader used read-only.
- Expected evidence: oracle syntax/import pass, deterministic oracle hash,
  exact nested frozen-source import/help, compose/teacher/provider/source checks,
  and `training_started=false`.
- Stop condition: first failed assertion blocks the live pair; otherwise emit
  the Step 2 End Report and start Step 3.

Step 3 / 4:
- Objective: execute the frozen pair in order and require the frozen oracle
  after each successful command.
- Scope: r9 exact cwd/argv/environment/order/output identity, raw logs, and two
  acceptance artifacts.
- Non-scope: rerun-on-failure, continuing after oracle failure, code mutation,
  or HP-5 optimization.
- Owner files/modules: frozen formal entrypoint/workflow/runtime owners and r9
  output/execution artifacts.
- Expected evidence: two exit-zero runs; two `accepted=true` oracle files; two
  complete outer iterations per run; exact checkpoint/aggregate/lifecycle facts.
- Stop condition: stop immediately at the first command/oracle failure;
  otherwise emit the Step 3 End Report and start Step 4.

Step 3 result: `BLOCKED` by E63 at order-1 acceptance. Legacy training exits
zero and completes both iterations, but oracle v3 assumes a persistent-only
manifest key must exist in legacy and assumes the replay-expanded update budget
stays at 16 instead of the observed valid `16 -> 24`. A diagnostic-only oracle
accepts the immutable order-1 artifacts. Persistent order 2 did not start. The
next bounded step is a separately authorized versioned oracle v4/amendment;
do not rerun order 1 or continue order 2 automatically.

#### HP-4c Oracle v4 Acceptance Amendment (Authorized)

This is one bounded secondary-contract step.

- Objective: replace only the rejected v3 acceptance assumptions with a
  versioned oracle v4/amendment and formally accept the immutable existing r9
  legacy order-1 artifacts.
- Scope: snapshot order-1 artifact hashes; freeze oracle v4; bind an amendment
  to r9 identity and v3 oracle/contract; validate optional legacy weight-version
  fields and replay-expanded updates `16 -> 24`; write acceptance and an
  unchanged-artifact attestation.
- Non-scope: UniLab source/config changes, r9 identity mutation, v3 mutation,
  legacy training rerun, persistent order 2, timing comparison, or HP-5.
- Owner files/modules: external experiment acceptance artifacts under the r9
  freeze directory; existing r9 order-1 manifest/datasets/checkpoints/metrics
  are read-only fixtures.
- Core parameter path: execution mode -> optional manifest/artifact weight
  version -> timing identity version; configured update floor -> cumulative
  aggregate/replay requirement -> actual update count.
- Test class: secondary contract path using the real completed order-1
  artifacts; semantic asserts are sufficient because no runtime lifecycle is
  changed or executed.
- Expected evidence: oracle v4 exits zero with `accepted=true`; all tracked
  training artifact hashes match before/after; `training_rerun=false`; order-2
  output/log/acceptance remain absent.
- Stop condition: persist the amendment, acceptance, attestation, and E64; then
  return control before persistent order 2.

Oracle v4 amendment result: `PASS` by E64. Frozen v4 accepts the immutable
existing legacy order 1 with actual updates `16 -> 24`; all 16 training files
are byte-identical before/after and `training_rerun=false`. Persistent order-2
output/log/acceptance remain absent. Return control before order 2; a future
resume must use the frozen v4 amendment and must not rerun order 1.

#### HP-4c r9 Persistent Order-2 Resume (Authorized)

Step 1 / 3:
- Objective: freeze an exact resume preflight around immutable accepted order 1
  and the absent persistent order-2 boundary.
- Scope: r9 identity, oracle v4/amendment, order-1 acceptance/attestation and
  recursive snapshot, r8 source/provider/assets/compose identity, order-2
  output/log/acceptance absence.
- Non-scope: training, order-1 mutation/rerun, source/config edits, or timing
  conclusions.
- Owner files/modules: external resume-preflight artifact under the r9 freeze
  directory; frozen source and order-1 artifacts are read-only.
- Expected evidence: exact hashes match, nested frozen import/help pass,
  provider entries match after order normalization, compose is persistent r9,
  and `training_started=false`.
- Stop condition: any mismatch blocks before training; otherwise emit Step 1
  End Report and start Step 2.

Step 2 / 3:
- Objective: execute persistent order 2 only and require frozen oracle v4
  immediately after a successful training command.
- Scope: exact r9 persistent argv/environment/cwd/output, one raw log, one v4
  acceptance, and complete persistent two-iteration lifecycle evidence.
- Non-scope: legacy order-1 rerun, rerun-on-failure, source changes, or HP-5.
- Owner files/modules: frozen formal entrypoint/workflow/persistent runtime and
  r9 persistent output/execution artifacts.
- Core parameter path: checkpoint activation versions `1 -> 2` -> six scenario
  requests/resets -> cumulative cache counters -> one final cleanup.
- Test class: live sentinel; compact manifest/metrics/oracle facts are sufficient
  because the real worker lifecycle is executed.
- Expected evidence: train exit zero; oracle v4 `accepted=true`; 55 timing
  records; versions `1 -> 2`; final counters 6 requests, 6 resets, 6 cache hits,
  2 resource init/close, 0 errors.
- Stop condition: first train/oracle failure stops; otherwise emit Step 2 End
  Report and start Step 3.

Step 3 / 3:
- Objective: complete the read-only iteration-aware amortization verdict using
  the accepted legacy/persistent pair.
- Scope: per-route/per-iteration request/workflow totals, persistent cache
  progression, final cleanup and per-iteration amortized cleanup, process
  elapsed time, and explicit single-pair limitations.
- Non-scope: stable production speedup claim, policy quality, code optimization,
  or automatic HP-5 authorization.
- Owner files/modules: accepted r9 metrics/manifests and derived evidence JSON.
- Expected evidence: a reproducible table and one verdict among
  `AMORTIZATION_CONFIRMED`, `NOT_DISTINGUISHED`, or `NEW_OWNER_CANDIDATE`.
- Stop condition: persist the verdict and E65, synchronize current governance,
  and return control before HP-5.

r9 resume result: `PASS` by E65. Persistent order 2 exits zero and oracle v4
accepts both iterations; legacy order 1 is unchanged and was not rerun. The
read-only verdict is `AMORTIZATION_CONFIRMED`: persistent request total/residual
fall 52.45%/95.97% in iteration 2 with four new cache hits and no new resource
init; cleanup occurs once and amortizes across two iterations. Full persistent
process time is nevertheless 20.64% slower than legacy in this single pair,
and learner work grows `16 -> 24`. No stable speedup or HP-5 owner is accepted;
return control before HP-5.

#### HP-4c r10 Multi-Repetition Two-Iteration Freeze (Authorized)

This is a freeze-only secondary-contract step. Formal execution is a separate
live gate.

- Objective: freeze a statistically interpretable repeated two-iteration
  legacy/persistent benchmark without changing source or workload semantics.
- Scope: exact r8 source/provider/assets; r9 `dagger_iterations=2` workload;
  four repetitions per route; r8 balanced order
  `L1,P1,P2,L2,L3,P3,P4,L4`; eight per-order run-dir/compose hashes; oracle v4
  semantics; new empty r10 output root; exact no-training preflight.
- Non-scope: any training command, reuse/overwrite of r8/r9 outputs, source or
  config edits, performance conclusion, HP-5, or default-on promotion.
- Owner files/modules: external r10 identity/oracle/contract/preflight artifacts;
  frozen UniLab source is read-only.
- Core parameter path: shared two-iteration workload -> balanced route/order and
  repetition identity -> per-order run directory -> Hydra compose hash -> oracle
  acceptance output.
- Test class: secondary contract path. Eight semantic compose assertions and
  immutable hash checks are sufficient because training is not executed.
- Expected evidence: four legacy plus four persistent orders; each exact compose
  hash frozen; normalized shared config identical; source/provider/assets/
  teacher/import/help contracts pass; output root absent; `training_started=false`.
- Stop condition: persist E66 and return before order 1. Any benchmark execution
  requires separate authorization and must stop on first train/oracle failure.

r10 freeze result: `PASS` by E66. Identity, eight exact compose hashes, oracle
v4 contract, primary-metric decision rule, and complete no-training preflight
are frozen. Formal output and execution logs remain absent. Return control
before order 1; HP-5 and default-on promotion remain closed.

#### HP-4c r10 Eight-Run Execution (Authorized)

- Objective: execute the exact frozen r10 benchmark and apply the pre-registered
  primary decision rule only after all eight orders pass frozen oracle v4.
- Scope: fresh immutable identity/preflight/output-absence checks; balanced order
  `L1,P1,P2,L2,L3,P3,P4,L4`; one raw log and one oracle-v4 acceptance per
  successful order; read-only paired process-time analysis after 8/8 acceptance.
- Non-scope: source/config/oracle/workload mutation, retrying a failed order,
  continuing after failure, HP-5 implementation, policy-quality acceptance, or
  default-on promotion.
- Owner files/modules: frozen r10 entrypoint/workflow/runtime, r10 execution and
  acceptance artifacts, and governance evidence derived from those artifacts.
- Core parameter path: frozen order/repetition identity -> exact Hydra compose ->
  two-iteration run -> oracle v4 -> paired repetition ratio -> frozen median and
  direction rule.
- Test class: S4 live sentinel plus T-persist/T-oracle/T-diff. The first command
  or oracle failure stops all later orders and returns control.
- Expected evidence: eight exit-zero commands, eight `accepted=true` oracle
  artifacts, exact frozen hashes/order, and a decision of either
  `STABLE_DIRECTION_SPEEDUP` or `NO_STABLE_SPEEDUP`; secondary lifecycle timing
  may explain but cannot override the primary decision.
- Stop condition: persist E67 and return before HP-5/default-on. A failure instead
  records E67 as blocked with the first failing boundary and no later execution.

r10 execution result: `PASS` for 8/8 command and oracle acceptance by E67; the
pre-registered primary verdict is `NO_STABLE_SPEEDUP`. Legacy/persistent medians
are 2.286095/2.891427 s, median paired ratio is 1.264792, and only 1/4 ratios is
below 1. Cache/residual amortization repeats but cannot override the primary
metric. No HP-5 owner or default-on promotion is authorized.

Step 4 / 4:
- Objective: determine whether iteration 1 -> 2 directly demonstrates
  cache/cleanup amortization and whether a recurring owner becomes eligible.
- Scope: per-route/per-iteration request and workflow timings, cache counters,
  one final cleanup divided over two iterations, absolute/share comparison, and
  explicit one-pair limitations.
- Non-scope: stable throughput claim, policy quality, source edits, or automatic
  HP-5 authorization.
- Owner files/modules: r9 metrics/manifests/acceptance artifacts and governance
  evidence only.
- Expected evidence: iteration-aware timing table and a falsifiable verdict
  (`AMORTIZATION_CONFIRMED`, `NOT_DISTINGUISHED`, or `NEW_OWNER_CANDIDATE`).
- Stop condition: persist the verdict, synchronize current evidence/checklists,
  and return control before HP-5.

#### HP-4c Step 3/3: Bottleneck Verdict

- Objective: determine whether one measured owner stage justifies HP-5 and
  select only that owner boundary.
- Scope: compare cold and steady-state results, compute per-stage shares and
  legacy/persistent ratios, separate one-time initialization from per-row
  cost, and record uncertainty/variance and the dominant measured stage.
- Non-scope: editing code, choosing an optimization from intuition, collapsing
  several bottlenecks into one claim, or treating policy quality as a
  performance metric.
- Owner files/modules: HP-4 evidence artifact and checklist; source modules are
  read-only during the verdict step.
- Expected evidence: S3/T-diff/T-scale analysis linked to every raw HP-4b run
  identity. The verdict must name the stage, its owner, absolute time,
  end-to-end share, variance, and the smallest proposed HP-5 change.
- Stop condition: if no stable dominant stage exists, do not enter HP-5;
  propose the smallest additional discriminator instead. Emit a Step 3/3 End
  Report; if a stable bottleneck exists, return control to the user to
  authorize one HP-5 branch.

### HP-5: Evidence-Gated Data-Path Optimization

- If cumulative materialization dominates, replace monolithic cumulative copy
  with immutable shards plus a manifest-backed dataset view.
- If collector transfer/packing dominates, propose a generic schema-aware
  shared batch channel owned by `src/unilab/ipc/`, then add DAgger schema as a
  client. Do not specialize APPO/off-policy storage in place.
- If learner updates dominate, optimize sampler/device staging inside the
  offline learner owner.
- Stop condition: one measured bottleneck, one owner-layer change, and an
  OFF-path regression gate per change.

### HP-6: Production Gate

- Run focused tests, affected distillation suite, IPC suite, config tests,
  Ruff, and one bounded formal workflow.
- Compare legacy and persistent routes for row schema, scenario counts,
  checkpoint lineage, and outer-barrier identity.
- Only after those pass may `persistent_async` be considered for default-on.
- `make test-all` remains required before PR creation.

#### HP-6a Production Readiness Gate (Authorized)

- Objective: determine whether the cumulative OFF-default runtime integration
  is ready to pay for the repository-wide production gate.
- Scope: read-only owner/default/lifecycle/lineage diff review; affected
  distillation, script/config, and IPC suites; targeted Ruff; Architecture
  atlas contract checks.
- Non-scope: source repair, new training/benchmark, `make test-all`, contract
  activation, default-on, commit, PR, or HP-5 optimization.
- Owner files/modules: existing Hydra/script/workflow/collector/offline/runtime
  owners and their current tests; governance documents record E68.
- Core parameter path: Hydra legacy default -> script route selection ->
  workflow outer barrier -> checkpoint/weight version -> scenario artifacts ->
  update/checkpoint -> metrics and cleanup.
- Test class: S1/S2/S3 secondary contract path. E67 already supplies the S4
  live sentinel, so this step needs no additional training.
- Commands: affected `pytest` suites, `uv run ruff check` on changed Python,
  and `npm run check` in the atlas helper.
- Expected result: no owner leakage or default drift; every command exits zero.
- Stop condition: first review/test failure records E68 `BLOCKED` and returns
  control. If all pass, record E68 `PASS` and return before HP-6b `make test-all`
  or any contract/default/PR decision.

HP-6a result: `BLOCKED` by E68 at the first read-only review finding.
`async_runtime.py` and `performance.py` retain pre-HP-4 audit-status claims that
live timing and A/B are absent, contradicting E61/E65/E67. Per the declared
stop condition, no affected test, Ruff, or atlas command ran. Source repair was
outside the authorized step; return control before changing the docstrings or
restarting HP-6a.

#### HP-6a1 Runtime Audit-Status Repair (Authorized)

- Objective: remove the E68 source-level evidence contradiction without
  changing runtime behavior.
- Scope: update the module audit-status text in `async_runtime.py` and
  `performance.py`; synchronize the equivalent current Method-to-Code evidence
  gap found during the pre-edit search; search affected runtime, Architecture,
  and current governance surfaces again; run compile, targeted Ruff, and atlas
  checks.
- Non-scope: executable logic, tests, config, contract activation, training,
  benchmark, `make test-all`, default-on, commit, or PR.
- Owner files/modules: the two runtime-owner module docstrings and the existing
  Contracts & Sentinels Method-to-Code card; governance records E69. Concept
  Figure, active contract, owners, and runtime routing remain unchanged.
- Core parameter path: source audit-status -> current E61/E65/E67 evidence
  boundary. No tensor or runtime value changes.
- Test class: S0/S1 secondary contract path; textual assertions plus compile,
  Ruff, and atlas source/contract checks are sufficient.
- Expected result: both modules say A/B/timing are runtime-confirmed, retain
  legacy OFF-default and `NO_STABLE_SPEEDUP`, and name production/physical
  acceptance as the real remaining gap.
- Stop condition: any equivalent stale source claim or verification failure
  blocks before HP-6a restart; otherwise record E69 PASS and continue to the
  separately authorized restart.

HP-6a1 result: `PASS` by E69. Two module audit-status docstrings and the
equivalent current Method-to-Code performance gaps now reflect E61/E65/E67,
`NO_STABLE_SPEEDUP`, OFF-default, and pending HP-6/physical acceptance. The
structured stale assertion is empty, both modules compile, targeted Ruff
passes, and atlas contracts pass. No executable behavior changed.

E70 amendment: E69 is `PARTIAL` for whole-Architecture coverage. Its assertion
missed the equivalent phrase `尚缺`, leaving U-RT-06/U-RT-08 stale. The source
docstring and Method-to-Code repair remains accepted; Runtime Atlas consistency
requires a separate bounded repair.

#### HP-6a Production Readiness Restart (Authorized)

- Objective: restart the E68-blocked gate after E69 and determine readiness for
  the later repository-wide HP-6b gate.
- Scope: repeat owner/default/lifecycle/lineage review, then run affected
  distillation, script/config, and IPC tests plus targeted Ruff and atlas checks.
- Non-scope: source repair discovered during review, training/benchmark,
  `make test-all`, v003 activation, default-on, commit, or PR.
- Owner files/modules: cumulative runtime integration and affected tests;
  governance records E70.
- Core parameter path: Hydra `legacy` default -> script selection -> workflow
  barrier -> checkpoint/weight version -> artifacts/update -> metrics/cleanup.
- Test class: S1/S2/S3 secondary contract path backed by E67 S4 live evidence.
- Expected result: structured owner probe and all affected commands exit zero.
- Stop condition: first review/test failure records E70 BLOCKED; all-pass
  records E70 PASS and returns before HP-6b or contract/default decisions.

HP-6a restart result: `BLOCKED` by E70 at cross-file consistency. The owner
probe passes; affected suites report 137 + 326 + 74 = 537 passed and 24 skipped;
targeted Ruff passes. Runtime Atlas U-RT-06/U-RT-08 still claim timing/A/B are
missing, contradicting E61/E65/E67. Source repair is outside E70 scope; return
control before editing the atlas or entering HP-6b.

#### HP-6a2 Runtime Atlas Status Repair (Authorized)

- Objective: repair the last E70 current-state contradiction and make its
  semantic class fail closed in the existing atlas checker.
- Scope: update U-RT-06/U-RT-08 timing/A/B gaps; add a durable checker assertion
  covering equivalent `尚缺`, `尚未`, `未连接`, `未执行`, and `absent` timing/A/B
  claims; run RED before data repair, then atlas and cross-file GREEN checks.
- Non-scope: executable training source/config/tests, Concept Figure, active
  contract, new training/benchmark, repeated E70 affected suites, `make
  test-all`, v003 activation, default-on, commit, or PR.
- Owner files/modules: Runtime Atlas current-state JSON and its existing
  `check_distillation_atlas.mjs`; governance records E71.
- Core parameter path: E61/E65/E67 evidence -> U-RT-06/U-RT-08 gap text ->
  semantic stale assertion -> atlas acceptance.
- Test class: S0/S1 secondary contract path. Deterministic RED/GREEN semantic
  assertions are sufficient because executable source is unchanged and E70
  already reports 537 passed/24 skipped.
- Expected result: checker fails on the old `尚缺` text, then passes only when
  both cards include E67, `NO_STABLE_SPEEDUP`, and pending HP-6 without stale
  timing/A/B absence claims.
- Stop condition: first unexpected RED/GREEN or consistency failure blocks;
  all-pass records E71 and restores HP-6a readiness PASS before returning at
  the HP-6b decision boundary.

HP-6a2 result: `PASS` by E71. The durable checker fails on the old U-RT-06
`A/B 尚缺` text and passes after U-RT-06/U-RT-08 record E67,
`NO_STABLE_SPEEDUP`, and pending HP-6. Cross-file current-atlas hits are empty,
registry/Concept mapping checks pass, and `git diff --check` passes. Combined
with E70's 537 passed/24 skipped and targeted Ruff, HP-6a readiness is PASS.
Return before HP-6b, contract activation, default-on, commit, or PR.

#### HP-6b Repository-Wide Production Gate (Authorized)

- Objective: execute the repository-required pre-PR full sweep on the cumulative
  OFF-default runtime-integration worktree.
- Scope: run exact `make test-all`, which expands to Ruff format/fix, mypy,
  pyright, and the complete non-slow pytest suite with coverage; inspect any
  formatter/fix diff and classify the first failure by owner.
- Non-scope: manual source repair after failure, slow tests, new live training
  or benchmark, v003 activation, default-on, commit, push, or PR.
- Owner files/modules: repository Makefile and all modules/tests selected by its
  standard gate; governance records E72.
- Core parameter path: repository source/config/test graph -> format/lint ->
  static type owners -> non-slow test collection -> coverage result.
- Test class: full-sweep S0/S1/S2/S3 with T kinds assigned by the repository
  inventory. It does not replace E67 S4 timing evidence or E28 physical-policy
  failure evidence.
- Command: `make test-all` from `/private/tmp/unilab-dagger-mainline` with the
  existing project environment and shared-memory permission required by IPC
  tests.
- Expected result: every make subtarget exits zero; any mechanical format/fix
  change is reviewed and remains semantics-free.
- Stop condition: first failing subtarget records E72 BLOCKED without manual
  repair. An all-pass result records E72 PASS and returns before contract,
  default, commit, or PR decisions.

HP-6b result: `BLOCKED` by E72 at the Makefile `format` target. Ruff format
reformatted 57 files; Ruff safe-fix corrected 15 of 17 findings; two F841 dead
assignments remain in
`scripts/deploy/check_robojudo_unilab_section8_runtime_torque.py:381-382`.
Mypy, pyright, and coverage pytest did not start. No manual repair was made;
return control before removing the variables or rerunning `make test-all`.

#### HP-6b1 Repository Lint-Owner Repair (Authorized)

- Objective: remove only the two E72-proven dead assignments that block the
  repository Ruff gate.
- Scope: delete `last_action` and `gait_phase` assignments from the section-8
  diagnostic `main()`; run targeted py_compile and Ruff on that file.
- Non-scope: change helper rollout state, diagnostic behavior, DAgger code,
  formatter-wide cleanup, test logic, contract/default/commit/PR action.
- Owner file/module:
  `scripts/deploy/check_robojudo_unilab_section8_runtime_torque.py` diagnostic
  entrypoint; governance records E73.
- Core parameter path: two `main()` local assignments -> no read/consumer ->
  Ruff F841. The same names inside helper rollout loops remain untouched.
- Test class: S0 secondary contract path (`py_compile`, targeted Ruff, lexical
  scope assertion). The existing Ruff RED is the regression oracle.
- Expected result: both F841 findings disappear and helper-local state remains.
- Stop condition: any additional targeted finding blocks before diff review or
  full rerun; otherwise record E73 PASS and continue to authorized E74.

HP-6b1 result: `PASS` by E73. Only the two dead `main()` assignments were
removed; targeted compile and Ruff pass; AST ownership proves both helper
rollout functions retain their `last_action` and `gait_phase` state. Continue
to the authorized E74 diff review before a full rerun.

#### HP-6b2 Mechanical Diff Review and Full Rerun (Authorized)

- Objective: establish that E72 formatter/auto-fix mutations are mechanical or
  explicitly safe, then rerun the exact repository production gate.
- Scope: compare E72-before-clean tracked files against HEAD by AST for files
  newly touched by the formatter; inspect every non-AST-equivalent safe-fix;
  rerun exact `make test-all` from the beginning.
- Non-scope: manual repair after a new failure, slow/S4 tests, v003 activation,
  default-on, commit, push, or PR.
- Owner files/modules: E72 formatter-expanded tracked Python surface and
  repository Makefile; governance records E74.
- Core parameter path: pre-E72 clean file -> formatter/safe fix -> AST or
  explicit token-level delta -> format/lint/type/non-slow coverage gate.
- Test class: S0 diff/type plus full-sweep S1/S2/S3. E67/E28 remain the separate
  S4 evidence boundary.
- Expected result: pure-format files are AST-equivalent; every safe-fix delta is
  enumerated and semantics-free; all Makefile subtargets exit zero.
- Stop condition: first unexplained diff or failed subtarget records E74
  BLOCKED; all-pass records E74 PASS and returns before activation/default/PR.

HP-6b2 result: `BLOCKED` by E74 at mypy. Mechanical review passes: 429 Python
files are AST-equivalent to the r8 frozen source and the only AST delta is the
two E73 dead assignments removed from the section-8 diagnostic. The rerun then
passes format/Ruff but mypy reports 20 errors in 8 files: 7 errors in four
branch-owned distillation runtime files and 13 errors in four HEAD-identical
baseline files. Pyright and coverage pytest did not start. Return before type
repair or another full rerun.

#### HP-6b3 Branch-Owned Type Repair (Authorized)

- Objective: repair the seven E74 mypy errors introduced on the DAgger runtime
  integration surface without changing runtime semantics.
- Scope: `collector.py`, `async_runtime.py`, `workflow.py`, and
  `g1_persistent_worker.py`; use explicit narrowing/Literal validation and
  existing fail-closed invariants; run four-file mypy/Ruff and affected tests.
- Non-scope: baseline files, `type: ignore`, broad `Any` casts, behavior/config/
  contract/default changes, full rerun before local PASS, commit, or PR.
- Owner files/modules: collector metrics accumulator, async request/result
  protocol, workflow iteration/result union, G1 teacher-spec adapter;
  governance records E75.
- Core parameter paths: optional accumulator -> indexed observation;
  queue payload -> validated result; manifest field mapping -> typed identity;
  updater int/result union -> normalized result; Hydra algo string -> SAC
  Literal contract.
- Test class: S0 mypy/Ruff plus S1/S2 affected contract/connectivity tests.
- Expected result: zero mypy errors in these four files and no semantic test
  regression.
- Stop condition: first unresolved type or affected-test failure blocks before
  baseline repair; otherwise record E75 PASS and continue to E76.

#### HP-6b4 HEAD-Baseline Type Repair (Authorized)

E75 result: `PASS`. The seven branch-owned errors are repaired with local
narrowing/fail-closed validation; targeted Ruff passes and affected suites report
111 passed. Evidence: `2026-07-17-hp6b3-branch-type-repair.md`.

- Objective: repair the thirteen E74 mypy errors in four files AST-identical to
  HEAD, using local type owners rather than weakening the repository gate.
- Scope: `models.py`, `playback.py`, `data.py`, and G1 `joystick.py`; preserve
  runtime validation and add explicit type narrowing/return typing; run targeted
  mypy/Ruff and local model/data/playback/G1 tests.
- Non-scope: DAgger owner files, type ignores, semantic redesign, S4 physics,
  full rerun before local PASS, contract/default/commit/PR actions.
- Owner files/modules: lazy model symbol loader, playback routing-mode adapter,
  dataset optional tensor/metadata validation, gait-constraint config accessor;
  governance records E76.
- Core parameter paths: lazy import symbol -> declared class type; config string
  -> routing Literal; optional dataset fields -> validated non-optional locals;
  optional metadata map -> narrowed map; config object -> declared return type.
- Test class: S0 mypy/Ruff plus S1/S2/S3 local semantic/persistence/playback
  contracts.
- Expected result: zero mypy errors in the four baseline files and local tests
  pass without weakening fail-closed behavior.
- Stop condition: first unresolved type/test failure blocks before full rerun;
  otherwise record E76 PASS and continue to E77.

#### HP-6b5 Final Repository Gate Rerun (Authorized)

E76 result: `PASS`. All thirteen baseline-owner errors are repaired; targeted
mypy/Ruff pass and local suites report 442 passed, 3 skipped. Evidence:
`2026-07-17-hp6b4-baseline-type-repair.md`.

E77 result: `BLOCKED` at Pyright after format, Ruff, and full mypy PASS. Pyright
reports six optional-flow diagnostics, all in `collector.py`; coverage pytest
did not start. Per stop-on-first-failure, no repair was attempted. Evidence:
`2026-07-17-hp6b5-pyright-blocked.md`.

#### HP-6b6 Collector Pyright Narrowing (Authorized)

E78 result: `PASS`. Targeted Pyright reports zero diagnostics, mypy/Ruff pass,
and direct collector suites report 86 passed. Evidence:
`2026-07-17-hp6b6-collector-pyright-narrowing.md`.

#### HP-6b7 Final Repository Gate Rerun (Authorized)

E79 result: `BLOCKED` at non-slow coverage pytest. Ruff, mypy, and Pyright pass;
pytest reports 14 failed, 1544 passed, 49 skipped, and 256 deselected. Ten G1
failures contradict the E76 gait-config accessor assumption; four other owner
failures remain unconfirmed. No repair was attempted. Evidence:
`2026-07-17-hp6b7-test-cov-blocked.md`.

#### HP-6b8 G1 Gait-Config Compatibility Owner Repair (Authorized)

E80 result: `PASS`. The exact ten E79 G1 failures pass; joystick mypy,
Pyright, and Ruff pass. Evidence:
`2026-07-17-hp6b8-g1-gait-config-compatibility.md`.

#### HP-6b9 Remaining Four-Failure Owner Diagnosis (Authorized)

E81 result: `PASS (diagnosis only)`. Stewart is an absent optional-provider /
test-selection issue; docs is HEAD-baseline generated support-matrix drift; CLI
is leakage of the frozen outer `UV_PROJECT_ENVIRONMENT` into a temporary-
checkout test. No repair was attempted. Evidence:
`2026-07-17-hp6b9-four-failure-owner-diagnosis.md`.

#### HP-6b10 Motrix Provider/Test-Selection Repair (Authorized)

- Objective: make Stewart runtime tests honor Motrix as an optional provider.
- Scope: Stewart test helper only; skip the two runtime/IK tests when
  `motrixsim` is unavailable while retaining static/config coverage; E82.
- Non-scope: install dependencies, change backend/env code, hide failures when
  Motrix is installed, docs/CLI, full rerun.
- Expected: Stewart module passes with exactly two provider-dependent skips.
- Stop: first targeted failure blocks before E83.

#### HP-6b11 Generated Support-Matrix Refresh (Authorized)

E82 result: `PASS`; Stewart reports 4 passed and 2 provider-dependent skips,
with Ruff PASS. Evidence: `2026-07-17-hp6b10-motrix-test-selection.md`.

- Objective: refresh the derived support matrix through its official owner.
- Scope: run `uv run scripts/generate_support_matrix.py --write`, inspect the
  two expected SAC rows, and run docs contract; E83.
- Non-scope: hand-edit generator output, registration changes, CLI/full rerun.
- Expected: generated diff only adds current registered facts and docs contract
  passes.
- Stop: unexpected generated change or docs failure blocks before E84.

#### HP-6b12 UV Project Environment Test Isolation (Authorized)

E83 result: `PASS`; official generation adds exactly the two expected SAC rows
and the docs contract passes. Evidence:
`2026-07-17-hp6b11-support-matrix-refresh.md`.

- Objective: isolate the temporary-checkout CLI test from the frozen outer uv
  environment without changing production environment precedence.
- Scope: the single CLI test fixture; clear `UV_PROJECT_ENVIRONMENT` before
  calling `run_demo`, run targeted CLI test/lint; E84.
- Non-scope: change `run_demo()` setdefault semantics, checkpoint routing,
  Makefile, environment identity, full rerun.
- Expected: targeted CLI test passes under the same externally frozen command
  environment.
- Stop: any production change requirement records E84 BLOCKED.

E84 result: `PASS`; the target passes under the externally frozen uv variable,
production code is unchanged, and Ruff/diff checks pass. Evidence:
`2026-07-17-hp6b12-uv-env-test-isolation.md`.

#### HP-6b13 Combined Fourteen-Regression Closure (Authorized)

E85 result: `PASS`; the combined process reports 12 passed and 2 expected
optional-provider skips. Evidence:
`2026-07-17-hp6b13-combined-fourteen-regression.md`.

#### HP-6b14 Final Repository Gate Rerun (Authorized)

E86 result: `PASS`. Formatter/Ruff/mypy/Pyright all pass; non-slow coverage
pytest reports 1556 passed, 51 skipped, 256 deselected, with 70% coverage.
Evidence: `2026-07-17-hp6b14-final-repository-gate.md`.

- Objective: run the exact complete `make test-all` after E85 PASS.
- Scope: formatter, Ruff, mypy, Pyright, and non-slow coverage pytest under the
  frozen isolated-worktree uv identity; E86.
- Non-scope: fixes after a new failure, slow/S4, Motrix installation, training,
  physical/performance claims, commit, push, or PR.
- Owner: repository Makefile.
- Test class: S0/S1/S2/S3 full sweep.
- Expected: every subtarget exits zero with exact totals.
- Stop condition: first failed subtarget records E86 BLOCKED; all-pass records
  HP-6b repository gate PASS and returns control.

- Objective: execute all fourteen E79 failure nodes together after E80-E84.
- Scope: ten G1 nodes, two Stewart nodes, one docs node, and one CLI node in a
  single pytest process under the frozen uv environment; E85.
- Non-scope: code/doc/test changes, Motrix installation, slow/S4, full repo
  rerun, training, commit, push, or PR.
- Owners: G1 compatibility, Stewart provider selection, generated docs, CLI
  env isolation.
- Test class: S1/S2/S3 combined regression contract.
- Expected: 12 passed and 2 explicit optional-provider skips.
- Stop condition: any unexpected failure records E85 BLOCKED; expected totals
  record E85 PASS and return before the full repository gate.

- Objective: reproduce and classify the two Stewart, one docs, and one CLI
  E79 failures without repairing them.
- Scope: four exact pytest node IDs, traceback/source/HEAD-diff inspection, and
  owner/branch-causality classification; governance records E81.
- Non-scope: implementation/test/dependency/environment changes, reruns beyond
  the four nodes, full repository gate, training, commit, push, or PR.
- Owners: Stewart env/import, documentation contract checker, CLI/demo local
  checkpoint resolver.
- Core paths: fixture -> constructor/checker/resolver -> first failed boundary.
- Test class: S0/S1/S2 secondary contract diagnostic.
- Expected result: exact symptom, owner, and branch-causality evidence for all
  four failures.
- Stop condition: record E81 classification and return before any repair.

- Objective: restore the G1 observation compatibility contract contradicted by
  E79 while retaining the E76 static return type.
- Scope: `_gait_constraint_cfg()` only, plus the ten existing E79 G1 regression
  cases and targeted type/lint gates; governance records E80.
- Non-scope: gait/reward semantics, observation dimensions, owner YAML edits,
  Stewart/docs/CLI, full repository rerun, training, commit, push, or PR.
- Owner: G1 reward-config accessor.
- Core path: missing field -> disabled default; dict -> dataclass; structured or
  fixture proxy -> duck-typed config; all -> gait-phase observation consumer.
- Test class: S0/S1/S2 compatibility contract path.
- Expected result: all ten G1 E79 failures pass and joystick mypy/Pyright/Ruff
  remain green.
- Stop condition: any required semantic/YAML change or targeted failure records
  E80 BLOCKED; all-pass records E80 PASS and returns before other owners.

- Objective: rerun the exact repository `make test-all` after E78 PASS.
- Scope: formatter, Ruff, mypy, Pyright, and non-slow coverage pytest from the
  current isolated worktree; governance records E79.
- Non-scope: repair after a new failure, slow/S4, training, physical quality,
  activation/default changes, commit, push, or PR.
- Owner: repository Makefile.
- Test class: S0/S1/S2/S3 full sweep.
- Expected result: every Makefile subtarget exits zero with exact totals.
- Stop condition: first failed subtarget records E79 BLOCKED; all-pass records
  HP-6b PASS and returns control.

- Objective: repair the six E77 optional-flow diagnostics without changing
  collection behavior.
- Scope: `collector.py` only; narrow the mutually exclusive rollout-policy
  contract and require a materialized action array before finite/max/env-step
  consumers; targeted Pyright/mypy/Ruff and collector contract tests.
- Non-scope: other owners, config/default/authority changes, full repository
  rerun, training, commit, push, or PR.
- Owner: distillation collector policy/action control flow; governance records
  E78.
- Core parameter paths: optional rollout policy -> `_policy_actions`; optional
  action array -> NumPy finite/max and `env.step`.
- Test class: S0/S1/S2 secondary contract path.
- Expected result: zero Pyright diagnostics in `collector.py`, mypy/Ruff pass,
  and collector-related tests pass.
- Stop condition: first unsupported narrowing or targeted failure records E78
  BLOCKED; all-pass records E78 PASS and returns before a new full rerun.

- Objective: rerun exact `make test-all` only after E75/E76 pass.
- Scope: standard format/Ruff/mypy/pyright/non-slow coverage gate and final
  formatter diff/cross-file review.
- Non-scope: manual repair after a new failure, slow/S4 tests, v003 activation,
  default-on, commit, push, or PR.
- Owner: repository Makefile; governance records E77.
- Test class: S0/S1/S2/S3 full sweep; E67/E28 remain S4 boundaries.
- Expected result: every Makefile subtarget exits zero with recorded test and
  coverage totals.
- Stop condition: first failed subtarget records E77 BLOCKED; all-pass records
  HP-6b PASS and returns before all later decisions.

## Current Execution Boundary

HP-0 through HP-3b2 pass their planned implementation and bounded lifecycle
gates. Gate 0A and HP-4a pass; E42 completes HP-4a2a and E43 completes the
HP-4a2b collector connector. E44 completes the three user-authorized HP-4a2c
owner/connector steps. E45 records the first blocked Gate 0B. E46 completes
measurement symmetry, and E47 reruns Gate 0B successfully with immutable
source/assets/workload/commands. Do not run HP-4b until separately authorized.
HP-4c is
a separate evidence-reading decision, and HP-5 cannot begin without its Step
End Report and explicit user authorization.
