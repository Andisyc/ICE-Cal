# Distillation Package Consolidation Design

## Status and accepted outcome

This design closes the repeated local-hotspot refactoring cycle by consolidating
the complete distillation, playback, and G1 environment maintenance surface in
one behavior-preserving engineering transaction.

The accepted outcome is not merely shorter functions. The resulting code must
have stable package-level responsibility boundaries, one owner for each mutable
state and durable artifact, compatibility at existing repository import and
entrypoint seams, and no remaining in-scope recommendation for another local
split.

The design baseline is the existing dirty working tree at commit
`41d37e957906fdbaaa73bca5ef1f77f3aa2efe73`. At design time it contains 80
modified or untracked paths. All pre-existing user changes are part of the
baseline and must be preserved. No branch creation, commit, push, destructive
cleanup, training, simulation, server execution, or deployment is authorized by
this design.

## Problem statement

Previous refactors selected files and long functions as isolated work units.
They successfully reduced several individual transactions, but left roughly 69
Python modules and 21.8k lines flat under `distill/`. The remaining problem is
therefore package ownership rather than isolated function length:

- dataset, collection, learning, workflow, runtime, FADA, and diagnostic owners
  live at the same package level;
- legacy import surfaces and production implementations are not visually or
  structurally distinct;
- repeated extraction can reduce a file while increasing navigation and caller
  knowledge across the package;
- playback composition remains distributed between a large script and a large
  compatibility module;
- `G1WalkEnv` still presents many framework bindings alongside lifecycle state,
  even where calculations already have focused owners.

Line count and function span remain selection signals, not independent reasons
to introduce abstractions.

## Preserved behavior and explicit non-scope

The consolidation must not change:

- model topology, tensor values, shapes, dtype, device, masks, gradient routes,
  optimizer ordering, update cardinality, or loss definitions;
- dataset fields, row/window order, scenario and role labels, filtering,
  serialization, replay balance, or batching behavior;
- same-state Oracle labeling, DAgger collection/update ordering, checkpoint
  activation, manifest commit-last ordering, resume, retry, or cleanup behavior;
- observation, reward, action, command, curriculum, domain-randomization, reset,
  latency, RNG consumption, or simulator/backend behavior;
- Hydra config names and values, registries, public call signatures, checkpoint
  schemas, import paths used within the repository, or CLI behavior;
- interactive playback checkpoint selection, session selection, keyboard
  control, viewer lifecycle, overlay, trace, and cleanup behavior.

The work does not claim training convergence, simulator correctness, checkpoint
quality, deployment readiness, or policy quality. It does not redesign FADA,
IDM, Planner, Oracle, SAC, reward, privileged observation, or curriculum
semantics.

## Considered approaches

### Rejected: continue local hotspot extraction

This preserves the failure mode that caused the repeated rounds. It has no
package-level terminal condition and can always identify one more large file.

### Accepted: whole-surface package consolidation

Move existing behavior into cohesive subpackages, preserve current paths at
explicit compatibility seams, remove duplicated transitional implementations,
and validate the complete affected surface in one human-visible transaction.

### Rejected: rewrite the training and playback pipelines

A rewrite would combine maintainability work with semantic and lifecycle risk.
The repository already has successful training evidence, so this engineering
unit must preserve rather than reinterpret the working behavior.

## Target architecture

### Distillation

The production owners are grouped by reason to change:

```text
unilab/algos/torch/distill/
├── contracts/       # stable config, identity, dataset and checkpoint contracts
├── datasets/        # dataset values, validation, diagnostics, IO and merge
├── collection/      # common, standard, transition and async collection
├── learning/        # trainer, routing, diagnostics and offline update
├── workflows/       # entry routing, DAgger, artifacts, resume and commit
├── runtime/         # persistent workers, process and resource lifecycle
├── observability/   # dependency-neutral debug and performance telemetry
└── fada/            # model, Oracle, source, collection, adaptation and persistence
```

Root modules retained for compatibility must be either:

1. a composition root that selects owners but contains no domain rule; or
2. a direct, stateless re-export facade.

A compatibility facade cannot own mutable state, validation, fallback logic,
checkpoint interpretation, data transformation, training decisions, or durable
writes. Modules that have no repository consumer at the old path are moved
without leaving a facade.

The existing `distill.fada` import surface may become a package with an
`__init__.py` re-export surface. Existing `fada_*` paths that still have callers
remain small compatibility modules until their consumers are migrated. No
dynamic import registry or `__getattr__` compatibility mechanism is admitted.

### Dataset and collection flow

```text
entry/composition
  -> collection transaction
  -> typed collection result
  -> dataset contract and validation
  -> dataset IO or learning transaction
```

Collection transaction state owns rollout counters, buffers, reset accounting,
history, and finalization exactly once. Pure label, projection, and validation
operations receive explicit values. FADA window collection retains pre-step
Oracle queries and its current acceptance/rejection accounting.

Dataset owners are divided into value construction, contract validation,
diagnostic projection, persistence, and multi-source assembly. Merge state may
accumulate compatibility facts, but it cannot load checkpoints, run a policy,
or mutate trainer state.

### Learning flow

`BehaviorDistillationTrainer` remains the sole owner of model references,
optimizer mutation, gradient execution, update count, and teacher evaluation.
Routing and diagnostics remain pure or read-only. Offline sampling owns index
pools and batch selection; the offline update transaction coordinates sampling
and trainer calls without owning either model or optimizer state.

The public `update` and offline entrypoints remain stable. Extraction must not
change detach/no-grad placement, optimizer step ordering, clipping already
present in the accepted implementation, loss reduction, or metric identity.

### Workflow and persistence flow

Workflow composition selects an admitted route and constructs typed inputs.
DAgger iteration owns one iteration's collection, aggregation, update, metrics,
and checkpoint activation. Artifact persistence alone owns hashes, atomic
writes, legacy adoption, resume identity, and manifest commit. Commit remains
the final durable action.

Diagnostics observe completed stage state and cannot feed values back into
training, collection, curriculum, or retry decisions.

### Runtime flow

Persistent resource creation, process lifecycle, request handling, exception
transport, and teardown are separated from collection semantics. A worker owns
its environment and policy resources for its declared lifetime. Runtime owners
may call collection owners but collection owners cannot import process/runtime
implementations.

### Playback

`scripts/play_interactive.py` becomes a CLI and viewer composition root. It may
parse CLI input, compose Hydra configuration, select the session factory, and
run the viewer transaction. It cannot own checkpoint interpretation, algorithm
construction, policy routing, overlay geometry, keyboard projection, or trace
formatting.

The visualization package owns:

- checkpoint and policy/session factories;
- distillation/FADA playback routing;
- viewer resource preparation and cleanup;
- keyboard and height-control projection;
- overlays and read-only trace formatting.

`interactive_playback.py` remains a stable compatibility entrypoint and direct
factory export surface. Moved implementations are removed rather than copied.

### G1 environment

`G1WalkEnv` remains the only owner of backend handles, mutable rollout state,
episode state, command/curriculum state, reset lifecycle, and framework hooks.
It keeps explicit framework-required methods and thin bindings where the reward
registry or environment interface requires methods on the environment object.

Focused `walk_*` modules own stateless or explicitly typed calculations for:

- observations and privileged observation materialization;
- reward terms and mode/gait aggregation;
- action/control projection and action diagnostics;
- command decisions;
- rollout snapshot values and reset planning;
- domain-randomization calculation and persistence projection.

No mixin forest, generic manager, backend-private probe, dynamic method
injection, or second mutable environment owner is admitted. A large environment
file is acceptable where lines are thin framework bindings; it is not split
solely to satisfy a line threshold.

## Dependency rules

- scripts and compatibility facades depend on production owners, never the
  reverse;
- workflows depend on contracts, datasets, collection, learning, runtime, and
  persistence interfaces; those owners do not import workflow entry modules;
- learning does not import workflow, scripts, playback, or backend code;
- collection does not mutate trainer, optimizer, artifact manifest, or workflow
  schedule state;
- FADA-specific owners may depend on stable collection, dataset, learning, and
  contract interfaces; generic owners do not depend on FADA workflow modules;
- visualization may depend on stable policy/checkpoint loading interfaces but
  training and environment modules never depend on visualization;
- G1 pure helpers do not import `joystick.py` or backend subclasses;
- production code never imports tests.

The final graph must be acyclic across these package groups. Any necessary
shared value object moves toward `contracts/`, not into a generic utility dump.

## State, failure, and cleanup ownership

- partial dataset and collection state remains transaction-local until final
  validation succeeds;
- artifact writes retain atomic/fail-closed behavior and cannot publish a mixed
  checkpoint/config/run identity;
- collector and worker exceptions continue through the existing error transport
  and lifecycle cleanup route;
- viewer and environment resources close on normal completion and exceptions;
- resume accepts only the same identities currently accepted and fails at the
  same or a nearer contract boundary;
- compatibility facades do not catch errors, add fallbacks, or translate values;
- diagnostics failures cannot silently alter the training route.

## One-shot construction sequence

These are internal engineering checkpoints, not user-visible approval gates.

1. **Freeze the baseline.** Record the current dirty-tree identity, public import
   surface, entrypoints, owner symbols, dependency graph, and currently passing
   characterization tests. Add missing behavior and import characterization
   tests before moving implementation.
2. **Create package boundaries.** Introduce the target subpackages and explicit
   dependency rules without duplicating implementation.
3. **Move contracts and datasets.** Establish stable low-level dependencies,
   then migrate dataset construction, diagnostics, IO, and merge.
4. **Move collection and runtime.** Consolidate standard, transition, FADA, async,
   and persistent-worker lifecycle around typed state owners.
5. **Move learning and workflows.** Consolidate trainer/offline ownership and the
   complete entry/DAgger/artifact transaction chain.
6. **Consolidate FADA.** Group model, Oracle, source, collection, adaptation,
   diagnostics, checkpoint, and workflow-specific owners under the FADA
   boundary while retaining required import compatibility.
7. **Finish playback and G1 boundaries.** Remove duplicated playback
   implementations, reduce the script/facade to composition, and move remaining
   stateless G1 calculations without splitting environment state ownership.
8. **Retire transitional duplication.** Replace used legacy paths with direct
   re-exports, remove unused private compatibility modules, and verify every
   symbol has one implementation owner.
9. **Run the complete proof route.** Repair all in-scope offline failures before
   returning to the user. Do not stop after the first package or hotspot passes.

## Completion gates

The engineering transaction is complete only when all of the following hold:

- the distillation production implementation is grouped under the accepted
  package boundaries rather than remaining a flat collection of owners;
- every mutable state, durable artifact, and semantic validation has one owner;
- compatibility modules contain no business decisions or mutable lifecycle;
- no duplicated old/new implementation or silent fallback remains;
- all repository import paths with current consumers still resolve;
- the package dependency graph satisfies the declared direction and is acyclic;
- transaction and orchestration functions are normally at or below 120 lines;
- production owner modules are normally at or below 500–600 lines;
- any exception to those size guides is documented as cohesive framework
  binding or value-definition code, not deferred refactoring debt;
- the retained size exceptions are cohesive owners: G1 command/backend
  lifecycle, reward binding dispatch, transition collection, offline
  sampling/update orchestration, trainer update state, dataset merge
  validation, workflow entry collection/training, performance telemetry, and
  the FADA collection transaction; deep owner-boundary tests prevent them
  from reacquiring neighboring responsibilities;
- `scripts/play_interactive.py` is approximately 300 lines or less and
  `interactive_playback.py` approximately 250 lines or less;
- `G1WalkEnv` contains environment lifecycle and explicit framework bindings,
  with no movable calculation, diagnostic formatting, or persistence rule left
  embedded solely to reduce effort;
- focused owner tests, public import tests, transaction failure/cleanup tests,
  affected suites, full repository tests, Ruff, compile checks, dependency
  checks, and `git diff --check` introduce no new failure;
- line/function recount and final maintainability review report no in-scope P0,
  P1, or unresolved P2 responsibility finding;
- the closeout does not recommend another local split inside this accepted
  scope. Any future change must be triggered by new behavior or concrete
  dependency evidence, not by continuing this refactor indefinitely.

Size guides are completion checks after ownership review. They do not authorize
arbitrary wrappers, mixins, services, or file splitting.

## Proof route

Before movement, characterization tests pin:

- public imports and re-exports;
- dataset schemas, row order, merge and persistence;
- collection same-state labeling, reset, filtering, counters, and failure paths;
- trainer tensor/update behavior and offline sampling;
- DAgger resume, iteration, checkpoint activation, cleanup, and commit order;
- FADA checkpoint/environment contracts and Oracle lineage;
- playback checkpoint/session routing, controls, overlays, trace, and cleanup;
- G1 observation, reward, action, command, reset, DR, and rollout snapshots.

During movement, focused tests run at each owner boundary so faults are localized,
but failures are repaired within the same authorized transaction. Final closure
requires all affected tests plus the full repository suite. Existing baseline
failures may remain only if reproduced before modification and proven unrelated;
no new failure may be hidden by deselection or threshold changes.

Offline proof establishes behavior preservation and maintainability only. No
simulator, training, checkpoint-quality, deployment, or real-robot claim follows
from this work.
