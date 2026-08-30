# Final Three-Hotspot Architecture Design

## Goal

Remove the three source-confirmed maintainability defects left after package
consolidation without changing training, tensor, environment, playback,
checkpoint, Hydra, reward, observation/action numeric, reset, DR, or RNG
behavior.

## Frozen behavior and authority

- Work in the existing dirty `main` checkout and preserve all unrelated user
  changes and artifacts.
- No branch, commit, push, training, simulation, remote work, checkpoint
  evaluation, schema change, or policy-quality claim.
- Legacy public imports and public collection/environment entrypoints remain
  valid.
- Every transformation is structural and must be covered by existing
  characterization tests plus new architecture fitness tests.

## Architecture

### 1. Distill dependency direction

Production owner packages may import concrete owners but never the public
`unilab.algos.torch.distill` aggregation package. The aggregation package is a
consumer-facing boundary only. Workflow modules replace aggregate imports with
direct imports from `contracts`, `datasets`, `collection`, `learning`,
`runtime`, `observability`, and `workflows`.

This removes the reverse dependency edge
`workflow owner -> public aggregation -> all owners` while preserving all
external imports.

### 2. Standard collection transaction

`collect_distillation_dataset_from_env` remains the public function and exact
signature. It becomes a thin adapter over an explicit
`StandardCollectionTransaction`. The transaction owns validation, reset,
per-step label/action collection, row admission, metadata finalization, and
performance accounting. Mutable buffers and counters live in one transaction
state rather than a 317-line function-local data clump.

Environment and policy objects remain borrowed resources: this transaction
does not create, close, persist, or replace them. Tensor construction, row
ordering, masks, devices, inference mode, and metadata values remain exact.

### 3. G1 framework binding owners

`G1WalkEnv` remains the registered environment and unique mutable runtime state
owner. Framework hooks are grouped into three explicit binding owners:

- observation binding: observation contract, privileged observation assembly,
  command/mode/gait projections, and symmetry layouts;
- control binding: command scheduling, action authority, action execution, and
  action diagnostics;
- runtime binding: state update, curriculum logging, rollback snapshot, and
  trace capture.

The binding owners are stateless mixins selected once in the class MRO. They do
not create new state, schemas, adapters, or public entrypoints. This is admitted
because it clarifies framework-hook ownership while keeping the single Env
lifecycle owner and eliminates unrelated change reasons from `joystick.py`.

## Dependency rules

- owner packages never import compatibility facades or the public aggregate;
- G1 binding owners may depend on pure `walk_*` helpers and declared UniLab
  base/backend contracts, never on backend subclasses;
- `joystick.py` is the composition root for the three G1 binding owners;
- compatibility modules remain direct re-export facades only.

## Proof route

1. Add architecture fitness tests for direct owner imports, the standard
   transaction boundary, G1 MRO ownership, and legacy identity compatibility;
   observe RED before production edits.
2. Run focused collection, workflow, G1, playback, and import-boundary tests.
3. Run Ruff, compileall, import sweep, `git diff --check`, affected regression,
   and repository regression.
4. Validate `code-review-expert` plan/final receipts and a complete
   `one-shot-execution` unit.

## Stop conditions

Stop only if the refactor requires a semantic decision, changes a public
signature/schema, cannot preserve a dirty overlapping user edit, or reveals an
unresolvable evidence conflict. Ordinary offline failures remain inside this
execution unit.
