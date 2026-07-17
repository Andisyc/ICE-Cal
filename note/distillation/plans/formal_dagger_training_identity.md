# Formal DAgger Training Identity

Status: FT-0 owner and deploy integration PASS; server materialization pending.
Date: 2026-07-17

## Decision

HP-1 through HP-7 are closed engineering work. E99 proves the optimized
persistent production route can complete one bounded iteration, but its r6
checkpoint and supervisor belong to a performance/integration sentinel. They
are not a formal-training lineage and must not be reused, resumed, or silently
promoted.

The formal DAgger route starts cleanly from the original completed parent
iteration 3. It creates a new run, output identity, freeze, supervisor, oracle,
and physical-evaluation boundary. Persistent execution is selected explicitly
for this run only; the repository default remains `legacy` and promotion
remains deferred under `DISTILL-TRAIN-v003`.

This plan changes no method semantics. The active Concept Figure and
`DISTILL-METHOD-v001` remain unchanged.

## Design Point Register

| design ID | canonical human name | active contract + section | Concept Figure block | formal-training boundary |
|---|---|---|---|---|
| DISTILL-DP-01 | Teacher Policies | `DISTILL-METHOD-v001#teacher-policies` | DT-M-01 | Freeze existing walk/stand checkpoint bytes; no height teacher |
| DISTILL-DP-02 | Command Intent | `DISTILL-METHOD-v001#command-intent` | DT-M-02 | Preserve walk, stand, and walk-to-stop command semantics |
| DISTILL-DP-03 | Role Data | `DISTILL-METHOD-v001#role-data` | DT-M-03 | Reuse only manifest-validated immutable role datasets |
| DISTILL-DP-04 | MoE Student | `DISTILL-METHOD-v001#moe-student` | DT-M-04 | Produce a candidate checkpoint, never automatic promotion |
| DISTILL-DP-05 | Student-State DAgger | `DISTILL-METHOD-v001#student-state-dagger` | DT-M-05 | Preserve rollout, relabel, cumulative aggregate, update, and version barriers |

## Source-Of-Truth Identity

| semantic object | active owner | frozen rule | excluded path | acceptance |
|---|---|---|---|---|
| Parent lineage | parent `run_manifest.json` iteration 3 | Exact checkpoint, aggregate, manifest paths, hashes, sizes, rows, and weight version | r6 HP-7 sentinel checkpoint | Fail closed on mismatch |
| Teacher identity | role artifacts and owner Hydra config | Exact walk/stand checkpoint paths and SHA-256 | Filename-only identity | Compose and manifest agree |
| Role data | manifest role artifacts | Exact path, hash, schema, role, projection, and sample count | Silent recollection or incompatible reuse | Role preflight reports compatible `REUSE` |
| Training semantics | `DISTILL-TRAIN-v003` and workflow owner | Same DAgger barrier, replay, quota, schema, and RNG contracts | HP benchmark semantics | Config and oracle match active contract |
| Execution identity | UniLab owner CLI and fresh supervisor | Explicit `persistent_async`, one argv/environment, one GPU identity | Default change or r6 supervisor | Pre/post identity checks |
| Output identity | new formal run root | All output/log/telemetry paths absent before execution | Resume, overwrite, retry, or r6 reuse | Absence preflight and one-new-run oracle |
| Candidate acceptance | formal oracle plus RT-10 | Training artifacts first; physical evaluation separately | Training success interpreted as promotion | Artifact PASS does not imply physical PASS |

The parent `distillation_metrics.json` remains audit-only unless the workflow
owner is shown to consume it. Its drift must be recorded, but it does not
replace checkpoint, aggregate, role-artifact, or teacher hard gates.

## Step Map

The work is split because identity materialization and real training cross an
independent runtime and human-authorization boundary.

### FT-0 / 2: No-training formal identity and oracle

- Objective: materialize an immutable formal-training identity before any
  output directory or training process exists.
- Scope: source/config hashes, owner-CLI Hydra compose, original parent
  iteration-3 checkpoint and aggregate, teachers and role datasets, workload,
  seed, device, dependencies/imports, new output paths, supervisor, oracle, and
  dry preflight.
- Non-scope: env construction, collection, learner updates, checkpoint
  creation, r6 evaluation/promotion, retry/resume, default change, production
  code modification, dependency sync, RT-10 execution, commit, or PR.
- Owner files/modules: a new deploy materializer/oracle under `scripts/deploy/`;
  generated identity, supervisor, and oracle are server artifacts; this plan
  and `checklists/current.md` own governance.
- Expected evidence: one no-training freeze JSON with `accepted=true`, empty
  failures, resolved Hydra YAML/hash, exact argv/environment, all input hashes,
  output absence, syntax-checked oracle/supervisor hashes, and
  `training_executed=false`.
- Stop condition: any mismatch, missing input, dirty runtime source, config
  disagreement, existing output, failed preflight, or r6 lineage records
  `BLOCKED`. FT-1 remains closed. PASS returns control with all hashes.

### FT-1 / 2: One formal DAgger training execution

- Objective: execute exactly the FT-0 identity and produce one candidate whose
  lineage and artifacts pass the frozen oracle.
- Scope: only the frozen workload through the formal single-entry workflow,
  followed immediately by the frozen artifact oracle.
- Non-scope: automatic retry, resume after failure, output repair, workload
  adjustment, second run, A/B, default-on, promotion, or interpreting training
  completion as RT-10 physical acceptance.
- Owner files/modules: existing Hydra/CLI/workflow, persistent runtime, learner,
  manifest, metrics, checkpoint, and cleanup owners; no copied runtime path.
- Expected evidence: command exit, lineage, scenario/weight barriers,
  cumulative dataset, update counts, checkpoint hash, metrics, cleanup, timing,
  and oracle verdict.
- Stop condition: first command or oracle failure stops this output identity.
  PASS returns control before RT-10 or promotion.

## FT-0 Freeze Requirements

FT-0 must materialize, not merely describe:

1. Repository root, committed HEAD, clean runtime diff, and SHA-256 for every
   runtime/config/materializer/oracle input.
2. `uv run --no-sync`; Python, Torch, CUDA, MuJoCo, UniLab,
   `UV_PROJECT_ENVIRONMENT`, `UV_CACHE_DIR`, and resolved import paths.
3. Owner-CLI composed/resolved Hydra config and SHA-256. Route fields use CLI
   flags, not passthrough overrides.
4. Original parent iteration-3 manifest, checkpoint, aggregate, role artifacts,
   teachers, sizes, hashes, rows, schemas, and weight version.
5. Scenario order, samples per role, env count, outer iterations, batch size,
   configured update floor, production-derived effective updates, seed, device,
   and execution mode.
6. A new absolute run root and separate log, time, telemetry, freeze,
   supervisor, and oracle paths. Every execution output must be absent.
7. Exact argv and environment arrays; shell prose is not canonical identity.
8. Fail-closed oracle clauses for source/config/dependency/GPU, command,
   lineage, scenario/weight versions, datasets/checkpoint, metrics, cleanup,
   and telemetry attribution limits.
9. A separate RT-10 specification naming candidate input, physical metrics,
   thresholds, and stop boundary. It is frozen in FT-0 but not run by FT-1.

Formal outer iterations and effective updates must be printed explicitly by
FT-0. They may not be copied from r6 or silently chosen by a helper. If no
accepted current document fixes them, FT-0 stops for a human decision before
writing an executable supervisor.

## Frozen Formal Workload R1

E103 records the human-selected two-round workload in
`formal_dagger_2round_r1.spec.json`:

- original parent iteration 3;
- aggregate rows `853504`, then `855040`;
- effective update schedule `[12320, 12352]`;
- total effective updates `24672`;
- seed `0`, logical device `cuda:0`, explicit `persistent_async`;
- new output root
  `/ssd1/cyx/UniLab/logs/distill_workflow/g1_walk_stand_formal_dagger_2round_20260717_r1`.

The server materializer must recompute this schedule from the real parent
aggregate through the production replay owner. The spec is an expected
identity, not authority to override a different observed value.

## Human-Control Boundary

- This document authorizes planning only.
- FT-0 materialization is a separate no-training action.
- FT-1 requires explicit authorization after FT-0 PASS.
- RT-10 requires explicit authorization after FT-1 artifact PASS.
- Promotion/default-on requires new stable evidence and a new human decision.

## Current Implementation State

E101 records the local owner implementation in
`src/unilab/algos/torch/distill/formal_identity.py`. It builds and validates
formal lineage, workload, owner-CLI argv/environment, fresh output identity,
source/artifact freeze records, and generated supervisor/oracle text. It
rejects r6/HP-7 sentinel lineage and never executes training.

E102 records the thin deploy connector and local file-level integration PASS.
It captures source/config/artifact, owner-CLI compose, dependency/import, GPU,
command, output, supervisor, oracle, and preflight identities without invoking
training. FT-0 remains PARTIAL only because the formal workload/output spec and
authenticated server no-training materialization are not yet frozen/executed.

E104 integrates the real aggregate workload discriminator into that connector.
The one-line materializer loads the manifest-owned aggregate, reads resolved
scenario quotas/replay fields, calls the offline replay owner, and compares the
observed schedule/total against the spec. A mismatch blocks preflight.
