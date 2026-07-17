# DAgger Runtime Integration Checklist

Status values: `PASS`, `PARTIAL`, `PENDING`, `BLOCKED`.

| Item | Owner | Evidence | Status |
|---|---|---|---|
| Interface-vs-copy decision | runtime plan | owner map and proposal | PASS |
| Isolated worktree | git/worktree | `codex/dagger-mainline-runtime` | PASS |
| Focused baseline | affected distill + IPC tests | `315 passed` | PASS |
| Persistent request/result protocol | distill runtime owner | S1/S2 protocol tests | PASS |
| AsyncRunner lifecycle reuse | `src/unilab/ipc/async_runner.py` interface | same spawned worker serves two requests; errors and cleanup verified | PASS |
| SharedWeightSync version identity | `src/unilab/ipc/weight_sync.py` interface | protocol mismatch and real two-version shared-memory probe | PASS |
| Outer iteration barrier | workflow owner | all scenario requests use `student_k` | PASS |
| Legacy workflow OFF path | workflow owner | default callback and manifest regression | PASS |
| HP-3a Hydra/script execution-mode connector | config + entrypoint owner | OFF compose, ON injected connector, fail-closed production gap | PASS |
| HP-3b real persistent runtime owner | distill runtime + SharedWeightSync | resident student and reusable exact-identity teachers/envs pass | PASS |
| HP-3b1 persistent student weights | distill runtime + SharedWeightSync | two-version spawned golden probe | PASS |
| HP-3b2 persistent role resources | distill runtime + collector/env owners | real cached teachers/envs and bounded MuJoCo | PASS |
| HP-3b2 exact resource cache key | distill resource owner | task/backend/resolved cfg/num_envs/teacher identity test | PASS |
| HP-3b2 per-request reset isolation | distill worker + env owner | command/done/transition-age lifecycle fixture | PASS - fake resource |
| HP-3b2 init/reuse/close counters | distill worker | repeated-request and exceptional-exit fixture | PASS - fake resource |
| HP-3b2 legacy/persistent dataset differential | workflow acceptance | schema/role/intent/scenario/teacher/version comparison | PASS - deterministic collector owners |
| HP-3b2 production factory wiring | config + entrypoint + runtime owner | ON connector uses real owner; OFF remains unchanged | PASS |
| HP-3b2 bounded MuJoCo sequence | distill runtime owner | walk/stand/transition/walk structured snapshot plus close report | PASS |
| Real persistent G1 connectivity | distill runtime owner | one spawned PID, two exact resources, four isolated resets, zero errors | PASS |
| HP-4 Gate 0A identity preflight | governance | isolated HP-3b2 diff and final-freeze field inventory; dirty tree explicitly non-formal | PASS |
| HP-4a metrics owner map | distill runtime/collector/workflow owners | E41 names stage boundaries; scripts remain assembly-only; connectors not integrated | PASS |
| HP-4a metrics schema and persistence | `distill/performance.py` | E41; 16 fake-clock/roundtrip/identity/fail-closed tests | PASS |
| HP-4a human metrics-contract approval | user decision boundary | user authorized HP-4a2a, HP-4a2b, and HP-4a2c; Gate 0B/HP-4b remain separate | PASS |
| HP-4a2a worker stage observations | persistent G1 worker + metrics schema | E42; fake-clock weight-sync/write/total observations; flat metrics unchanged | PASS |
| HP-4a2b collector stage observations | collector + metrics accumulator + worker pass-through | E43; exact role/transition fake-clock oracle and default-OFF isolation | PASS |
| HP-4a2c1 pure identity enrichment | metrics owner | focused command: 22 passed; exact seven-stage golden and malformed order/schema fail closed | PASS |
| HP-4a2c2 run-local artifact and resume | parent workflow + metrics owner | focused workflow/performance command: 38 passed; atomic reload, idempotent completed resume, manifest identity, legacy isolation | PASS |
| HP-4a2c3 formal OFF-default connector | config/entrypoint assembly | E44; distinct teacher hashes + resolved cfg/algo.seed/device/num_envs; legacy passes none | PASS |
| HP-4 Gate 0B final immutable identity | governance + run artifact owner | E60; r8 deterministic bundle, identity, pre-execution oracle v2 contract, empty output root, and no-training preflight | PASS |
| HP-4 Gate 0B execution-env repair | command/environment identity owner | E54; 171-package provider snapshot live match, nested frozen import and train_distill help exit 0, output absent | PASS |
| HP-4 Gate 0B workflow teacher config repair | Hydra workflow config owner | E56; RED 99 vs 98, workflow 98/98, generic 99/99, two real checkpoint guards, r7 refreeze | PASS |
| HP-4 Gate 0B bundle repair | governance + raw artifact owner | E49; 1241 files, required build inputs, two equal hashes, uv build/import/XML/compose pass | PASS |
| HP-4a2d1 legacy request metrics | metrics + legacy collection + workflow connector | focused command: 238 passed; mode-specific records, atomic artifact, integer callback/OFF isolation | PASS |
| HP-4a2d2 workflow/learner metrics | workflow + offline + trainer owners | E46; aggregation/staging/forward/backward/optimizer/checkpoint exact fake-clock evidence | PASS |
| HP-4a2d3 cleanup-final persistence | runtime close + metrics/workflow finalizer | E46; both routes persist cleanup; persistent close counters required; manifest/hash/count reload | PASS |
| HP-4b bounded A/B identity | formal entrypoint + workflow owner | E61 eight exact frozen orders, eight exit-zero commands, eight oracle acceptances, complete attestation | PASS |
| HP-4b legacy semantic acceptance | workflow + dataset + oracle owners | E61 four legacy runs pass identical role/scenario/transition/checkpoint/metrics/cleanup contracts | PASS |
| HP-4b persistent lifecycle acceptance | workflow + runtime owners | E61 four persistent runs pass one-version resident lifecycle, exact resources, 3 resets/requests, 0 errors, complete cleanup | PASS |
| HP-4b structured timing evidence | run artifact owner | E61 eight complete 28-record artifacts with raw values, median, range, stdev, paired ratios and measurement definition | PASS |
| HP-4b stable end-to-end speedup | evidence/governance owner | E61 paired ratios cross 1 and route median persistent e2e is 6.34% longer; no stable speedup claim | PARTIAL |
| HP-4b fork scenario identity repair | workflow fork owner + data annotation contract | E51; preserve scenario/preserve-row flags; existing transition-aware annotation/merge chain; parent source hashes unchanged; 288 passed, 8 skipped | PASS |
| HP-4b acceptance-oracle repair | experiment acceptance owner | E58 frozen oracle/contract, kind-aware checks, existing order-1 acceptance, seven hashes unchanged, no rerun | PASS |
| HP-4b persistent output materialization repair | workflow owner | E60 exact RED/GREEN, 493 affected tests, Ruff, deterministic r8 freeze and no-training preflight | PASS |
| HP-4c bottleneck verdict | evidence/governance owner | E62 absolute time/share/range/stdev/CV, per-scenario cache counters, timer-source trace, one-time/recurring classification | PASS |
| HP-4c HP-5 authorization | user decision boundary | E65 confirms cold-cost amortization but no recurring warm owner; no source optimization authorized | BLOCKED |
| HP-4c two-iteration discriminator | governance + formal runtime owners | E65 accepted pair, version/cache/cleanup progression, iteration-aware verdict | PASS |
| HP-4c iteration-aware oracle v4 repair | acceptance-oracle owner | E64 v4/amendment, accepted existing order 1, 16-file identity attestation, no training rerun | PASS |
| HP-4c r9 persistent order-2 resume | formal runtime + acceptance owners | E65 exact preflight, one persistent run, oracle v4, no legacy rerun | PASS |
| HP-4c r10 repeated two-iteration freeze | governance + benchmark identity owner | E66 8 balanced orders, per-order compose hashes, v4 oracle/decision contract, empty output, preflight | PASS |
| HP-4c r10 eight-run execution | formal runtime + acceptance owners | E67 exact balanced sequence, 8/8 v4 acceptance, NO_STABLE_SPEEDUP | PASS |
| HP-6a production readiness | repository + governance owners | E69/E71 status closure plus E70 537 pass/24 skip and Ruff | PASS |
| HP-6a1 runtime audit-status repair | async/performance + current atlas evidence owners | E69 local repair plus E71 whole-Architecture closure | PASS |
| HP-6a production readiness restart | repository + governance owners | E70 executable gates green; E71 cross-file blocker resolved | PASS |
| HP-6a2 Runtime Atlas status repair | runtime atlas + checker owners | E71 semantic RED/GREEN, zero stale hits, registry consistency | PASS |
| HP-6b repository-wide production gate | repository Makefile owners | E86 exact gate: 1556 passed, 51 skipped; static gates pass; coverage 70% | PASS |
| HP-6b1 repository lint-owner repair | section-8 diagnostic owner | E73 two dead locals removed; compile/Ruff/AST pass | PASS |
| HP-6b2 mechanical diff review + full rerun | repository formatter + Makefile owners | E74 AST review pass; mypy 20 errors in 8 files; later targets not run | BLOCKED |
| HP-6b3 branch-owned type repair | DAgger collector/runtime/workflow/G1 owners | E75 zero scoped mypy errors; Ruff pass; 111 affected tests pass | PASS |
| HP-6b4 HEAD-baseline type repair | model/playback/data/G1 config owners | E76 mypy/Ruff pass; 442 passed, 3 skipped | PASS |
| HP-6b5 final repository rerun | repository Makefile owners | E77 format/Ruff/mypy pass; Pyright 6 collector errors; coverage not run | BLOCKED |
| HP-6b6 collector Pyright narrowing | distillation collector owner | E78 Pyright 0; mypy/Ruff pass; 86 collector tests pass | PASS |
| HP-6b7 final repository rerun | repository Makefile owners | E79 static gates pass; test-cov 14 failed, 1544 passed, 49 skipped | BLOCKED |
| HP-6b8 G1 gait-config compatibility repair | G1 reward-config accessor owner | E80 exact ten G1 failures plus type/lint pass | PASS |
| HP-6b9 remaining four-failure diagnosis | Stewart/docs/CLI owners | E81 provider, generated-doc, execution-env classifications | PASS |
| HP-6b10 Motrix provider/test selection | Stewart test owner | E82 4 pass, 2 optional-provider skips; Ruff pass | PASS |
| HP-6b11 generated support matrix | derived docs owner | E83 exact two generated rows; docs contract pass | PASS |
| HP-6b12 uv env test isolation | CLI test owner | E84 target passes under frozen outer env; Ruff/diff pass | PASS |
| HP-6b13 combined 14-regression closure | G1/Stewart/docs/CLI owners | E85 12 passed, 2 expected provider skips | PASS |
| HP-6b14 final repository rerun | repository Makefile owner | E86 1556 passed, 51 skipped; static gates pass; coverage 70% | PASS |
| Legacy/persistent differential | workflow acceptance | role/intent/scenario/age/teacher-action semantics | PASS |
| Production gate | repository | affected suites and `make test-all` | PASS | E86 exact gate: 1556 passed, 51 skipped; static gates pass; coverage 70% |

## Current Stop Condition

HP-1 through HP-4c and HP-6b are closed. E67 records
`NO_STABLE_SPEEDUP`; no HP-5 owner exists and persistent execution remains
OFF-default. E86 passes the repository S0-S3 gate. The active runtime contract
is `DISTILL-TRAIN-v003`: integration complete, promotion deferred, default
`legacy`. Remaining boundaries are RT-10 physical acceptance, optional Motrix
runtime, slow/S4 evidence, and explicit diagnostic-only labeling of the manual
route.
