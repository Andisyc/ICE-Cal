# Current Distillation Acceptance Checklist

Status values: `PASS`, `PARTIAL`, `PENDING`, `BLOCKED`.

| Item | Owner | S/T evidence | Status | Evidence |
| --- | --- | --- | --- | --- |
| Teacher checkpoint input dimensions | `teacher.py` | S1 / T-contract | PASS | `tests/algos/test_g1_distillation_contract.py` |
| Role/intent dataset schema roundtrip | `data.py` | S1 / T-contract | PASS | `tests/algos/test_g1_distillation_contract.py` |
| Active/inactive collection filtering | `collector.py` | S2 / T-connect | PASS | collection contract tests |
| Per-expert behavior loss | `trainer.py` | S1 / T-regression | PASS | selected-expert tests |
| Inactive expert optimizer isolation | `trainer.py` | S1 / T-regression | PASS | inactive-expert drift test |
| DAgger aggregation | `dagger.py` | S2 / T-regression | PASS | iterative aggregation test |
| DAgger rollout expert matches intent | `dagger.py` | S2 / T-regression | PASS | fixed-expert rollout test |
| DAgger outer rollout/update iteration | `dagger.py::run_iterative_dagger_updates` | S2 / T-regression | PASS | `tests/algos/test_g1_distillation_contract.py` multi-iteration fixture |
| Formal online DAgger route | `train_distill.py::run_online_dagger_update` | S2 / T-connect | PASS | `tests/scripts/test_train_scripts.py` online-dagger routing fixture |
| Manual collect/offline path isolation | workflow owner | S2 / T-contract | PENDING | formal workflow must label it diagnostic-only |
| Active single-entry training contract | distillation contract registry | S0 / T-contract | PASS | `contracts/active/training/DISTILL-TRAIN-v003.md`; integration complete, promotion deferred, legacy default |
| Concept Figure outer-loop alignment | Concept Figure | S0 / T-contract | PASS | DAgger edges name rollout/relabel/aggregate/update cycle |
| Role-aware teacher dataset reuse | `workflow.py` artifact manifest owner | S2 / T-contract+connect | PASS | stand/walk `REUSE`; absent height `COLLECT` fixture |
| Role artifact content identity | `workflow.py` | S1 / T-contract | PASS | teacher/dataset/config/schema hash tests |
| Per-role reuse decision | `workflow.py` | S1 / T-contract | PASS | `REUSE/COLLECT/STALE/INCOMPATIBLE` fixtures |
| Single-entry bootstrap owner | workflow + script dispatch | S2 / T-connect | PASS | default OFF; enabled task-owner and generated-path fixtures |
| Multi-role cumulative DAgger | workflow owner | S2 / T-connect+probe | PASS | previous-round checkpoint and 4-to-6 cumulative source probe |
| Workflow resume and fork | workflow manifest owner | S1-S2 / T-contract+connect | PASS | missing-round resume and immutable-parent fork fixtures |
| Legacy dataset manifest adoption | workflow owner | S1 / T-contract | PASS | explicit tensor/metadata validation before manifest write |
| Formal `train --algo distill` route | CLI + Hydra profile | S2 / T-connect | PASS | CLI route and `g1_walk_stand` compose tests |
| Role dataset schema migration | data/workflow owners | S2 / T-contract | PENDING | adding a height feature must migrate legacy role rows explicitly or fail `INCOMPATIBLE` |
| Concept Figure to active-contract mapping | docs governance | S1 / T-contract | PASS | E6 + `check_distillation_atlas.mjs` |
| Concept Figure connector geometry | architecture atlas | S2 / T-integration | PASS | E8: explicit anchors, orthogonal routes, 18 px collision guard, browser QA |
| Method-to-Code owner/source mapping | architecture atlas | S1 / T-contract | PASS | E6 + repository-local path validation |
| Browser source-navigation delivery | atlas viewer/server | S3 / T-integration | PASS | E7: 19 links, rendered click, exact server path/line, CLI exit 0 |
| Checkpoint identity and parent chain | workflow manifest owner | S2 / T-connect | PASS | immutable bootstrap/iteration/fork hashes and parent references |
| RT-1 reset distribution reproduction | G1 reset provider | S1 / T-probe | PASS | E15: active command, gait, and non-zero reset qvel reproduced |
| RT-1 command-observation skew reproduction | playback session + routing | S1 / T-probe | PASS | E15: active routing command differs from cached policy observation |
| Playback standing reset identity | `play_interactive.py` + G1 reset provider | S2/S4 / T-connect+live | PASS | E18: 32/32 MuJoCo resets use standing-only owner contract |
| Playback reset command distribution | playback initialization | S1-S2 / T-regression | PASS | E18: reset command abs max is 0 |
| External command / policy obs atomicity | G1 env + playback session | S1-S2 / T-dataflow+regression | PARTIAL | E16 `258 passed`; live route/expert synchronization still pending |
| Repeated standing reset sentinel | playback live probe | S4 / T-live | PASS | E18: 32/32 reset command/qvel/obs/gait records pass |
| Standing-teacher recovery authority | differential probe | S4 / T-live+differential | PASS | E21: final exact post-walk snapshot, WT/WS 80-step differential, all 14 checks pass |
| Transition training contract inheritance | docs governance | S0 / T-contract | PASS | E23 established v002 transition semantics; active registry points to v003, which supersedes and inherits v002; v002 is archived |
| Transition dataset scenario schema | `data.py` + batch transport | S1 / T-contract+roundtrip | PASS | E24; 70 distillation contract tests |
| Transition collector owner | `collector.py` | S2 / T-connect+probe | PASS | E25; atomic switch, teacher role change, age chronology, and done-row reset probe |
| Transition DAgger workflow integration | workflow owner | S2 / T-connect+regression | PASS | E26; scenario dispatch, cumulative lineage, quota balancing, resume mismatch, and legacy OFF-path regression |
| Reset and walk-to-stop physical gate | acceptance owner | S4 / T-live | BLOCKED | E28; 32-start grid has done_count=19 and stop-speed decay failure |
| Low-speed walking activation | playback + policy | S4 / T-live | PARTIAL | command routing contract exists; physical quality unaccepted |
| Walk-to-stop recovery | transition-conditioned DAgger workflow | S2/S4 / T-connect+live | BLOCKED | E27 collected real transition rows; E28/E29 show insufficient post-switch coverage and failed recovery gate; E30 adds guards |
| Height teacher integration | no current teacher | S0-S4 | BLOCKED | qualified height checkpoint absent |
| Promoted checkpoint selection | no owner | S2/S4 | PENDING | proposal only |

| RT-7 scenario compose and manifest contract | workflow + config owner | S2 / T-connect+compose | PASS | E26; three scenarios and 0.50/0.25/0.25 quotas compose successfully |
| Formal real-artifact transition run | workflow + checkpoint owner | S2 / T-formal | PASS | E27; real teachers, student checkpoint, transition datasets, and manifest lineage persisted |
| Student transition live sentinel | playback + MuJoCo acceptance owner | S4 / T-live | BLOCKED | E28/E29/E31; RT-9c candidate is not promotable |

| RT-9a transition failure mechanism audit | collector + offline training owner | S1/S4 / T-dataflow+live | PASS | E29; first failed boundary is transition coverage/training exposure |
| RT-9b transition coverage contract | `collector.py` + workflow dispatch | S1/S2 / T-contract+connect | PASS | E30; horizon and max-age checks are enforced and persisted |
| RT-9b transition replay exposure | `offline.py` + workflow profile | S1/S2 / T-contract+connect | PASS | E30; insufficient `max_updates` fails closed |
| RT-9c role artifact reuse and bounded workflow | workflow + manifest owner | S2 / T-formal+lineage | PASS | E31; stand/walk are REUSE, aggregate=640, candidate and hashes persisted |
| RT-9c strengthened transition coverage | transition collector + dataset audit | S1/S2 / T-dataflow | PASS | E31; 128 rows, post-switch rows=96, max age=23, min horizon=20, schema v002 |
| RT-9c replay exposure | offline trainer + workflow manifest | S1/S2 / T-contract+formal | PASS | E31; DAgger iteration records updates=128 for the configured transition quota |
| RT-9c student physical acceptance | playback + MuJoCo acceptance owner | S4 / T-live | BLOCKED | E31; 26/32 episodes terminate and stop-speed decay fails |
| RT-9d policy-quality isolation probe | MoE policy + dataset metadata owner | S1/S2 / T-dataflow+probe | PASS | E32; transition workflow passes full soft student, while deployment uses command-selected expert |
| RT-9e command-conditioned transition rollout | `train_distill.py` + transition collector owner | S2 / T-connect+probe | PASS | E33; transition collector and workflow resolve active->0/inactive->1, fixture and connector suite pass |
| RT-10 bounded retrain artifact | workflow + checkpoint owner | S2 / T-formal+persist | PASS | local manifest reached `DAGGER_ITERATION_1_COMPLETE`; final checkpoint exists; physical quality not implied |
| RT-10 physical acceptance | MuJoCo acceptance owner | S4 / T-live | PENDING | bounded retrain artifact exists, but no accepted repeated-reset/walk-to-stop physical grid is recorded |
| HP-1 DAgger persistent collector protocol | distill runtime + UniLab IPC interfaces | S1/S2 / T-contract+connect | PASS | E34; 4 protocol tests, 53 IPC/runtime tests, and 280 distill/workflow/script tests pass |
| HP-2 DAgger outer barrier adapter | workflow owner + persistent service interface | S1/S2 / T-contract+connect | PASS | E35; OFF manifest unchanged, ON barrier trace and real spawned-runner connector pass |
| HP-3 persistent distillation runtime connector | Hydra/script + distill runtime owner | S0/S2/S4 / T-contract+connect+live | PASS | E36-E40; OFF-default connector, resident student, exact resource cache, dataset differential, real G1 lifecycle |
| HP-3a Hydra/script connector | config + entrypoint owner | S0/S2 / T-contract+connect | PASS | E36; default legacy, injected ON route, cleanup, and missing production factory fail-closed |
| HP-3b real persistent runtime | distill runtime + SharedWeightSync | S1/S2/S4 / T-order+connect+live | PASS | E37-E40; one resident student plus exact cached G1 teacher/env resources and cleanup report |
| HP-3b1 shared resident student | persistent runtime + SharedWeightSync | S1/S2 / T-order+connect | PASS | E37; versions 1/2 and exact worker weight sums 3/9 in same spawned PID |
| HP-3b2 persistent role resources | distill runtime + collector/env owners | S2/S4 / T-connect+live | PASS | E38-E40; exact identities, reset isolation, semantic differential, production factory, bounded MuJoCo and close counters |
| HP-4a structured metrics contract | `distill/performance.py` | S1/S3 / T-contract+persist | PASS | E41; 16 fake-clock, identity, strict reload, roundtrip, duplicate, missing-stage, and derived-rate tests |
| HP-4 runtime metrics integration | collector/worker/workflow/offline/trainer connectors | S2/S3/S4 / T-connect+persist+live | PASS | E61 eight formal runs persist complete request/workflow/learner/checkpoint/cleanup timing with lifecycle counters |
| HP-4 entry identity gate | governance + run artifact owner | S0/S3 / T-persist+oracle | PASS | E60 binds deterministic r8 source/output identity to oracle v2 before execution; output absent |
| HP-4a2 request artifact connector | runtime/collector/workflow/offline/trainer + metrics owner | S1/S2/S3 / T-connect+persist | PASS | E42-E46; both request modes plus workflow/learner/cleanup -> full identity -> atomic JSON/reload/resume |
| HP-4b bounded legacy/persistent A/B | formal entrypoint + workflow/runtime owners | S2/S4 / T-connect+live+diff | PARTIAL | E61 8/8 execution, oracle, semantic, lifecycle and timing artifacts pass; paired direction is unstable, so no stable speedup conclusion |
| HP-4b fork scenario identity repair | workflow fork + data annotation owners | S1/S2 / T-contract+connect | PASS | E51 preserves scenario/row-role identity, source hashes unchanged, 288 passed and 8 skipped; refreeze remains separate |
| HP-4 Gate 0B r8 executable freeze | governance + frozen run artifact owner | S0/S2/S3 / T-persist+oracle | PASS | E60 deterministic bundle/identity, frozen oracle, build/import/teacher/compose/493-test preflight, no training |
| HP-4b persistent output materialization | workflow owner | S1/S2/S4 / T-contract+connect+live | PASS | E60 no-mkdir spawned RED/GREEN proves iteration parent exists before persistent dispatch; live formal rerun remains HP-4b |
| HP-4c bottleneck verdict | evidence/governance owner | S3 / T-diff+scale | PASS | E62 no stable recurring owner: cleanup is once/invocation, two resource cache misses are once/worker, warm residual ~2.25 ms |
| HP-4c two-iteration discriminator | governance + formal runtime owners | S3/S4 / T-diff+scale+oracle | PASS | E65 accepted pair confirms cold/cache/cleanup amortization; no warm HP-5 owner or stable speedup claim |
| HP-4c iteration-aware oracle v4 repair | acceptance-oracle owner | S0/S3 / T-oracle+persist | PASS | E64 frozen v4/amendment, accepted existing order 1, all 16 training files unchanged, order 2 absent |
| HP-4c r9 persistent order-2 resume | formal runtime + acceptance owners | S0/S4 / T-persist+live+oracle | PASS | E65 exact preflight, persistent exit zero, oracle v4 accepted, execution complete, no legacy rerun |
| HP-4c r10 repeated two-iteration freeze | governance + benchmark identity owner | S0/S3 / T-diff+persist+oracle | PASS | E66 8 balanced orders, per-order compose/oracle identity, frozen decision rule, empty output, no-training preflight |
| HP-4c r10 eight-run execution | formal runtime + acceptance owners | S4 / T-live+persist+oracle+diff | PASS | E67 8/8 command+oracle acceptance; primary verdict NO_STABLE_SPEEDUP; no HP-5/default-on |
| HP-6a production readiness | repository + governance owners | S1/S2/S3 / T-review+contract+lint | PASS | E69/E71 resolve status drift; E70 owner probe, 537 pass/24 skip, targeted Ruff |
| HP-6a1 runtime audit-status repair | async/performance + current atlas evidence owners | S0/S1 / T-static+lint+atlas | PASS | E69 source/Method-to-Code plus E71 whole-Architecture closure |
| HP-6a production readiness restart | repository + governance owners | S1/S2/S3 / T-review+contract+lint | PASS | E70 executable gates green; E71 resolves U-RT-06/U-RT-08 cross-file blocker |
| HP-6a2 Runtime Atlas status repair | runtime atlas + checker owners | S0/S1 / T-static+contract+atlas | PASS | E71 durable semantic RED/GREEN, zero stale current-atlas hits, registry consistency |
| HP-6b repository-wide production gate | repository Makefile owners | S0/S1/S2/S3 / T-full-sweep+type+persist | PASS | E86 exact make test-all: 1556 passed, 51 skipped; static gates pass; coverage 70% |
| HP-6b1 repository lint-owner repair | section-8 diagnostic owner | S0 / T-static+lint | PASS | E73 two dead main locals removed; compile/Ruff/AST owner assertion pass |
| HP-6b2 mechanical diff review + full rerun | repository formatter + Makefile owners | S0/S1/S2/S3 / T-diff+type+full-sweep | BLOCKED | E74 AST review pass, format/Ruff pass, mypy 20 errors/8 files; pyright/coverage not run |
| HP-6b3 branch-owned type repair | DAgger collector/runtime/workflow/G1 owners | S0/S1/S2 / T-type+connect+oracle | PASS | E75 zero scoped mypy errors; Ruff pass; 111 affected tests pass |
| HP-6b4 HEAD-baseline type repair | distill model/playback/data + G1 config owners | S0/S1/S2/S3 / T-type+persist+connect | PASS | E76 mypy/Ruff pass; 442 passed, 3 skipped |
| HP-6b5 final repository rerun | repository Makefile owners | S0/S1/S2/S3 / T-full-sweep+type+persist | BLOCKED | E77 format/Ruff/mypy pass; Pyright 6 collector errors; coverage not run |
| HP-6b6 collector Pyright narrowing | distillation collector owner | S0/S1/S2 / T-type+contract | PASS | E78 Pyright 0; mypy/Ruff pass; 86 collector tests pass |
| HP-6b7 final repository rerun | repository Makefile owners | S0/S1/S2/S3 / T-full-sweep+type+persist | BLOCKED | E79 static gates pass; test-cov 14 failed, 1544 passed, 49 skipped |
| HP-6b8 G1 gait-config compatibility repair | G1 reward-config accessor owner | S0/S1/S2 / T-type+compat+contract | PASS | E80 exact ten G1 failures pass; mypy/Pyright/Ruff pass |
| HP-6b9 remaining four-failure diagnosis | Stewart/docs/CLI owners | S0/S1/S2 / T-diagnostic+diff | PASS | E81 provider, generated-doc, and execution-env owners separated |
| HP-6b10 Motrix provider/test selection | Stewart test owner | S1/S2 / T-provider+integration | PASS | E82 4 passed, 2 optional-provider skips; Ruff pass |
| HP-6b11 generated support matrix | derived docs owner | S0/S1 / T-generated+docs | PASS | E83 exact two generated SAC rows; docs contract pass |
| HP-6b12 uv env test isolation | CLI test owner | S0/S1 / T-env+contract | PASS | E84 target passes under frozen outer env; Ruff/diff pass |
| HP-6b13 combined 14-regression closure | G1/Stewart/docs/CLI owners | S1/S2/S3 / T-regression+integration | PASS | E85 12 passed, 2 expected provider skips |
| HP-6b14 final repository rerun | repository Makefile owner | S0/S1/S2/S3 / T-full-sweep+type+coverage | PASS | E86 1556 passed, 51 skipped; mypy/Pyright/Ruff pass; coverage 70% |
| HP mainline owner-aware merge | mainline integration owner | S0/S1/S2/S3 / T-merge+full-sweep | PASS | E87 commit `06d31ad6`; 1578 passed, 30 skipped, 256 deselected; Ruff/mypy/Pyright pass |
| Persistent server runtime identity | workflow/runtime metrics owners | S3/S4 / T-live+persist+identity | PASS | E88 stable collector PID `1127593`; workflow PID `1127462`; per-iteration weight versions 1/2/3 |
| Persistent live learner bottleneck | offline learner + metrics owners | S3/S4 / T-live+profile | PARTIAL | E89 batch staging 515.90 s and about 61% of iteration-2 workflow; staging sub-owner split unmeasured |
| HP-7 advanced learner-staging optimization | offline/data/performance owners | S0/S3 / T-plan+live-owner | PASS | E92-E99: owner cache, formal integration, production sentinel, and one frozen 12320-update bounded workflow PASS; no end-to-end A/B or promotion claim |
| HP-7a learner-staging discriminator | offline/data owners | S1/S3 / T-benchmark+semantic-diff | PASS | E92: current 31.835 s, cached 1.336 s, 23.83x; pool build is 93.8%; all four semantic differentials pass |
| HP-7b immutable label-pool cache design | offline sampler owner | S0/S1 / T-design+memory-bound | PASS | E93 freezes one invocation-local CPU/int64 pool cache per loaded dataset, exact RNG equivalence, `8N` payload bound, owner files, three evidence gates, and HP-7c stop |
| HP-7c implementation and formal validation | offline/data/workflow owners | S1/S2/S3/S4 / T-regression+benchmark+formal | PASS | E99: frozen r6 one-iteration workflow accepted, 12320 updates, checkpoint/metrics/cleanup complete; staging 9.32% of wall |
| HP-7c bounded persistent workflow freeze | governance/workflow/acceptance owners | S0/S3/S4 / T-identity+oracle+live | PASS | E99: r6 freeze/oracle/supervisor accepted before and after one bounded workflow; no second run or promotion authorized |
| Formal DAgger training plan | governance + active training contract | S0 / T-plan+lineage | PASS | `plans/formal_dagger_training_identity.md`: clean lineage starts from original parent iteration 3; r6 sentinel excluded |
| Formal DAgger two-round r1 spec | formal identity owner | S0/S1 / T-workload+lineage+output | PASS | E103: parent iter 3, rows 853504/855040, updates 12320/12352, total 24672, fresh output root, no training |
| FT-0 aggregate workload discriminator integration | offline replay owner + deploy connector | S1 / T-dataset+replay+fail-closed | PASS | E104: real aggregate labels drive per-iteration schedule; mismatch enters freeze failures; 23 focused tests pass |
| FT-0 formal identity materialization | distill formal-identity owner + deploy connector | S0/S1 / T-identity+compose+oracle | PARTIAL | E101-E102 owner and local connector PASS; reviewed formal spec plus authenticated server no-training materialization/preflight pending |
| FT-1 formal DAgger execution | formal CLI/workflow/runtime/artifact owners | S2/S3/S4 / T-formal+artifact+live | PENDING | Closed until separately authorized FT-0 PASS; one output identity, no retry/resume/promotion |
| Formal candidate RT-10 physical acceptance | playback/physical acceptance owner | S4 / T-physical | PENDING | Separate post-training gate; training artifact PASS is not physical PASS |

## Current Stop Condition

RT-7 workflow dispatch and manifest/quota tests pass, and RT-8 has now executed
the real bounded workflow. The physical acceptance gate remains blocked:
E28 shows repeated student termination and failure to decay speed after stop.
E29 isolates the first failed boundary as transition-state coverage and
training exposure. E30 guards that boundary in collection and offline
training. E31 shows the strengthened contract is exercised in a reused-role
bounded retrain, but the physical student gate still fails. Do not promote
this candidate or start an unbounded run. E21 confirms standing-teacher
authority; it does not prove student recovery.
Dataset reuse must remain role- and manifest-validated. A large offline update
count is not evidence of multiple DAgger iterations.

Current closure: E61/E65/E67 complete the formal A/B, two-iteration
amortization, and repeated discriminator. The verdict is `NO_STABLE_SPEEDUP`,
so no HP-5 owner or default-on promotion is authorized. E86 passes the complete
repository S0-S3 gate. `DISTILL-TRAIN-v003` records the integrated runtime as
active contract semantics while promotion is deferred and `legacy` remains the
default. RT-10 physical acceptance, optional Motrix runtime, and slow/S4 tests
remain separate open boundaries.
