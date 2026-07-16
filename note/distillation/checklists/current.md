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
| Active single-entry training contract | distillation contract registry | S0 / T-contract | PASS | `contracts/active/training/DISTILL-TRAIN-v002.md`; v001 archived |
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
| Transition training contract v002 | docs governance | S0 / T-contract | PASS | E23; active registry points to v002 and v001 is archived |
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
| RT-10 bounded retrain with repaired transition rollout | workflow + checkpoint + MuJoCo gate | S2/S4 / T-formal+live | PENDING | Reuse role artifacts, collect new transition data, then rerun the physical grid |

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
