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
| Active single-entry training contract | distillation contract registry | S0 / T-contract | PASS | `contracts/active/training/DISTILL-TRAIN-v001.md` |
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
| Playback standing reset identity | `play_interactive.py` + G1 reset provider | S2/S4 / T-connect+live | PARTIAL | root cause archived; implementation and repeated-reset gate pending |
| Low-speed walking activation | playback + policy | S4 / T-live | PARTIAL | command routing contract exists; physical quality unaccepted |
| Walk-to-stop recovery | transition-conditioned DAgger proposal | S2/S4 / T-connect+live | BLOCKED | distribution gap archived; transition semantics and implementation unconfirmed |
| Height teacher integration | no current teacher | S0-S4 | BLOCKED | qualified height checkpoint absent |
| Promoted checkpoint selection | no owner | S2/S4 | PENDING | proposal only |

## Current Stop Condition

Do not recommend another policy-quality training run until the playback reset
fix and transition-conditioned collection have approved owners and acceptance
entries. The single-entry workflow and identity manifest now pass their code
gates, but they do not remove those physical distribution gaps. Dataset reuse
must remain role- and manifest-validated. A large offline update count is not
evidence of multiple DAgger iterations.
