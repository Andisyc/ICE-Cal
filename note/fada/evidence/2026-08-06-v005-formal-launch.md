# v005 Formal Training Launch

Date: 2026-08-06

Remote run: `/ssd1/cyx/FADA_runs/20260806_planner_idm_v005`

The isolated remote repo was copied from the completed v004 run and updated only with the validated
v005 FADA owners, config, and focused tests. Remote focused regression passed with `42 passed`.

The campaign uses the final walking Oracle, final standing Oracle, and exactly 20 same-lineage
intermediate walking checkpoints (`model_240.pt` through `model_4800.pt`). It enables paper source,
Oracle shadow, persistent async collection, exact standing cold-start coverage, and Planner replay
ratios `50/25/25` with static `50/50`.

The v004 Planner/IDM checkpoint is supplied through `initial_weights_path`. Startup checkpoint
inspection recorded schema 2, `completed_iterations=0`, `samples_seen=0`, and empty quality metrics,
confirming weights-only initialization with fresh optimizers, replay, counters, and evidence.

Startup acceptance passed with parent PID `17228`, collector PID `17291`, and GPU 0 allocation.
Monitoring stopped immediately after this gate at the user's request. No iteration-completion or
closed-loop quality claim is made.

## Completion

The campaign later completed `8/8` iterations with `1,572,864` samples and all eight source
artifacts. The final checkpoint SHA-256 is
`d35a32d93b0387e534f6fcdd86b724c44187e308dbca1412435bffe95b6ed90c`. It was pulled to
`/Users/sss9999/locomotion/FADA/planner_idm_v005.pt`; the local hash matched, strict inference loading
restored 16 quality metrics, and a zero-input probe produced a finite `(1,29)` action. These checks
accept training completion and artifact integrity, not closed-loop stability.
