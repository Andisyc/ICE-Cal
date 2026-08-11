# FADA Async v2 Checkpoint Validation

Date: 2026-08-05

## Training completion

- Remote run: `/ssd1/cyx/FADA_runs/20260805_planner_idm_async_v2`.
- Execution mode: `persistent_async`.
- Completed iterations: 8/8.
- Samples seen: 1,572,864.
- Source artifacts: `iteration_0000.pt` through `iteration_0007.pt`, 32 GB total.
- Local checkpoint: `/Users/sss9999/locomotion/FADA/planner_idm_async_v2.pt`.
- Remote/local SHA-256: `434b0a89c7fc0b25820252f02c75577f3018dfa5ceae12683249115871809b56`.

Strict schema load, finite parameter checks, and the Planner-IDM forward-shape contract passed.
Playback and interactive contract tests passed, 28 tests total.

## Closed-loop differential

The live sentinel used 8 G1WalkFlat MuJoCo environments, command `[0.4, 0.0, 0.0]`, autoreset
disabled, a 500-step target, and task-owned failure limits of 0.3 m minimum base height and 65
degrees maximum tilt.

| Policy | Seeds | Completed steps | Min height | Mean vx after warmup | Max abs action | Result |
|---|---:|---:|---:|---:|---:|---|
| FADA async v2 | 1, 2, 3 | 138, 178, 101 | 0.2714, 0.2698, 0.2927 | 0.613, 0.572, 1.014 | 1.485, 1.271, 1.448 | failed |
| Final Oracle | 1, 2, 3 | 500, 500, 500 | 0.6916, 0.6780, 0.6986 | 0.376, 0.373, 0.375 | 0.844, 0.896, 0.831 | passed |

The same environment and command are stable under the final Oracle, so the FADA failures are not
explained by the live sentinel or G1 environment. Training completion and checkpoint connectivity
are live-confirmed; closed-loop Planner-IDM policy quality is rejected.
