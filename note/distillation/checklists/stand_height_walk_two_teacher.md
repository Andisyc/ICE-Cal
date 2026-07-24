# StandHeight And Walk Acceptance Checklist

Status values: `PASS`, `PARTIAL`, `PENDING`, `BLOCKED`.

| Item | Owner | S/T evidence | Status | Evidence |
| --- | --- | --- | --- | --- |
| Active v002 contract and Concept Figure mapping | docs governance | S0 / T-contract | PASS | `npm.cmd run check`: atlas OK, 2026-07-23 |
| New task composes without changing legacy tasks | G1 config/registry | S0/S2 / T-compose+differential | PASS | E114: retained Step 2 focused suite, `108 passed, 24 warnings in 19.46s` |
| StandHeight actor observation is 99-D | G1 observation owner | S1/S2 / T-shape+connect | PASS | E114: retained Step 2 focused suite, `108 passed, 24 warnings in 19.46s` |
| Dynamic stand rewards use per-env target | G1 reward owner | S1 / T-value+metamorphic | PASS | E114: retained Step 2 focused suite, `108 passed, 24 warnings in 19.46s` |
| 98-D actor conversion is output-equivalent | checkpoint adapter | S1 / T-shape+metamorphic | PASS | E114: Step 3 adapter/connector suite, `8 passed in 6.77s` |
| Adapter metadata and hashes persist | checkpoint adapter | S1/S3 / T-persist | PASS | E114: Step 3 adapter/connector suite, `8 passed in 6.77s` |
| Critics/replay/optimizer are not migrated | off-policy training owner | S1/S2 / T-contract+connect | PASS | E114: Step 3 adapter/connector suite, `8 passed in 6.77s` |
| Walk and StandHeight rows share 99-D | distill data/workflow | S1/S2 / T-contract+roundtrip | PASS | E113: Step 4 focused suite, `27 passed in 20.56s`, 2026-07-23 |
| `target_height` survives collect/save/load/batch/multitask/DAgger | collector/data/DAgger | S1/S2 / T-shape+roundtrip | PASS | E113: `(N, 1)` roundtrip and connector fixtures |
| Legacy 98-D and v002 99-D sources remain isolated | distill data/workflow | S1 / T-negative+regression | PASS | E113: mixed dimensions and mixed `target_height` presence fail closed |
| Two experts map active->0 and inactive->1 | workflow/trainer/playback | S1/S2 / T-role+connect | PASS | E113: new profile compose and connector fixtures |
| Selected expert update isolation | behavior trainer | S1 / T-grad+regression | PASS | E113: inactive parameters and optimizer state remain unchanged |
| Two-expert checkpoint strict roundtrip | checkpoint/student | S1/S2 / T-persist+shape | PASS | E113: strict reload returns one finite 29-D action |
| StandHeight one-env live route | G1 env/sentinel | S3/S4 / T-live | PASS | E115: one-env/one-step MuJoCo snapshot and training compose preflight |
| StandHeight SAC async runner dispatch | off-policy runner | S2 / T-connect | PASS | E116: `AsyncRunner` + `DoubleBufferOffPolicyRunner` owner dispatch |
| 99-D workflow persistent runtime opt-in | distill runtime/workflow | S2 / T-connect+lifecycle | PASS | E116: two roles, three scenarios, target-height keys, and one service close; `3 passed in 3.83s` |
| Non-nominal command x height transition collection | collector/workflow | S0/S1/S2 / T-value+roundtrip+connect | PASS | E117: exact 3x3 grid, active 0.754 m, post-switch targets, index-96/teacher roundtrip, legacy and persistent connectors; `10 + 7 passed` |
| StandHeight teacher physical quality | training/acceptance owner | S4 / T-live+differential | PASS | E117: user-provided Stage-2 gate passed at 0.650/0.702/0.754 m; exact checkpoint identity retained |
| Retrained two-teacher non-nominal transitions | workflow/acceptance owner | S4 / T-live | PARTIAL | E117: round-2 parent failed 4/9 recovery cases; repaired fork not trained or evaluated |

## Stop Condition

Steps 2-4 deterministic implementation are closed by E114 and E113. Step 5's
bounded live route is closed by E115, and E116 closes the optional async
connector boundary without changing the `legacy` distillation default. E117
closes the local non-nominal collection-distribution repair. A new immutable
SSH fork and its physical acceptance remain open and require exact
checkpoint/run identity. No shape, load, or deterministic fixture result is
policy-quality evidence.
