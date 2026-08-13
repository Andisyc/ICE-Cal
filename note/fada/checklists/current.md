# FADA Current Checklist

| Acceptance | Owner | Tier / kind | Status | Evidence |
|---|---|---|---|---|
| Paper default dimensions and output shapes | `fada.py` | S1 / unit | passed | `E2`, `test_paper_defaults_and_policy_shapes` |
| Planner residual future reconstruction | `FADAPlanner` | S1 / unit | passed | `E2`, `test_planner_head_is_residual_to_latest_observation` |
| IDM non-causal future attention reaches first action | `FADAInverseDynamicsModel` | S1 / gradient | passed | `E2`, `test_idm_first_action_can_depend_on_later_future_tokens` |
| Eq. 4.2 ignores non-first action targets | `first_action_mse` | S1 / semantic | passed | `E2`, `test_first_action_loss_ignores_nonexecuted_chunk_entries` |
| Eq. 4.2 and 4.3 keep targets and gradient owners separate | source loss helpers | S1 / semantic | passed | `E2`, `test_source_losses_route_causal_and_oracle_targets_separately` |
| Eq. 4.2 includes valid Oracle-shadow rows and ignores invalid rows | `idm_source_loss` | S1 / semantic | passed | `E19`, `test_idm_source_loss_uses_only_valid_oracle_shadow_rows` |
| Invalid window shapes fail closed | `FADASourceBatch` | S1 / contract | passed | `E2`, `test_source_batch_rejects_noncausal_shape_mismatch` |
| Architecture/Inspector schemas remain valid | Atlas | S1 / document | passed | `E5`, `npm --prefix note/architecture/auxiliary/atlas_app run check` |

The rows above close the reusable model boundary; the active UniLab integration proof is tracked below.

## UniLab training integration

| Acceptance | Owner | Tier / kind | Status | Evidence |
|---|---|---|---|---|
| Legacy distill routing is unchanged with FADA disabled | `train_distill.py` | S2 / OFF regression | passed | `E6`, OFF dispatch plus legacy characterization |
| FADA config activates as one validated parameter family | config + composition root | S2 / contract | passed | `E7`, flat-task Hydra compose |
| Windows remain inside one episode and one future command | FADA collector | S1 / provenance | passed | `E6`, command and done-boundary tests |
| IDM targets executed action, Planner targets Oracle action | FADA collector/trainer | S1 / semantic | passed | `E6`, causal bootstrap and target tests |
| Iteration zero uses Oracle and later iterations use Planner-IDM | FADA workflow | S2 / DAgger connectivity | passed | `E6`, two-iteration fake UniLab env route |
| IDM and Planner update in separate ordered passes | FADA trainer | S1 / gradient | passed | `E6`, trainer/replay test |
| Paired module/optimizer checkpoint and quality metrics persist consistently | FADA checkpoint | S1 / persistence | passed | `E19`, schema-v2 checkpoint round-trip |
| Formal UniLab entrypoint reaches FADA owner only when enabled | `train_distill.py` | S2 / connectivity | passed | `E6`, OFF/ON dispatch test |
| Formal FADA config selects UniLab persistent async collection | config + composition root | S2 / routing | passed | async branch test, `execution_mode=persistent_async` default |
| One worker request collects current-policy and intermediate-Oracle batches at one weight version | FADA persistent worker | S2 / barrier | passed | worker iteration-artifact test |
| Collector artifact validates schema, architecture, row count, and source metadata before replay | FADA artifact owner | S1/S2 / persistence | passed | artifact round-trip and async parent-boundary tests |
| Real G1WalkFlat MuJoCo rollout enters FADA training | Oracle + FADA workflow | S3 / live sentinel | passed | `E11-E12`, remote Oracle audit and two-iteration MuJoCo run |
| Environment reset keeps backend and `_state` command/observation carrier aligned | `NpEnv.reset_all` + collector | S1 / lifecycle | passed | `E19`, authoritative reset regression |
| Same-state Oracle shadow restores physics, env carrier, counters, RNG, pending forces, and autoreset | `NpEnv` + MuJoCo backend | S1/S3 / transaction | passed | `E19`, exception restoration and real MuJoCo shadow sentinel |
| Intermediate Oracle rollout actions remain separate from final-Oracle Planner labels | collector | S1 / provenance | passed | `E19`, intermediate-label separation test |
| Paper-exact route requires 20 unique intermediate checkpoints and 2:1 budget before env creation | FADA composition root | S2 / fail-closed | passed | `E19`, exact-set preflight test and live CLI failure |
| Saved checkpoint reports the four source-boundary quality metrics | evaluator + checkpoint | S1/S3 / quality | passed | `E19`, evaluator test and old-checkpoint real MuJoCo diagnosis |
| Paper-default v003 source campaign completes with required evidence | formal UniLab run | S4 / training | passed | Async v2 completed 8/8 with 1,572,864 samples and 8 source artifacts |
| v003 checkpoint passes explicit closed-loop stability threshold | FADA playback evaluation | S4 / acceptance | failed | FADA terminated at steps 101-178 across seeds 1-3; final Oracle passed 500/500 |

The async v2 training run is complete and its checkpoint contract is accepted, but its closed-loop
policy quality is rejected. Training completion must not be presented as walking-stability acceptance.

## Standing and walk-to-stand curriculum

| Acceptance | Owner | Tier / kind | Status | Evidence |
|---|---|---|---|---|
| Disabled curriculum preserves the walking-only FADA route | config + FADA worker | S1/S2 / OFF regression | passed | `test_fada_unilab_training.py` focused suite |
| Main-source quotas are exact across walk, static stand, and walk-to-stand | allocation owner | S1 / contract | passed | stable-largest-remainder and artifact tests |
| Static-standing windows use zero commands, standing Oracle labels, and `G1StandStill` | FADA worker + collector | S1/S2 / provenance | passed | standing-window and dual-environment routing tests |
| Walk-to-stand windows retain active history and an all-zero future command | FADA collector | S1 / temporal contract | passed | transition-window test |
| Missing standing Oracle or standing environment fails before source mutation | FADA runtime/worker | S1/S2 / fail-closed | passed | negative checkpoint, Oracle, and environment tests |
| Standing owner composes with standing reset distribution | Hydra owner config | S2 / composition | passed | `G1StandStill`, MuJoCo, reset qvel `0.0`, standing fraction `1.0` probe |
| v004 formal standing curriculum completes | formal UniLab run | S4 / training | passed | completed v004 checkpoint pulled locally |
| v004 checkpoint passes closed-loop walk/stand/transition stability | FADA playback evaluation | S4 / acceptance | failed | Planner diverges during cold-start/walk before safe standing-Oracle handoff |

## v005 cold-start and replay repair

| Acceptance | Owner | Tier / kind | Status | Evidence |
|---|---|---|---|---|
| Static standing contains reset-aligned cold-start and steady windows at 50/50 | FADA collector/worker | S1/S2 / provenance | passed | exact cold-start and worker artifact tests |
| Every source row persists scenario, Planner eligibility, and cold-start identity | source artifact | S1 / persistence | passed | schema-v2 artifact round-trip and row validation |
| Planner batches are fixed 50/25/25 with static 50/50 | FADA replay | S1 / sampling | passed | exact quota and config-drift rejection tests |
| Intermediate walking Oracle rows train IDM but cannot enter Planner replay | collector + replay | S1 / provenance | passed | eligibility and missing-stratum tests |
| Parent rejects missing or inconsistent v005 row provenance before replay mutation | async parent boundary | S1/S2 / fail-closed | passed | parent artifact positive/negative tests |
| Production checkpoint requires finite per-scenario and cold/steady metrics | evaluator + serializer | S1 / evidence | passed | metric split and serializer fail-closed tests |
| v004 initialization restores weights only | FADA composition root | S1/S2 / initialization | passed | optimizer construction follows strict compatible policy load; resume is exclusive |
| Bounded real MuJoCo v005 source sentinel | UniLab runtime | S3 / live sentinel | passed | 8 rows=`4/2/2`, static=`1/1`, shadow-valid=`1.0`, production checkpoint sealed |
| Formal v005 persistent-async training | formal UniLab run | S4 / training | passed | completed `8/8`, `1,572,864` samples; local checkpoint hash verified and strict-loaded |
| v005 passes three-scenario, three-seed closed-loop acceptance | FADA playback | S4 / acceptance | pending | only after formal training |

## Historical Context Phase-1 privileged residual teacher

| Acceptance | Owner | Tier / kind | Status | Evidence |
|---|---|---|---|---|
| Existing SAC and G1 observations remain unchanged when Phase-1 is disabled | config + G1 env | S1/S2 / OFF regression | passed | 2026-08-10 Phase-1 evidence, `177 passed` regression |
| Reusable per-reset `g` sampler covers nominal and candidate-knee rows and matches applied gains | G1 DR owner | S1/S3 / intervention | passed | v002 64-env MuJoCo reset, runtime Kp error `0.0`; v003 narrows active candidate set |
| Actor observation excludes `g`; critic observation appends exactly 29 values | G1 observation owner | S1 / privilege boundary | passed | `(64,98)/(64,130)` MuJoCo observation evidence |
| Frozen nominal SAC plus zero residual reproduces nominal action | residual actor | S1 / semantic | passed | exact-equality actor contract test |
| Only residual branch receives actor gradients | residual learner | S1 / gradient | passed | residual update and nominal no-gradient test |
| Collector, learner, replay, and playback share the fused-action rule | off-policy runtime | S2 / connectivity | passed | worker/playback tests and one-update sentinel |
| Teacher checkpoint binds nominal checkpoint identity and strict-loads | checkpoint owner | S1 / persistence | passed | hash, tensor-tamper, mismatch, and sentinel checkpoint evidence |
| One-environment MuJoCo update completes with finite losses/actions | training entrypoint | S3 / live sentinel | passed | `/private/tmp/fada-context-phase1-sentinel2-20260810/model_1.pt` |
| Broad bilateral v002 teacher improves declared paired trajectory precision metrics | evaluation owner | S4 / quality | failed | completed 5000-step run improved velocity/yaw but failed five lateral/non-degradation checks |
| Paired branches start from identical state, `g`, command, observation, and RNG | paired evaluator | S1/S3 / causal pairing | passed | 7 semantic tests and 64-env MuJoCo paired sentinel |
| Straight-line, fall, residual, and clipping metrics serialize by all three `g` strata | paired evaluator | S1/S3 / measurement | passed | `2026-08-10-context-phase1-paired-evaluator.md` and emitted JSON |
| Existing training/playback paths remain unchanged when evaluator is not invoked | explicit evaluation entrypoint | S2 / OFF regression | passed | 331 relevant regressions passed; 4 permission-sensitive cases rerun outside sandbox |
| v003 formal protocol accepts only nominal and fixed-left-knee-0.9 strata | formal protocol owner | S1/S3 / acceptance | passed | 28 local + 28 remote owner tests; formal protocol match |
| v003 trajectory reward measures lateral/yaw error in the episode-start frame | G1 reward owner | S1/S3 / semantics | passed | owner frame test, 64-env finite MuJoCo step, CUDA preflight |
| v003 formal Hydra profile drift fails before environment creation | runtime composition root | S1/S2 / fail-closed | passed | local and CUDA preflight exact-profile match |
| v003 formal privileged-teacher training produces an accepted checkpoint | UniLab runner | S4 / training + quality | failed | 5000/5000 completed; formal gate failed seven trajectory checks (`2026-08-11-context-phase1-v003-fixed-left090.md`) |
| Left-knee gain scaling creates a repairable baseline failure | frozen nominal policy | S3 / intervention validity | failed | five-seed scan at `1.0/0.9/0.8/0.7` showed no degradation and zero falls |

## Context latent-repair method

The rows in this section are retained as evidence for the stopped differentiable-dynamics route.
They are not current ICA acceptance criteria. ICA has no active method/training contract yet.

| Acceptance | Owner | Tier / kind | Status | Evidence |
|---|---|---|---|---|
| Context output is latent residual `delta_z`, fused exactly as `z_repaired = z + delta_z` before Decoder | historical Context method | note / semantic | superseded | `FADA-CONTEXT-METHOD-v003`, differentiable-trajectory decision |
| Fault probe trajectory is Context input; healthy same-command trajectory is the reference | rollout lifecycle | note / semantic | note-confirmed | `FADA-CONTEXT-METHOD-v003`, 2026-08-12 decision |
| Tracker Encoder and Decoder are frozen; only Context Encoder trains | parameter ownership | note / gradient contract | note-confirmed | `FADA-CONTEXT-METHOD-v003`, 2026-08-12 decision |
| Primary loss compares the differentiably predicted adapted trajectory with the healthy reference | historical Context trainer | note / supervision | rejected | 10/10 real-MuJoCo candidates rejected on 2026-08-12 |
| Actions and free/optimized `delta_z` are not treated as Context ground truth | supervision boundary | note / semantic | note-confirmed | `FADA-CONTEXT-METHOD-v003`, 2026-08-12 decision |
| Zero `delta_z` is numerically identical to the original Planner-IDM path | Context integration | S1 / equivalence | passed | `test_zero_context_policy_is_exactly_the_nominal_planner_idm_path` |
| Exact healthy `E/D` checkpoint, command, and left-knee `0.9` intervention are bound | experiment owner | S2/S3 / identity | passed | v005 SHA-256, fixed `0.4` command, same-start cross-env MuJoCo preflight |
| Fault transitions preserve continuous Decoder-reachable nominal-fault sequences | dataset owner | S1/S3 / provenance | passed | schema-v1 round trip, continuity validation, real MuJoCo preflight |
| Fault transitions cover perturbed and current-Context visited states | dataset owner | S3 / provenance | pending | requires later data aggregation after Context exists |
| Differentiable ensemble passes held-out one-step, short-horizon, and disagreement gates | dynamics owner | S1/S3 / model validity | pending | architecture and thresholds not selected |
| Context consumes only causal deployable rollout fields and never `g` | Context input owner | S1/S2 / privilege boundary | passed | dataset exposes only observation/action history and command to Context |
| Trajectory loss yields finite gradients while only Context parameters change | Context trainer | S1/S3 / gradient | passed | 2026-08-12 differentiable-core evidence; v005 checkpoint sentinel |
| Pretraining boundary constructs disjoint optimizers and performs zero parameter updates | training setup | S1/S3 / ownership | passed | real paired MuJoCo preflight, `optimizer_steps=0`, all parameters unchanged |
| Model-predicted Context improvement transfers to paired MuJoCo rollouts | Context evaluator | S3/S4 / transfer | failed | 10/10 gates rejected; mean trajectory MSE worsened 27.32% |
| Held-out and history-ablation tests rule out a constant `delta_z` | Context evaluator | S3/S4 / identifiability | pending | evaluation protocol not yet accepted |

## Active Context Support-Query method

| Acceptance | Owner | Tier / kind | Status | Evidence |
|---|---|---|---|---|
| Existing IDM exposes `encode_latent`/`decode_latent` with exact zero-residual equivalence | `FADAInverseDynamicsModel` | S1 / compatibility | passed | `test_idm_latent_split_is_exactly_original_forward`, original FADA regressions |
| Context reads full Support target/realized/action sequence and emits one `[B,128]` `delta_z` | Support Context owner | S1 / tensor contract | passed | Support-Query focused tests |
| Support and Query share exact command/fault but use different rollout identities | collector + dataset | S1/S3 / provenance | passed | fake independent-reset test; 2026-08-13 MuJoCo preflight |
| Query supervision uses fault realized future and physically executed first action only | loss owner | S1/S3 / semantics | passed | first-action invariance test; MuJoCo tensor evidence |
| Planner/IDM remain frozen and only Context receives finite gradients | training setup | S1/S3 / gradient | passed | focused gradient test; MuJoCo gradient norm `1.7284e-04` |
| Fixed-0.7 real MuJoCo data has nonzero zero-Context action loss | preflight | S3 / identifiability | passed | first-action MSE `1.8847e-05`, threshold `1e-08` |
| Offline training lowers held-out Query action MSE | Context trainer | S4 / training | pending | training not started |
| Fixed Context improves second-rollout trajectory under the same 0.7 fault | calibrated evaluator | S4 / quality | pending | post-training evaluation required |
| Cross-fault condition generalization beats a constant-delta baseline | later Context campaign | S4 / identifiability | out of scope | fixed-0.7 first stage only |

## Historical Context Phase-1 v004 full-action teacher

| Acceptance | Owner | Tier / kind | Status | Evidence |
|---|---|---|---|---|
| Teacher directly emits complete 29D action with no residual fusion | full-action actor | S1 / semantic | passed | local actor tests and CUDA preflight `full_action_output=true`, `residual_fusion=false` |
| Teacher warm start matches original actor while retaining no nominal branch | full-action actor | S1 / initialization | passed | numerical equivalence and owner-structure test |
| Actor optimizer owns every full-action teacher parameter | full-action learner | S1 / gradient | passed | optimizer identity test and one-update test |
| Formal environment is fixed left-knee `0.9` for every row | G1 DR owner | S1/S3 / intervention | passed | config contract and local/remote MuJoCo sentinels |
| Baseline and teacher evaluation restore the same fixed-0.9 snapshot | paired evaluator | S1/S3 / causal pairing | passed | positive and mixed-strength negative tests |
| Remote CUDA preflight matches v004 formal profile | composition root | S3 / preflight | passed | `preflight_cuda.json`, 2026-08-11 |
| v004 formal full-action training completes | UniLab runner | S4 / training | passed | completed `5000/5000`, `10,262,528` env steps; checkpoint SHA recorded |
| v004 teacher outperforms original policy under the same fixed-0.9 physics | paired evaluator | S4 / quality | failed | lateral/yaw improved, but progress `-0.0256 m` and forward MAE `0.4027 m/s` |

## Historical Context Phase-1 v005 forward-progress teacher

| Acceptance | Owner | Tier / kind | Status | Evidence |
|---|---|---|---|---|
| Progress termination is default-off and preserves existing G1 behavior | G1 environment | S1/S2 / OFF regression | passed | focused local and remote regression suites |
| Grace, command threshold, reset-yaw projection, and exact speed boundary are correct | G1 environment | S1 / semantics | passed | pure owner tests at step 50 and `0.20 m/s` |
| Original fixed-0.9 actor survives while stationary v004 teacher is rejected | paired evaluator | S3 / live discriminator | passed | 64 env, seed 101: baseline 60 steps/0 failures; v004 teacher 50 steps/100% failures |
| Remote CUDA preflight matches v005 formal profile | composition root | S3 / preflight | passed | `preflight_cuda.json`, full action, no residual fusion, collector not started |
| v005 formal full-action training completes | UniLab runner | S4 / training | passed | `5000/5000`, `10,262,528` env steps, final checkpoint SHA recorded |
| v005 teacher outperforms original policy under the same fixed-0.9 physics | paired evaluator | S4 / quality | failed | teacher 50 steps/`0.0282 m`/100% failure vs baseline 400 steps/`2.5806 m`/0% failure |

## Historical Context Phase-1 v006 behavior-anchored teacher

| Acceptance | Owner | Tier / kind | Status | Evidence |
|---|---|---|---|---|
| Teacher still emits one complete 29D action with no inference residual fusion | full-action actor | S1 / semantic | passed | actor contract test |
| Frozen nominal actor anchors teacher actions and receives no gradients | full-action learner | S1 / gradient | passed | zero/perturbed anchor loss and optimizer ownership tests |
| Formal profile fixes anchor `10.0`, actor LR `3e-5`, 100-step saves, and 1000-step budget | composition root | S2 / contract | passed | Hydra formal-profile test |
| Local MuJoCo actor update remains finite | training entrypoint | S3 / live sentinel | passed | focused owner suite |
| Remote CUDA no-training preflight matches v006 | composition root | S3 / preflight | passed | dimensions `(98,130,29,29)`, frozen anchor, collector not started |
| First v006 checkpoint preserves walking under fixed left-knee `0.9` | checkpoint discriminator | S4 / quality | passed | model 100: 60/60 survival, `0.3479 m` progress |
| A v006 checkpoint passes the unchanged paired quality gate | paired evaluator | S4 / quality | failed | model 100/500 both worsened maximum lateral and yaw drift |
