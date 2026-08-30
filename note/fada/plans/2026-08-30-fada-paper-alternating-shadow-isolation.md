# FADA Paper-Aligned Alternating Training Repair

> Status: USER-APPROVED / ONE-SHOT LOCAL CONSTRUCTION

## Objective

Restore the paper's per-iteration `IDM pass -> fixed-IDM Planner pass` source-training loop and
retire the later permanent-IDM-freeze route that sends an incompletely trained Planner into the
Collector after one outer iteration.

## Preserved behavior

- v022 live-privileged grouped-DR SAC teacher, Reward, curriculum, normalization, and admitted
  20+1 Oracle lineage.
- Planner-IDM `66/29/3`, `H=30`, `K=6`, causal future-action pairing, final-Oracle Planner labels,
  intermediate-Oracle IDM diversity, and first-action receding-horizon execution.
- Raw action coordinates and `action_scale` identity. No action clip, clamp, min-max transform,
  output squashing, or other corrective limiter is added.
- Persistent async runner lifecycle and existing checkpoint inference compatibility.

## Engineering boundary

1. `FADATrainer` remains the unique optimizer-order owner. The formal source route performs IDM
   updates first and then Planner updates while IDM parameters are temporarily non-trainable only
   for the Planner pass. Permanent Planner-only freezing is retired from the formal route.
2. The FADA collector remains the trajectory and same-state Oracle-shadow owner. Its exact-snapshot
   transaction remains unchanged because MuJoCo per-environment domain-randomization parameters
   reside in the owning BatchEnvPool and are not transferable through the current rollout snapshot.
3. Existing collection windows and replay roles remain the data contract. Trajectory pairs train
   IDM; final-Oracle first actions train Planner; intermediate Oracles never gain Planner-label
   authority.
4. Existing checkpoint and diagnostics owners are extended only where the restored schedule needs
   identity or reporting. Oracle paths are resolved from the existing lineage gateway/config, not
   by a second persistence abstraction.

## Implementation steps

1. Add RED cases for formal-route rejection of split training and positive IDM/Planner update
   budgets; retain the existing ordered-update and temporary-freeze regressions.
2. Retire `planner_from_idm` from formal configuration/workflow branches and route v022 source
   training through `alternating_idm_then_planner`; remove the split-stage IDM initialization
   gateway so formal training always starts and persists the paired policy.
3. Keep source batches and their role, shape, dtype, finite, action-scale, and weight-version
   validations unchanged at their existing boundaries.
4. Persist the alternating schedule and paired optimizer state through the existing schema-5
   checkpoint owner. Historical split-stage schema-5 checkpoints remain inference-readable but
   cannot enter formal source training.
5. Keep the existing collection and training diagnostics, now with both IDM and Planner loss,
   gradient norm, and update counts active on every iteration.
6. Run focused RED/GREEN tests, affected FADA suites, Ruff, type checks used by the repository, and
   `git diff --check`. No simulator, server, long training, commit, push, or branch creation occurs.

## Heisenbug stabilization closure

- Victim: NumPy validation in `fada_windows._validate_records`.
- Trigger: causal-window construction after repeated native environment operations.
- Owner: source-training schedule and student weight publication into the Collector.
- Corrupter candidates: the permanent-IDM-freeze route publishes an undertrained Planner whose raw
  actions drive native MuJoCo outside the teacher-supported rollout distribution; NumPy validation
  later observes damage as the victim.
- Closure: remove the invalid formal route and restore the already-owned alternating update and
  publication invariant. The exact native writer remains unproven until one bounded server run.

## Proof and stop conditions

- RED must fail because the current formal route admits permanent IDM freezing and publishes the
  Planner-only student after one iteration.
- GREEN must prove ordered parameter deltas, temporary freeze restoration, formal-route rejection
  of split training, causal role preservation, checkpoint identity, and no action transformation.
- Local evidence cannot prove server-native crash elimination or policy quality. The next genuine
  boundary after local closure is one bounded server runtime audit, separately authorized.
- Stop if implementation would change teacher behavior, Reward, domain randomization, privileged
  normalization, action definition, or require a new checkpoint schema incompatible with playback.
