# FADA v016 Single-Reward Migration Plan

> Status: IMPLEMENTED / MODULE-CORRECT; formal runtime not executed.

## Objective

Retire v015 command-conditioned dual Reward and restore one phase-neutral locomotion Reward shared
by zero and nonzero commands. Preserve 98-D Actor compatibility, Planner–IDM semantics, the
privileged Critic tail, left-knee Gain distribution, and 20+1 lineage.

## Affected engineering boundary

1. Replace the nominal dual-Reward profile with a single-Reward, no-phase profile derived from the
   successful basic locomotion configuration.
2. Make the privileged profile inherit that nominal profile and add only privilege/Gain.
3. Make privileged preflight reject mode dispatch and every `stand_*` Reward term or override.
4. Replace v015 Reward-routing tests with configuration-absence, tensor-preservation, and temporal
   preference tests: survival must outrank early termination under matched commands.
5. Run module evidence, then formal runtime, then a bounded nominal policy-quality campaign before
   any privileged long run.

## Engineering Boundary Record

- **Accepted behavior:** one phase-neutral locomotion Reward is evaluated for every command; command
  changes only the velocity-tracking target.
- **Preserved behavior:** the shared G1 environment may retain its legacy Reward-mode implementation
  for other tasks; Actor 98-D, state66/previous-action29/command3, privileged Critic tail, left-knee
  Gain distribution, SAC optimizer, and 20+1 checkpoint lineage do not change.
- **Semantic owner:** the nominal Hydra task profile owns the effective Reward; the privileged
  profile may only inherit it and add privilege/Gain; privileged runtime preflight owns admission.
- **Public boundaries:** Hydra compose output, `validate_fada_single_reward`, and
  `FADAPrivilegedSACRuntime.validate_training_config`.
- **Forbidden dependencies:** no Reward decision in `scripts/`, no new runner/backend interface, no
  second Reward formula in tests, and no hidden fallback to the retired dual-Reward profile.
- **State/lifecycle:** no new mutable state; the existing disabled gait-clock lifecycle must keep the
  two compatibility slots zero at reset and after every step.
- **Legacy isolation:** remove the v015 task profile and v015 validator from the active FADA route;
  retain generic G1 mode/stand code because repository consumers still exist.
- **Failure behavior:** enabled `reward.mode`, any nested `stand_*` Reward authority, nonzero phase
  Reward, or enabled gait constraint rejects before environment creation.
- **Proof route:** config composition and semantic preflight tests first; preserved tensor/phase and
  Planner-IDM tests next; official environment/update connectivity remains a later formal audit.

## Affected Module Set

| ID | Relation | Obligation | Proof destination |
|---|---|---|---|
| `NOMINAL-SINGLE-REWARD-CONFIG` | config/semantic/gradient/runtime | replace v015 profile | Hydra composition tests |
| `PRIVILEGED-ORACLE-CONFIG` | config/semantic/persistence/runtime | change inheritance only | Hydra identity tests |
| `FADA-SINGLE-REWARD-PREFLIGHT` | call/config/semantic/runtime | replace dual validator | S1 fail-closed tests |
| `G1-PHASE-LIFECYCLE` | data/semantic/runtime | verify preserved | existing zero-slot tests |
| `FADA-INPUT-CONTRACT` | data/gradient/persistence | verify preserved | 98→66/29/3 regression |
| `ORACLE-CHECKPOINT-LINEAGE` | persistence | verify preserved | existing 20+1 tests |

No model, learner, optimizer, backend, asset, replay, or checkpoint-format owner is modified.

## Execution steps

### Task 1 — Freeze executable semantics

1. Confirm the v016 Reward Ordering Card and Module Test Cards.
2. Add tests that require a single composed Reward, reject `reward.mode`, reject nested `stand_*`,
   and preserve phase/tensor identities.
3. Run the focused tests and observe failure specifically because v015 is still active.

### Task 2 — Migrate the owner configuration

1. Replace `mujoco_no_gait_dual_reward.yaml` with `mujoco_no_gait_single_reward.yaml`.
2. Keep the successful base locomotion terms, zero all phase terms, disable gait constraint, and
   remove every `stand_*` and `reward.mode` field.
3. Point the privileged profile at the new nominal owner without duplicating Reward values.

### Task 3 — Replace the admission guard

1. Replace `validate_fada_no_gait_dual_reward` with `validate_fada_single_reward`.
2. Reject enabled mode dispatch, recursively reject `stand_*` authority, and retain no-gait/gait-
   constraint checks.
3. Keep the guard in the existing FADA semantic owner; add no wrapper, registry, or schema.

### Task 4 — Prove GREEN and review

1. Run the new focused RED tests, then the affected module suite and static checks with `uv run`.
2. Update stale v015 formal-test semantics without executing its simulator route in this unit.
3. Refresh module registry, evidence, one-shot receipt, and final R2 maintainability review.
4. Stop before formal runtime, simulation, policy-quality evaluation, server work, or training.

## Stop conditions

- Any active gait/feet-phase authority.
- Any separate standing/walking/recovery Reward family.
- A return improvement accompanied by shorter episodes or persistent 100% termination.
- Any change to Planner–IDM tensor, causal, optimizer, or checkpoint semantics.
