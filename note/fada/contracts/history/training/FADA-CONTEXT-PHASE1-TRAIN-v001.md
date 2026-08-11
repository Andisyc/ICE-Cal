---
contract_id: FADA-CONTEXT-PHASE1-TRAIN-v001
status: superseded
effective_date: 2026-08-10
updated_date: 2026-08-10
superseded_by: FADA-CONTEXT-PHASE1-TRAIN-v002
method_contract: FADA-CONTEXT-PHASE1-METHOD-v001
scope: default-off privileged residual SAC teacher training with 29D actuator strength
---

# FADA Context Phase-1 Training Contract

`algo.runtime_impl=privileged_residual_sac` is the single owner selection for this route. Existing SAC
configs that do not select this runtime preserve their actor, observation dimensions, collection,
checkpoint, and playback behavior.

The Phase-1 route loads a declared standard SAC checkpoint as a frozen nominal actor and trains only
the privileged residual branch. The SAC critics consume the fused executed action. Current and next
`g` values come from the final 29 dimensions of the critic observations stored in replay; the
pre-existing critic-only linear velocity remains separate.

The environment owner samples one 29D `g` per reset and applies the same row to Kp, Kd, privileged
info, and critic observation. The formal Phase-1 profile uses only knee candidates initially and
disables unrelated Kp/Kd randomization so the causal intervention remains inspectable.

Training may start only after OFF regression, randomization support, actor fusion, frozen-gradient,
runtime connectivity, and checkpoint identity tests pass. A bounded one-environment update is an
engineering sentinel, not teacher-quality acceptance.

Formal quality acceptance requires paired evaluation on the same anomaly seeds and commands:
nominal frozen policy versus privileged residual teacher, using declared trajectory/line metrics in
addition to survival and velocity tracking. Those numeric thresholds remain a later human decision.

The explicit evaluation owner is `scripts/evaluate_context_teacher_phase1.py`. Not invoking this
entrypoint leaves all training and playback paths unchanged. Invocation fails closed when the
teacher checkpoint identity is invalid, actor/critic/privilege dimensions drift, paired snapshot
identity is not exact, or nominal/left-knee/right-knee strata are absent.
