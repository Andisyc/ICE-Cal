---
contract_id: FADA-TRAIN-v011
status: active
effective_date: 2026-08-26
supersedes: FADA-TRAIN-v010
method_contract: FADA-METHOD-v011
scope: fresh persistent-async unified-Oracle alternating training
---

# FADA Alternating Source Training Contract v011

The official route is one fresh `persistent_async` campaign. `training.fada.phase` and
`pretrained_idm_path` are retired and rejected. Resume, warm start, and an existing output path are
also rejected before runtime creation.

The final Oracle loader accepts the approved unified distillation checkpoint. Intermediate source
checkpoints retain the SAC loader and are consumed only by the IDM source route. Every admitted
artifact records `training_schedule=alternating_idm_then_planner`; every outer iteration performs
the configured IDM updates before the configured Planner updates. Planner backpropagation may flow
through IDM operations but cannot mutate IDM parameters.

Schema-5 checkpoints contain the Planner-IDM weights, architecture, schedule identity, completed
iteration and sample counters, quality metrics, and both optimizer states. Historical schemas are
playback-only. Long training still requires a fresh formal-runtime audit, exact server identities,
and separate human launch authorization.
