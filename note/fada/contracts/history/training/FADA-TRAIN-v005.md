---
contract_id: FADA-TRAIN-v005
status: active
effective_date: 2026-08-05
updated_date: 2026-08-05
supersedes: FADA-TRAIN-v004
method_contract: FADA-METHOD-v005
scope: persistent-async UniLab Planner-IDM DAgger with cold-start and scenario-balanced Planner replay
---

# FADA UniLab Source Training Contract

`training.fada.v005_replay.enabled` is the single owner flag. OFF preserves the v004 artifact,
collector, replay, update, and metric behavior. ON activates together:

- exact cold-start collection for half of `static_stand`;
- row-level scenario, Planner eligibility, and cold-start identity;
- IDM sampling from the complete replay;
- Planner-only fixed scenario and cold-start stratification;
- scenario-resolved production checkpoint metrics.

The formal route remains `persistent_async`. The child owns resident environments, Oracles, rollout
policy, and source artifact creation. The parent owns replay, ordered IDM/Planner optimization,
quality evaluation, and checkpoint persistence.

Intermediate Oracle rows are always walking rows and `planner_eligible=false`. Main
optimal/current-policy rows are `planner_eligible=true`. No command inference may reconstruct missing
identity after collection.

`training.fada.initial_weights_path` may initialize Planner and IDM parameters from a compatible
checkpoint. It never restores optimizer state, replay, iteration cursor, samples-seen counters, or
quality metrics. `resume_path` remains rejected on the paper-exact route.

Formal training is outside the local engineering closure. Training may start only after OFF/ON,
artifact round-trip, exact sampler, gradient ownership, production serializer, and bounded real
MuJoCo source sentinels pass.

## Non-scope

Planner/IDM architecture changes, Oracle training, standing-to-walking transitions, target-domain
adaptation, deployment controllers, and hard zero-command action gates remain outside this contract.
