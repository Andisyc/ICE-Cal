---
contract_id: FADA-TRAIN-v022
status: active
effective_date: 2026-08-29
supersedes: FADA-TRAIN-v017
method_contract: FADA-METHOD-v022
scope: 5000-iteration live-privileged grouped-DR teacher, sealed 20+1 lineage, then Planner-IDM
---

# FADA Source Training Contract v022

## Unit A — privileged SAC teacher

Compose `mujoco_fada_privileged_oracle_live_input_dr_curriculum` for one 5000-iteration
`G1WalkFlat/MuJoCo` run. The Actor and Critic consume normalized live
`g1_fada_privileged_v1` information. Penalty curriculum remains enabled, while grouped physical
randomization expands by the iteration schedule defined in `FADA-METHOD-v022`.

The fixed-input, live-input nominal, and grouped-DR profiles are diagnostic or policy-quality
profiles. They do not become the source lineage merely because a terminal run finishes.

## Unit A persistence gate

The authoritative source lineage must save `model_240.pt, model_480.pt, …, model_4800.pt` and
`model_5000.pt` under one `oracle_lineage_id`, with one sealed checkpoint contract and consistent
configuration/layout hashes.

Current code couples `privileged_dr_curriculum_validation=true` to `checkpoint_mode=validation` and
`save_interval=1000`. Consequently the successful v022 validation run cannot provide the required
intermediate checkpoints. The next engineering unit is to add a sealed grouped-DR lineage profile
without changing the successful perturbation schedule or privileged-input normalization path.

## Unit B — Planner–IDM

IDM may start only after all 20 intermediate checkpoints and the final checkpoint exist and pass
lineage admission. The current `mujoco_fada_privileged_idm` route is implemented, but its attempted
launch correctly failed when `model_240.pt…model_4800.pt` were absent. That failure is a persistence
gate, not evidence against the learned policy.

### Unit B replay distribution amendment — 2026-09-01

The active Planner–IDM route preserves the main scenario allocation
`walk/static_stand/walk_to_stand = 0.50/0.25/0.25`. Replay draws from walking main-source rows at
`0.20/0.80` cold/steady; static-standing replay keeps `0.50/0.50`. Collector admission remains
balanced at `0.50/0.50` cold/steady for both walk and static standing, so replay distribution does
not alter source collection identity.

Walking steady-state speed is the planar command norm `||(v_x, v_y)||_2`. Its bins are
`slow < 0.25`, `medium in [0.25, 0.60)`, and `high >= 0.60 m/s`, sampled with replacement at
`0.10/0.30/0.60`. Planner applies this hierarchy to Planner-eligible rows. IDM first allocates main
and intermediate-Oracle rows by stable largest remainder for the retained `1:2` source ratio
(`171/341` for batch size `512`), then applies the same speed strata independently within the main
walking and intermediate walking pools.

Before either optimizer mutates parameters, every positive Planner and IDM stratum must exist.
Planner high-speed, IDM-main high-speed, and IDM-intermediate high-speed pools each require eight
expected replay passes. Their required update counts are computed independently with integer ceiling;
IDM uses the maximum of its two pool-specific requirements. Actual per-iteration updates are the
maximum of the configured update count and the corresponding coverage requirement.

Speed identity is derived from the existing `command` tensor. This amendment does not change
`FADASourceBatch`, source-artifact or checkpoint schemas, model inputs, losses, Collector behavior,
or the non-v005 uniform replay path.

## Current status

- Live privileged input and normalization: implemented and exercised.
- Iteration-based grouped DR curriculum: implemented and exercised.
- Qualitative v022 policy quality: Reward and episode length observed high; exact metrics not sealed.
- Sealed grouped-DR 20+1 lineage: not implemented and not trained.
- Planner–IDM transition: blocked only on the missing admitted lineage and subsequent runtime audit.
- Speed-stratified Unit B replay: implemented with offline owner, integration, and lifecycle tests;
  no new training or policy-quality claim.

No document in this Contract authorizes a server run, checkpoint reuse, or policy-quality claim
beyond the evidence explicitly recorded for the matching configuration and lineage.
