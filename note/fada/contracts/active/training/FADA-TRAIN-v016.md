---
contract_id: FADA-TRAIN-v016
status: active
effective_date: 2026-08-27
supersedes: FADA-TRAIN-v015
method_contract: FADA-METHOD-v016
scope: nominal single-Reward validation followed by gain-targeted privileged Oracle and Planner-IDM
---

# FADA Source Training Contract v016

## Gate N — nominal single Reward

Gate N uses standard SAC on one `G1WalkFlat/MuJoCo` task with no privileged observation and no
physical domain randomization. One locomotion Reward applies to every command. The resolved profile
must contain no enabled Reward mode dispatcher and no `stand_*` Reward terms or overrides.
`gait_phase_enabled=false`; both retained phase slots are zero at reset and after every step; all
feet-phase scales are zero and gait constraint is disabled.

Gate N must reject the v015 failure ordering. Policy-quality admission requires survival and command
tracking evidence; a less-negative return accompanied by shorter episodes or 100% termination is a
failure, not progress.

## Unit A — final privileged Oracle

Only after Gate N policy-quality admission may Unit A inherit the exact single-Reward task and add
the existing privileged SAC runtime plus the sealed left-knee Gain distribution. Preflight rejects
dual-Reward mode dispatch, any `stand_*` Reward scale or term, enabled gait clock, nonzero phase
Reward, gait constraint, unrelated randomization, observation-noise drift, or lineage drift.

The final Actor remains 98-D and state66 retains two zero placeholders. The final run alone produces
checkpoints `240…4800` plus `5000` under one lineage. Gate N checkpoints remain forbidden from the
final lineage and Planner–IDM label ownership.

## Unit B — Planner–IDM

Unit B remains blocked until the v016 privileged Oracle passes separately authorized formal-runtime
and policy-quality gates. Existing Planner–IDM semantics and checkpoint schema remain unchanged.

## Authority

The v016 single-Reward configuration and fail-closed preflight are locally implemented with current
module evidence. This Contract and that evidence do not authorize or prove official runtime,
simulation, training, server operation, Git publication, deployment, or policy quality.
