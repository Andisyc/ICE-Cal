---
contract_id: FADA-TRAIN-v016
status: superseded
effective_date: 2026-08-27
supersedes: FADA-TRAIN-v015
superseded_by: FADA-TRAIN-v017
method_contract: FADA-METHOD-v016
scope: historical nominal gate followed by gain-targeted privileged Oracle and Planner-IDM
---

# FADA Source Training Contract v016 — Historical

## Historical route

v016 first defined a nominal single-Reward Gate N, then allowed the final privileged Oracle to add a
left-knee Gain distribution before producing its 20+1 lineage. This route was implemented and passed
local module/formal connectivity checks, but its trained checkpoint terminated every episode and did
not satisfy policy-quality requirements.

## Supersession reason

The left-knee Gain variable belongs to downstream failed-rollout collection, not perfect Oracle
construction. v017 removes that semantic leakage and invalidates all v016 execution admissions.
