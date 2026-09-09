---
contract_id: FADA-TRAIN-v016
status: superseded
effective_date: 2026-08-27
supersedes: FADA-TRAIN-v015
superseded_by: FADA-TRAIN-v017
method_contract: FADA-METHOD-v016
scope: historical nominal gate followed by privileged Oracle and Planner-IDM
---

# FADA Source Training Contract v016 — Historical

## Historical route

v016 first defined a nominal single-Reward Gate N followed by a privileged Oracle and its 20+1
lineage. Targeted actuator faults are outside this training boundary. The trained checkpoint
terminated every episode and did not satisfy policy-quality requirements.

## Supersession reason

Targeted actuator faults belong to downstream failed-rollout collection, not perfect Oracle
construction. v017 records that boundary and invalidates conflicting execution admissions.
