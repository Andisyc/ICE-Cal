---
contract_id: FADA-TRAIN-v015
status: historical
effective_date: 2026-08-27
supersedes: FADA-TRAIN-v014
superseded_by: FADA-TRAIN-v016
method_contract: FADA-METHOD-v015
scope: phase-neutral nominal validation followed by privileged gain-targeted Oracle and Planner-IDM
---

# FADA Source Training Contract v015 — Historical

## Gate N — nominal phase-neutral dual Reward

The first training gate uses standard SAC on one `G1WalkFlat/MuJoCo` task with no privileged
observation and no physical domain randomization. `gait_phase_enabled=false`; both retained phase
slots are zero at reset and after every step. `feet_phase`, `feet_phase_contrast`, and
`feet_phase_contact` are zero, gait constraint is disabled, and Command dispatches the standing or
walking Reward family. This gate may establish policy quality only for the nominal task.

## Unit A — final privileged Oracle

After separate policy-quality admission of Gate N, the final Oracle profile inherits the exact
phase-neutral dual-Reward task and adds only the existing privileged SAC runtime plus the sealed
v014 left-knee actuator-strength distribution. Oracle preflight rejects an enabled gait clock,
nonzero phase Reward, gait constraint, unrelated DR, observation-noise drift, or lineage drift.

The final Actor remains 98-D; state66 retains two constant-zero compatibility slots. The 5,000
iteration and 20+1 lineage contract remains unchanged. Gate N checkpoints are forbidden from the
final lineage and from Planner–IDM label ownership.

## Unit B — Planner–IDM

Unit B remains blocked until the final privileged v015 Oracle passes separately authorized formal
runtime and policy-quality gates. Existing state/action/command splits, action-free future,
IDM-before-Planner order, frozen-IDM Planner gradients, and checkpoint semantics remain unchanged.

## Authority

This Contract authorizes no simulation, training, server operation, Git publication, deployment, or
policy-quality claim. Those remain separate explicit actions.

## Historical disposition

The live campaign invalidated the dual-Reward objective through 100% termination and shrinking
episodes. All v015 module/formal receipts remain wiring evidence only and cannot authorize v016.
