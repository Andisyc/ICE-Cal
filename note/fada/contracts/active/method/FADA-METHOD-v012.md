---
contract_id: FADA-METHOD-v012
status: active
effective_date: 2026-08-26
supersedes: FADA-METHOD-v011
scope: paper-aligned single-task privileged-Oracle Planner-IDM source training
---

# FADA Planner–IDM Method Contract v012

## Authority and scientific object

ICE-Cal owns one generic privileged locomotion Oracle and the Planner–IDM distilled from it. The
Oracle is trained directly in the ICE-Cal checkout with the UniLab-derived high-throughput MuJoCo
and SAC infrastructure. A distilled policy, a sibling-repository artifact, or a fault-specific teacher
cannot become the final Oracle.

The first task is only `G1WalkFlat`. One privileged final Oracle supplies every Planner first-action
label and every Oracle-shadow pair. Exactly twenty intermediate checkpoints from the same Oracle
run supply suboptimal realized trajectories for IDM coverage only.

## Deployable observation and future

The deployable locomotion fields are split into `x_t` (66-D proprioceptive state and gait phase),
`a_{t-1}` (29-D previous action), and `c_t` (3-D command). The Planner receives an `H x 95`
history made from `(x_t, a_{t-1})` and the command separately. It predicts an action-free `K x 66`
future relative to the latest `x_t`.

The IDM consumes `H x 66` state history, `H x 29` executed-action history, and `K x 66` future,
then emits `K x 29` actions. Future observations must not contain previous-action or command
fields. This fail-closed rule prevents the first future token from exposing the supervised action.
The active dimensions are `H=30`, `K=6`, state `66`, Planner history `95`, action `29`, and command
`3`.

## Causal supervision and update ownership

Realized trajectory fields pair physically observed action-free futures with the actions actually
executed over the same window. Oracle-source IDM rows use the physically rolled final-Oracle
shadow pair. Intermediate rows remain trajectory-source IDM evidence, while their visited states
are separately relabeled by the final Oracle for Planner supervision.

Every outer iteration completes the configured IDM updates before the Planner updates. Planner
gradients traverse the IDM computation, but IDM parameters and optimizer state remain unchanged
during the Planner pass. Loss and deployment both use the first action of the six-action chunk in a
receding-horizon loop.

## Task and reward boundary

The first source campaign has no independent standing task, standing Oracle, transition task, or
walk/static/transition scenario quota. A zero command remains an ordinary sample of the locomotion
task.

Gait phase may be observed but cannot own reward credit. Any non-zero gait-phase or prescribed
footfall reward, including `reward.scales.feet_phase`, is forbidden and must reject before environment
creation. This prevents positive reward for stepping under a zero-velocity command.

## Explicit non-scope

This Contract does not claim Oracle convergence, standing quality, walking stability, source-to-
target transfer, formal-route reachability, real-robot safety, or ICE-Cal calibration efficacy.
