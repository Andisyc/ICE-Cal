---
contract_id: FADA-METHOD-v017
status: active
effective_date: 2026-08-27
supersedes: FADA-METHOD-v016
scope: nominal privileged-Oracle Planner-IDM source training with downstream-only failure injection
---

# FADA Planner–IDM Method Contract v017

## Source task and Reward authority

ICE-Cal owns one `G1WalkFlat/MuJoCo` source task and one scalar locomotion Reward for every command.
Command changes only the velocity-tracking target; it never selects a standing, walking, or recovery
Reward family. Gait phase has no behavioral authority: the two compatibility positions remain exact
zero so the Actor stays 98-D and the Planner–IDM split remains state66 + previous-action29 + command3.

## Perfect Oracle ownership

The Oracle is a source-domain teacher trained only under nominal dynamics. Its Actor consumes the
deployable 98-D observation. Its Critic may consume the existing typed privileged observation, but
that observation contains no randomized execution fault whose compensation is later claimed by
ICE-Cal. Actuator strength, Gain attenuation, delay, bias, friction, mass, COM, Kp/Kd variation,
external pushes, observation-noise drift, and every other physical/domain perturbation are disabled
for Oracle training.

The 20 intermediate checkpoints (`240…4800`) and final checkpoint (`5000`) come from this one nominal
privileged-Oracle run and share one `oracle_lineage_id`. Intermediate checkpoints provide IDM
suboptimal-policy coverage; the final checkpoint owns source action labels and Oracle-shadow.

## Failure ownership

Gain, delay, bias, and other execution failures belong only to downstream failed-rollout collection
after the Oracle and distilled Planner–Tracker source backbone have been trained and frozen. A
left-knee Gain experiment may be the first calibration smoke case, but it cannot alter, label, or
authorize the Oracle training distribution.

## Planner–IDM semantics

Planner–IDM tensor layout, causal future–action pairing, IDM-before-Planner ordering, frozen-IDM
Planner gradients, Oracle-shadow, first-action supervision, and receding horizon remain unchanged.

## Evidence boundary

All v016 module, formal-runtime, training, checkpoint, and policy observations are historical because
they bind the superseded gain-targeted Oracle identity. v017 currently authorizes documentation and
engineering planning only. It does not claim that the current configuration implements the nominal
Oracle or that any v017 checkpoint exists.
