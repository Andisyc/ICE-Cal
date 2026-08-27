---
contract_id: FADA-METHOD-v016
status: superseded
effective_date: 2026-08-27
supersedes: FADA-METHOD-v015
superseded_by: FADA-METHOD-v017
scope: historical single-Reward phase-neutral gain-targeted privileged-Oracle Planner-IDM source training
---

# FADA Planner–IDM Method Contract v016 — Historical

## Source task and Reward authority

ICE-Cal owns one `G1WalkFlat/MuJoCo` source task and one scalar locomotion Reward for every command.
There is no standing Reward family, walking Reward family, recovery Reward family, or
command-conditioned Reward dispatcher. Command changes only the velocity target already consumed by
the ordinary tracking terms: a zero command asks the same locomotion objective to track zero
velocity; a nonzero command asks it to track motion.

Gait phase has no behavioral authority. It is not sampled or advanced; `feet_phase`,
`feet_phase_contrast`, and `feet_phase_contact` are zero; gait constraints are disabled. The two
legacy phase positions remain constant-zero compatibility placeholders so the deployable Actor stays
98-D and the Planner–IDM split remains state66 + previous-action29 + command3.

## Source distribution and Oracle

The privileged Oracle adds only the existing typed privileged Critic tail and left-knee actuator
attenuation at action index `3`: non-nominal `g` is sampled uniformly from `[0.8,1.0]` and nominal
probability is `0.3`. Actor receives no Gain value. Friction, mass, COM, independent Kp/Kd,
observation noise, external push, delay, bias, and other unrelated randomization remain disabled.

A nominal standard-SAC gate must first validate the exact same single-Reward task without privileged
observation or physical randomization. Its checkpoint is validation-only and never joins the final
Oracle lineage or labels Planner–IDM data.

## Planner–IDM and lineage

Planner–IDM tensor layout, causal future–action pairing, IDM-before-Planner ordering, frozen-IDM
Planner gradients, Oracle-shadow, first-action supervision, receding horizon, and exact 20+1
privileged-Oracle lineage remain unchanged.

## Supersession reason

v016 incorrectly allowed the downstream left-knee Gain calibration variable to own the upstream
Oracle training distribution. This context pollution invalidated the meaning of a perfect nominal
Oracle. v017 separates nominal teacher construction from later failed-rollout perturbation.
