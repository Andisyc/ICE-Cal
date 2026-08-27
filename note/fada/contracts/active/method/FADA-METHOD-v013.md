---
contract_id: FADA-METHOD-v013
status: active
effective_date: 2026-08-27
supersedes: FADA-METHOD-v012
scope: single-task privileged-Oracle Planner-IDM source training with command-conditioned no-gait dual Reward
---

# FADA Planner–IDM Method Contract v013

## Source Oracle

ICE-Cal owns one generic privileged SAC Oracle trained directly on
`G1WalkFlat/MuJoCo`. A zero command is an ordinary atom in the same locomotion command
distribution; there is no standing task, transition task, second Oracle, or distillation route.
The Actor keeps the deployable 98-D observation and receives no mode bit. The command is the sole
stand/walk mode authority.

The task uses two command-conditioned Reward branches. Zero-command rows receive standing
support/stability geometry, including base/feet support alignment, contact, height, tilt, base velocity,
and upper-body posture. Nonzero-command rows receive velocity tracking plus ordinary balance terms.
Standing support terms cannot affect walking rows, and tracking terms cannot affect standing rows.
`stand_action_l2` and `stand_still` are forbidden because a loaded stand requires nonzero support
action and should not be rigidly anchored to the default lower-body pose.

Gait phase may remain observable but cannot own Reward credit. `feet_phase`,
`feet_phase_contrast`, `feet_phase_contact`, every equivalent phase/footfall scale, and the gait
constraint are zero or disabled. Oracle preflight rejects any violation before environment creation.

## Planner–IDM and lineage

The v012 Planner–IDM tensor and causal contracts remain unchanged: raw 98 splits into state66,
previous-action29, and command3; Planner history is `H×95` with command separate and predicts an
action-free `K×66` future; IDM maps `H×66 + H×29 + K×66` to `K×29`. Oracle-shadow,
first-action supervision, IDM-before-Planner ordering, gradients through frozen IDM, and
receding-horizon first-action execution remain mandatory.

One Oracle run supplies exactly twenty intermediate checkpoints at `240…4800` for IDM coverage and
one final checkpoint at `5000` for all Oracle labels. All share one sealed lineage and resolved
Reward/config identity.

## Evidence boundary

Config composition and module tests do not prove Oracle convergence, standing or walking quality,
formal-route reachability, Planner-IDM readiness, transfer, or calibration efficacy.
