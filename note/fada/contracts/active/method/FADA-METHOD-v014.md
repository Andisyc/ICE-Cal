---
contract_id: FADA-METHOD-v014
status: active
effective_date: 2026-08-27
supersedes: FADA-METHOD-v013
scope: gain-targeted privileged-Oracle Planner-IDM source training with command-conditioned no-gait dual Reward
---

# FADA Planner–IDM Method Contract v014

## Source Oracle

ICE-Cal owns one privileged SAC Oracle trained directly on one `G1WalkFlat/MuJoCo` task. The v013
command-conditioned no-gait Reward remains unchanged: zero-command rows receive standing
support/stability terms, nonzero-command rows receive walking tracking terms, the command is the
sole mode authority, and gait/feet-phase Reward and constraints remain disabled.

The source distribution is nominal plus one task-relevant execution-gain perturbation. On reset,
the existing actuator-strength owner samples only the left-knee actuator at action index `3`.
Thirty percent of rows are exactly nominal; the remaining rows sample an effectiveness multiplier
`g` uniformly from `[0.8, 1.0]`. Ground friction, gravity, COM, base/body mass, armature, independent
Kp/Kd, joint-position bias, torque RFI, control delay, and pushes are disabled. Existing low-amplitude
joint observation noise is retained.

The deployable Actor remains 98-D and receives no explicit gain value. The typed privileged Critic
tail observes the applied effectiveness through the existing Kp/Kd scale fields; the optional
duplicate 29-D actuator-strength tail remains disabled. This profile covers left-knee attenuation,
not gain amplification or general full-domain robustness.

## Planner–IDM and lineage

The v013 Planner–IDM contract remains unchanged: raw 98 splits into state66, previous-action29, and
command3; Planner history is `H×95` with command separate and predicts an action-free `K×66`
future; IDM maps `H×66 + H×29 + K×66` to `K×29`. Oracle-shadow, first-action supervision,
IDM-before-Planner ordering, gradients through frozen IDM, and receding-horizon first-action
execution remain mandatory.

One admitted Oracle run supplies exactly twenty intermediate checkpoints at `240…4800` for IDM
coverage and one final checkpoint at `5000` for all Oracle labels. All checkpoints share one sealed
lineage and the same resolved gain-targeted config identity.

## Evidence boundary

Config composition and module tests can prove the distribution and information boundary. They do
not prove Oracle convergence, downstream IDM/Planner learning, formal runtime reachability,
transfer, or calibration efficacy.

