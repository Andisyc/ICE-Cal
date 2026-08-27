---
contract_id: FADA-TRAIN-v014
status: active
effective_date: 2026-08-27
supersedes: FADA-TRAIN-v013
method_contract: FADA-METHOD-v014
scope: serial gain-targeted privileged-Oracle lineage then fresh single-task Planner-IDM training
---

# FADA Source Training Contract v014

## Unit A — privileged Oracle

The official route remains `privileged_locomotion_sac` on one `G1WalkFlat/MuJoCo` environment.
The 5,000-iteration schedule, 20+1 checkpoint lineage, 98-D Actor input, typed privileged Critic
tail, command distribution, and v013 no-gait dual Reward remain unchanged.

The resolved Hydra profile must contain exactly one physical randomization mechanism:

- `actuator_strength.enabled=true`;
- `sampling_mode=single_candidate` and `candidate_actuator_indices=[3]`;
- `multiplier_range=[0.8, 1.0]` and `nominal_probability=0.3`;
- `include_in_critic_obs=false`, because the typed Kp/Kd scale fields already expose the applied
  effectiveness to the Critic;
- every unrelated domain-randomization mechanism is disabled, with torque RFI equal to zero.

Oracle preflight validates this distribution before environment creation. The existing checkpoint
preflight also seals the retained observation-noise profile (`joint_angle=0.01`, `joint_vel=0.1`,
other configured sensor scales zero). The existing checkpoint config hash seals the complete
resolved profile; a checkpoint from v013 or any override profile is not interchangeable with v014.

## Unit B — Planner–IDM

Unit B retains action-free future, final-Oracle label ownership, intermediate IDM-only coverage,
IDM-before-Planner serial updates, and frozen-IDM Planner gradients. Reduced source DR changes the
evidence scope, not the tensor or optimizer contract: an admitted v014 lineage supports only the
nominal-plus-left-knee-attenuation source distribution.

Unit B remains blocked until a separately authorized formal runtime audit and policy-quality audit
admit the v014 final Oracle and its twenty same-lineage intermediate checkpoints.

## Authority

Training, simulation, server operation, Git publication, deployment, and policy-quality evaluation
remain separate explicit actions. Local module evidence cannot authorize a long run.
