# FADA Paper-Aligned Source Training v012 Design

Status: human-confirmed; Unit A implemented and locally verified on 2026-08-26. Training remains unauthorized.

## Scientific object

The source-domain object is one task-specific privileged locomotion Oracle and a deployable
Planner–IDM distilled from its behavior. ICE-Cal owns the method, configuration, training entry,
and all checkpoints. The repository reuses its UniLab-derived MuJoCo, asynchronous collection,
and SAC infrastructure, but no Oracle artifact is imported from the sibling UniLab checkout.

The final Oracle is trained directly from the `G1WalkFlat` task reward under domain randomization.
It observes the deployable locomotion input plus a typed privileged bundle. Twenty intermediate
checkpoints from that same run broaden inverse-dynamics coverage; they are never Planner-label
authorities.

## Oracle observation, optimization, and lineage

The Oracle is a generic `privileged_locomotion_sac` policy, not the actuator-fault-specific
`privileged_full_action_sac`. Its actor, not only its critic, receives the privileged bundle:

- base linear velocity;
- ordered per-body contact forces and binary contact flags;
- local terrain-height samples and root clearance;
- ordered actuator state: sampled Kp/Kd scales and normalized torques;
- the actual sampled domain-randomization parameters for the current environment instance.

The bundle has a versioned field layout, scales, units, body/joint order, and resolved width. The
same canonical layout and hashes of task, reward, domain-randomization, asset, backend, action
scale, seed, and run lineage are persisted in every checkpoint and checked before policy loading.
The critic receives at least the same information. Backend asset/model metadata is materialized on
the cold path; hot steps consume cached numeric arrays only.

The first run uses SAC for 5,000 iterations. It saves checkpoints at iterations
`240, 480, ..., 4800` and a final checkpoint at `5000`. The first twenty form the intermediate set;
the final checkpoint is the sole Oracle used for labels and shadows. All twenty-one artifacts must
share one `oracle_lineage_id` and exact contract hashes.

## Task, reward, and domain distribution

The first transaction contains only `G1WalkFlat/MuJoCo` with its velocity-command distribution.
There is no independent standing environment, standing policy, walk-to-stand scenario, or
scenario quota. A zero-velocity command, if sampled by the one locomotion task, remains an
ordinary command rather than a second task owner.

The task reward must not contain a gait-phase or prescribed-footfall reward. In particular,
`reward.scales.feet_phase` must be zero or absent, and the v012 preflight rejects any non-zero
phase-conditioned gait-reward alias. Gait phase remains an observation/reference input; removing
reward authority does not remove that state. This prevents a policy from being rewarded for
stepping when the velocity command is zero. Tracking, stability, posture, action regularization,
and termination remain task-owned reward terms.

Source randomization follows the locomotion family in FADA Table 6: ground friction, base CoM,
added base mass, link-mass scaling, DoF position bias, Kp/Kd scale, torque RFI, one-step control
delay, and pushes. Missing capabilities are added at the environment/backend owner and exposed
through the cached sampled-parameter bundle; scripts must not emulate them.

## Planner–IDM input and causal target

The current 98-D locomotion observation is split rather than projected to an opaque vector:

```text
x_t: 66 = gyro 3 + projected gravity 3 + joint position 29
          + joint velocity 29 + gait phase 2
a_{t-1}: 29
c_t: 3 velocity command
```

The Planner history token is 95-D, formed from `x_t` and `a_{t-1}`, with the current 3-D command
encoded separately. The Planner predicts an action-free future `Y_t^K` of shape `K x 66`. Its
residual head is anchored to the latest 66-D `x_t`, not to the 95-D history token.

The IDM consumes `X_t^H` (`H x 66`), executed action history `A_t^H` (`H x 29`), and the
action-free future (`K x 66`) and emits `K x 29` actions. Previous action is therefore visible to
the Planner without putting the supervised action into the future token. This resolves the paper's
field-level notation using the causal constraint: `o_{t+1}` cannot expose target `a_t` to the IDM.

## Source data and optimization

All rows belong to the one locomotion task. Final-Oracle rollouts and Planner–IDM rollouts carry a
same-state final-Oracle shadow. Intermediate-checkpoint rollouts provide realized, causally matched
future/action pairs only for IDM coverage. At every visited student or intermediate state, the final
Oracle supplies the Planner's relabeled first action.

Optimal and suboptimal source data retain a 1:2 budget. Each outer iteration completes the IDM
updates first. The Planner pass then differentiates through IDM operations while IDM parameters
and optimizer state are frozen. The default loss supervises only the first action, and deployment
executes only that action before replanning.

## Persistence, compatibility, and proof boundary

v011 method/training Contracts and all receipts bound to them become historical. Existing schema-5
Planner–IDM checkpoints remain historical inference artifacts only; they cannot initialize or resume
v012 because their Oracle authority, observation layout, task distribution, and source identity are
different. v012 introduces new Oracle-lineage and Planner–IDM schema identities and starts both
training campaigns fresh.

Engineering is divided into two serial units. Unit A creates and validates the privileged Oracle
lineage. Unit B changes the Planner–IDM input and source collector and cannot launch until Unit A
has one admitted final Oracle and exactly twenty admitted intermediate checkpoints. Offline tests do
not authorize simulator execution, training, policy-quality claims, Git actions, or deployment.

## Falsifiable predictions

- With a zero velocity command, removing gait reward eliminates direct positive credit for periodic
  stepping; persistent stepping is then evidence of another reward or policy defect.
- Intermediate checkpoints reduce IDM error on off-nominal visited states without changing Planner
  label authority.
- Any future tensor containing its paired first target action is rejected before replay admission.
- Mixing checkpoints across Oracle lineages is rejected before environment creation.

## Paper provenance

The source semantics follow FADA Sections 3–4, Table 5, Table 6, and Appendix B.1–B.2,
arXiv:2606.28476. The action-free future split is an explicit causal implementation decision because
the paper lists previous action among deployable inputs but does not disambiguate whether it is a
predicted future field.
