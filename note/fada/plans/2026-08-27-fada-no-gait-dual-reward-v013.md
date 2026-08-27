# FADA v013 No-Gait Dual-Reward Oracle Plan

> Status: human-approved for local construction. Training, simulation, server, and Git actions remain out of scope.

## Outcome

Repair the single-task `G1WalkFlat/MuJoCo` privileged Oracle so that the command is the sole mode
authority: zero command receives a standing support/stability objective and nonzero command receives
the locomotion tracking objective. The Actor remains deployable at 98 dimensions and receives no
explicit mode bit.

## Invariants

- one task, one Oracle, one command distribution; `rel_standing_envs=0.3` and
  `rel_transition_envs=0`;
- `mode_observation=false`;
- `feet_phase`, `feet_phase_contrast`, and `feet_phase_contact` are zero or absent;
- `gait_constraint.enabled=false` and `penalty_scale=0`;
- standing terms include support geometry and stability but exclude `stand_action_l2` and
  `stand_still`;
- walking terms include velocity tracking and ordinary balance terms but exclude standing support
  geometry;
- no new Reward implementation or second environment path.

## Engineering steps

1. Add failing config/preflight tests for the exact dual-Reward contract and negative overrides.
2. Add one Hydra profile owned by the Oracle task and compose it from the Oracle config.
3. Replace the old `reward.mode` rejection with a structural validator at Oracle preflight.
4. Activate v013 Contracts and update the Design Inspector and module card.
5. Run focused Reward, Oracle, formal-route, and configuration regressions; review the diff for
   duplicated ownership and hidden compatibility paths.

## Evidence boundary

Local tests can prove composition, dispatch structure, negative admission, and repository
connectivity. They cannot prove that SAC converges, that the resulting checkpoint stands or walks,
or that Planner-IDM training is ready. Those remain formal-runtime and policy-quality gates.
