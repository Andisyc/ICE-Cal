---
contract_id: FADA-METHOD-v015
status: historical
effective_date: 2026-08-27
supersedes: FADA-METHOD-v013
superseded_by: FADA-METHOD-v016
scope: historical phase-neutral privileged-Oracle Planner-IDM source training
---

# FADA Planner–IDM Method Contract v015 — Historical

## Source task and phase authority

ICE-Cal owns one `G1WalkFlat/MuJoCo` source task. Command is the sole stand/walk mode authority:
zero-command rows receive standing support/stability Reward and nonzero-command rows receive walking
tracking Reward. Gait phase has no behavioral authority: it is not sampled or advanced, all
feet-phase Reward terms are zero, and gait constraints are disabled.

The deployable Actor remains 98-D for Planner–IDM compatibility. The two legacy gait-phase slots are
retained only as constant zero placeholders. They carry no clock, target, mode, or privileged
information. Consequently the existing split remains state66 + previous-action29 + command3.

## Source distribution and Oracle

Targeted actuator faults are not part of Oracle training. They belong only to downstream
failed-rollout collection after the source Oracle is frozen. The deployable Actor receives no
explicit fault value; the typed privileged Critic contract remains separate from fault injection.

Before privileged training is admitted, one nominal standard-SAC profile must prove the same
phase-neutral dual-Reward task without privileged observation or physical DR. That checkpoint is an
engineering validation artifact only and cannot label Planner–IDM data or join the final Oracle
lineage.

## Planner–IDM and lineage

Planner–IDM tensor, causal pairing, optimizer ordering, freeze, Oracle-shadow, first-action
supervision, and receding-horizon semantics remain unchanged. The final admitted privileged Oracle
run alone supplies checkpoints `240…4800` plus final `5000` under one sealed lineage.

## Evidence boundary

Offline tests can prove constant-zero phase slots, dimensions, Reward dispatch, profile isolation,
and preflight rejection. They cannot prove nominal or privileged policy quality, formal runtime
reachability, Planner–IDM learning, transfer, or calibration efficacy.

## Historical disposition

The v015 live run showed 100% termination while episode length collapsed as reported Reward became
less negative. v016 therefore retires the dual-Reward objective. This Contract is preserved only as
history and grants no implementation or training authority.
