---
contract_id: FADA-TRAIN-v024
status: active / formal-runtime-blocked / not-run
effective_date: 2026-09-07
method_contract: FADA-METHOD-v024
supersedes: FADA-TRAIN-v023
scope: construction and evidence gates for the command-gated phase-contact Oracle lineage
---

# FADA Command-Gated Phase-Contact Oracle Training Contract

## Current admission

The design, `mujoco_fada_phase_contact` production selector, owner implementation, affected-module
plan, and offline Module Test evidence are current. A bounded official reset/step smoke exists, but
no v024 checkpoint or full formal runtime receipt exists. Training remains forbidden until the
full formal runtime audit is current for v024.

`mujoco_fada_phase` remains a v023 compatibility profile. It is not the v024 training selector and
must not be renamed or treated as equivalent.

## Source Oracle training

The canonical `task=sac/g1_walk_flat/mujoco_fada_phase_contact` selector expresses one explicit
`G1WalkFlat` MuJoCo configuration with:

- exact command dead-zone normalization before standing sampling;
- 30 percent null-command environments;
- `heading_command=false` and one final canonical command consumed by observation, phase, and Reward;
- command-owned per-environment phase state machine with `[pi, pi]` null phase, deterministic
  pi-separated non-null startup, and explicit null/non-null transitions;
- translation-and-yaw `s_cmd` with zero null frequency and bounded `[f_min, f_max]` non-null frequency;
- radian-to-cycle phase normalization and `duty_factor=0.55` contact windows;
- finite negative phase/contact mismatch cost with no height target or constant match bonus;
- null-command foot-force balance and torque relaxation only;
- v022 typed privilege and broad physical DR, with targeted actuator faults excluded from Oracle
  training and generic all-joint Kp/Kd randomization retained;
- behavior profile `command_gated_phase_contact_v1`;
- the sealed 20 intermediate checkpoints at iterations `240..4800` plus `model_5000.pt`.

The engineering plan must choose and test the two `s_cmd` saturation spans, `f_min`, `f_max`,
contact-force threshold, contact term scale, and the two standing-quality scales. It must test the
joint `duty_factor`/`f_min` double-support and single-support timing region. Parameter selection must
not change the semantic invariants or behavior ordering in the Method Contract.

## Planner-IDM distillation

Planner-IDM distillation may start only from one admitted v024 20+1 Oracle lineage. Source windows
include the same null-command and commanded-motion regimes that the Source Oracle owns. Phase and command remain visible; all
windows retain `source_behavior_profile=command_gated_phase_contact_v1`.

The Planner-IDM architecture, optimizer ownership, causal future/action pairing, and checkpoint
schema remain unchanged unless separately reviewed.

## Target collection and adaptation

Target collection, LoRA adaptation, and evaluation remain owned by FADA-ADAPT v003. A v024 route
must reject source checkpoints whose behavior profile or command/phase contract differs, even when
tensor dimensions match. Existing v023 artifacts remain historical compatibility inputs only and
cannot establish v024 quality.

## Required proof sequence

1. Module tests establish exact dead-zone ordering, dual-channel `s_cmd`, frequency boundaries,
   reset and four regime transitions, pi separation, radian-to-cycle conversion, contact truth
   tables, Reward boundedness, observation visibility, backend-interface use, and configuration
   identity. Controlled counterexamples must detect synchronized-phase, translation-only frequency,
   radian/duty-unit, positive-bonus, and duplicated-dead-zone defects.
2. Formal runtime audit proves the official selector reaches reset, rollout, update, checkpoint,
   and reload boundaries without training for quality.
3. A separately authorized bounded experiment checks all six relations in the v024 Reward Ordering
   Card and reports the external gait telemetry named there.
4. Only policy-quality audit may compare v024 against v023 and the clean baseline.

## Stop conditions

- Stop if any null-command environment advances phase or receives a gait bonus for standing still.
- Stop if dead-zone logic is duplicated downstream or differs between observation and Reward.
- Stop if a non-null transition loses pi separation, cannot activate on pure yaw/lateral/backward
  command, or uses radians directly as a duty fraction.
- Stop if contact calibration uses backend-private access or parses assets in a hot path.
- Stop if a relative-height target, positive null-command constant reward, or unbounded frequency
  mapping reappears.
- Stop on behavior-profile, observation, checkpoint-lineage, or target-route mismatch.
- Do not infer gait or policy quality from config composition, unit tests, or a formal route audit.
