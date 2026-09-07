# Retired original-height control experiment

Status: historical. This profile is not an active training entrypoint.

Identity: `original_height_mixed_v1`; mode: `original_command_height_v1`.

The original repository path is `compute_feet_phase_height_targets` followed by
the legacy branch of `G1WalkRewardBindings._reward_feet_phase`, not v3. It uses
an entire-cycle cubic rise/fall without a sustained zero-height support interval.
Measured heights are relative to the lower foot site. The reward is
`exp(-(left_error_squared + right_error_squared) / feet_phase_tracking_sigma)`.
The original phase-locomotion profile sets this denominator to `0.04` m^2;
it is not the standard deviation used by v3's `exp(-sum_error / (2 * scale^2))`.

The adaptation only conditions the original target amplitude:
`demand = sqrt(vx^2 + vy^2 + (0.3 * yaw_rate)^2)`;
`height = 0.06 * demand / (demand + 0.3)`.
The original curve and reward calculation are directly reused. The existing
velocity gate is configured with threshold zero, so it does not suppress the
height reward at zero command or during reverse motion. The gait weight remains
1.0, matching the current experiment; other base reward weights are unchanged.

Actor and critic observation assembly appends two raw radian phase values,
not sine/cosine pairs. Phase remains enabled with offset initialization and the
existing 1.5 Hz cycle. Zero command zeros target amplitude only. No input
dimension or clock convention is changed.

This is an original-reward adaptation control, not a curve-only ablation versus
v3: relative-height measurement and the original error denominator also differ.
Equal vertical translations of both feet remain indistinguishable under the
original reward. Zero targets alone therefore do not guarantee physical contact.

The historical training selector and dedicated launcher were retired on
2026-09-07. Use one of the canonical identities instead:

```bash
cd /ssd1/cyx/ICE-Cal
uv run --frozen --no-sync python scripts/train_offpolicy.py \
  algo=sac \
  task=sac/g1_walk_flat/mujoco_clean_baseline
```

Use `mujoco_fada_source` only when the phase-neutral privileged source teacher and
its grouped domain randomization are intended. This document preserves the old
reward semantics as evidence; it no longer defines an executable workflow.
