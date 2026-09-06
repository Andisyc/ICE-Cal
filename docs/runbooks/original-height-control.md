# Original height reward control experiment

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

Training entry (after syncing to the server):

```bash
cd /ssd1/cyx/ICE-Cal
bash scripts/train_original_height_oracle.sh
```

The entry uses 512 environments, batch size 2048, 5000 iterations, a new timestamped
lineage/log directory and the existing 240-iteration save cadence. It does not
resume or overwrite the root checkpoint. Old tasks and the v3 launcher are preserved.
The new play profile permits incremental keyboard velocity commands and inherits
nominal domain-randomization settings.
