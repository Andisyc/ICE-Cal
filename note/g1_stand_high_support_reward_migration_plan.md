# G1 Stand High-Support Reward Migration Plan

## Problem

`G1StandStill` can stand, but the trained policy still settles into a low loaded equilibrium: the knee bends, base height falls below the intended Walking-compatible height, and increasing a generic pose term did not solve the issue.

The next change must not be another isolated weight tweak. The mature pattern is a Standing-specific high-support bundle:

- height tracking relative to support/contact geometry;
- upright torso and low base acceleration;
- both-feet contact and no slip;
- reasonable foot distance / base-over-feet geometry;
- posture/default-joint regularization that does not forbid necessary loaded support action;
- action smoothness or support-action regularization, but not zero-action output.

## Reference Evidence

### Humanoid-Gym

Source:

- repo: https://github.com/roboterax/humanoid-gym
- env code: https://raw.githubusercontent.com/roboterax/humanoid-gym/main/humanoid/envs/custom/humanoid_env.py
- config code: https://raw.githubusercontent.com/roboterax/humanoid-gym/main/humanoid/envs/custom/humanoid_config.py

Observed reward family:

- `base_height`: tracks base height relative to stance/contact foot height, not just world `z`.
- `orientation`: keeps torso flat/upright.
- `default_joint_pos` and `joint_pos`: posture anchors, with special handling so yaw/roll do not drift.
- `feet_distance` and `knee_distance`: keep a healthy support footprint.
- `foot_slip`: penalizes contacted foot sliding.
- `feet_contact_number`: ties contacts to the phase/contact expectation.
- `base_acc`, `vel_mismatch_exp`, `action_smoothness`: discourage shaking and unstable high-frequency support.

Important design lesson:

`base_height` is part of a bundle. It is not used alone. The implementation measures base height against foot support geometry, which is closer to loaded support posture than a raw world-height penalty.

### Unitree RL Gym G1

Source:

- repo: https://github.com/unitreerobotics/unitree_rl_gym
- G1 config: https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/g1/g1_config.py
- base env: https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/legged_gym/envs/base/legged_robot.py

Observed G1 setup:

- default G1 posture uses a lightly bent knee pose rather than all-zero leg joints.
- `base_height_target` is explicit.
- reward scales include `orientation`, `base_height`, `alive`, `contact_no_vel`, `feet_swing_height`, `contact`, and `hip_pos`.

Important design lesson:

For G1, mature code does not assume zero action or fully straight legs. It combines a default support pose, base-height target, contact terms, and stability terms.

### Gait-Conditioned Humanoid Work

Source:

- arXiv: https://arxiv.org/abs/2505.20619

Observed method idea:

- standing, walking, running, and transitions use gait-specific objectives;
- reward routing is used to reduce interference between modes;
- straight-knee stance and natural posture are encouraged with mode-aware rewards and curriculum.

Important design lesson:

Standing and Walking should not share action-level objectives. They can share high-level stability concepts, but they need separate mode/task-specific reward bundles.

## UniLab Adaptation Contract

Do not copy reference code line-for-line. Adapt the semantic pattern into UniLab's owner structure:

| Reference concept | UniLab owner | Current state | Target adaptation |
| --- | --- | --- | --- |
| Support-relative base height | `src/unilab/envs/locomotion/g1/joystick.py`, later `standing_rewards.py` | `base_height` uses current terrain/base-height context | add Standing-only support-relative or support-aware height term |
| Upright torso | existing `penalty_orientation`, `stand_tilt_l2`, `stand_tilt_margin_l2` | active | keep, do not over-tune first |
| Both-feet contact | `stand_both_feet_contact`, contact sensor helpers | active | keep and test with height bundle |
| Contacted-foot no slip | `stand_feet_slide_l2` | active | keep as contact stability part |
| Foot distance / support footprint | `stand_feet_y_width_l2`, `stand_feet_x_l2`, `stand_base_feet_center_*` | active | keep, retune only after height bundle evidence |
| Default posture | `pose`, `stand_still`, `pose_weights`, default qpos/keyframe | active but insufficient | keep as weak posture regularizer, not the main height solution |
| Smooth/support action | action-rate/action-l2/support-action diagnostics | partially diagnostic | add only after height/contact bundle proves clean > crouch |

## Workload Estimate

This is a medium multi-step reward migration, not a one-line config edit.

- Step 1: reference-to-UniLab design contract and docs, 0.5 day.
- Step 2: S1 reward lab expansion for high-support bundle, 0.5 day.
- Step 3: implement one owner-local Standing height bundle term, 0.5-1 day.
- Step 4: config route and isolation tests, 0.5 day.
- Step 5: live sentinel before training, 0.5 day.
- Step 6: short training run and playback log review, training time dependent.

## Step Plan

### Step 1: Document And Freeze The Design Contract

Scope:

- Record reference-derived bundle design.
- Update architecture and testing docs.

Non-scope:

- No reward code change.
- No training launch.

Files:

- `note/g1_stand_high_support_reward_migration_plan.md`
- `note/g1_locomotion_modularization_atlas.md`
- `note/testing/test_control_board.md`
- `note/testing/test_inventory.md`

Test class:

- S0, T-oracle, T-connect.

Stop condition:

- Docs identify owner modules, reference concepts, test tiers, and live gap.

### Step 2: Expand Reward Lab Before Implementation

Scope:

- Add a controlled comparison for high-support stand, low crouch, high but unsupported pose, rear-lean pose, sliding feet, missing foot contact, and bad base-over-feet geometry.

Owner:

- `tests/config/test_reward_injection.py` or a dedicated `tests/envs/locomotion/g1/test_stand_high_support_reward.py`.

Required S/T:

- S1/S2.
- T-value, T-role, T-diff, T-oracle.

Expected result:

- Clean high-support stand ranks highest.
- Low crouch loses because of Standing height bundle, not because of generic pose.
- Missing contact or sliding cannot win by only matching height.

Stop condition:

- Test fails on current reward if low crouch remains too competitive.

Status, 2026-07-09:

- Added `test_g1_stand_still_high_support_bundle_prefers_loaded_high_stand_over_low_crouch`.
- Test class: S1/S2 reward lab, T-value/T-role/T-diff/T-oracle.
- The lab compares two semantic rows:
  - `clean_high_support`: base height at `base_height_target`, both feet contacted, no slip, no support action.
  - `loaded_low_crouch`: base height `0.625`, both feet contacted, no slip, upright torso, nonzero support action sampled from the latest playback trace.
- Current result is expected xfail:

```text
clean=0.280000, loaded_low=0.169298, ratio=0.605
```

- Interpretation: current `G1StandStill` reward makes low loaded crouch worse than clean stand, but not sufficiently worse; it still retains 60.5% of clean reward and has no `stand_support_height_*` owner term.
- This is a diagnostic xfail, not a physics proof. Step 3 should add the Standing high-support owner term, then this test should stop xfail-ing and become a normal passing contract.
- Fresh verification:

```text
uv run pytest tests/envs/locomotion/g1/test_gait_constraint.py::test_g1_stand_still_reward_lab_prefers_clean_stand_over_pose_failures tests/envs/locomotion/g1/test_gait_constraint.py::test_g1_stand_still_high_support_bundle_prefers_loaded_high_stand_over_low_crouch -q -rx
-> 1 passed, 1 xfailed

uv run pytest tests/envs/locomotion/g1/test_gait_constraint.py -q
-> 58 passed, 1 xfailed

uv run ruff check tests/envs/locomotion/g1/test_gait_constraint.py
-> All checks passed
```

### Step 3: Add Standing-Only High-Support Height Bundle

Scope:

- Add a Standing-only term that follows Humanoid-Gym's idea: height should be evaluated relative to support/contact geometry.
- Keep it behind `G1StandStill` reward config; do not affect `G1WalkFlat`.

Candidate term names:

- `stand_support_height_exp`
- `stand_support_height_margin_l2`

Preferred semantics:

- positive reward near target height;
- hinge penalty only below target minus tolerance;
- multiply by stand mask;
- combine with both-feet contact/contact balance so a floating or one-foot state cannot fake height.

Owner:

- initial small implementation may live beside current G1 reward methods in `joystick.py`;
- if the method grows, extract to planned `src/unilab/envs/locomotion/g1/standing_rewards.py`.

Required S/T:

- S1/S2.
- T-value, T-role, T-transform, T-oracle.

Stop condition:

- Step 2 reward lab passes and `G1WalkFlat` reward keys remain unchanged.

Status, 2026-07-09:

- Implemented `stand_support_height_margin_l2` in the G1 standing reward owner.
- Enabled it only in `conf/offpolicy/task/sac/g1_stand_still/mujoco.yaml`.
- Added config parameter `stand_support_height_margin: 0.02`.
- The term computes support-relative height as base `z` minus contacted-foot `z`; when no foot is contacted it falls back to the average of both foot heights, so missing contact cannot avoid the existing contact penalties.
- The term applies only through `_stand_mode_mask(ctx)` and therefore remains a Standing reward term.

Observed reward-lab delta:

```text
before Step 3:
clean=0.280000, loaded_low=0.169298, ratio=0.605

after Step 3:
clean=0.280000, loaded_low=0.098012, ratio=0.350
stand_support_height_margin_l2 contribution: clean=-0.0, loaded_low=-0.071286
```

Fresh verification:

```text
uv run pytest tests/envs/locomotion/g1/test_gait_constraint.py::test_g1_stand_still_high_support_bundle_prefers_loaded_high_stand_over_low_crouch -q -rx
-> 1 passed

uv run pytest tests/envs/locomotion/g1/test_gait_constraint.py -q
-> 59 passed

uv run pytest tests/config/test_reward_injection.py -q
-> 14 passed, 2 warnings

uv run pytest tests/scripts/test_start_sh.py -q
-> 5 passed

uv run pytest tests/envs/locomotion/g1/test_symmetry_contract.py -q
-> 4 passed, 8 warnings

uv run ruff check src/unilab/envs/locomotion/g1/joystick.py tests/envs/locomotion/g1/test_gait_constraint.py tests/config/test_reward_injection.py
-> All checks passed

uv run python -m py_compile src/unilab/envs/locomotion/g1/joystick.py tests/envs/locomotion/g1/test_gait_constraint.py tests/config/test_reward_injection.py
-> pass
```

Remaining gap:

- This is S1/S2 reward-ordering proof and config isolation proof, not S4 physics proof. Step 5 live sentinel is still required before long training.

### Step 4: Config Isolation And Contract Tests

Scope:

- Enable the new term only in `conf/offpolicy/task/sac/g1_stand_still/mujoco.yaml`.
- Verify `G1WalkFlat` remains pure upstream-style Walking.

Required S/T:

- S0/S2.
- T-connect, T-oracle, T-diff.

Commands:

```bash
uv run pytest tests/config/test_reward_injection.py -q
uv run pytest tests/envs/locomotion/g1/test_gait_constraint.py -q
uv run pytest tests/scripts/test_start_sh.py -q
```

Stop condition:

- `G1StandStill` has the high-support bundle.
- `G1WalkFlat` has no Standing high-support terms.

Status, 2026-07-09:

- Added explicit config isolation assertions to `tests/config/test_reward_injection.py`.
- `G1StandStill` now asserts both the active scale `stand_support_height_margin_l2: -1500.0` and parameter `stand_support_height_margin: 0.02` through the composed config and `BackendAdapter` override.
- `G1WalkFlat` asserts no `stand_support_height_margin_l2` in the backend reward scales.
- `G1WalkHeight` asserts the height-tracking path keeps `track_base_height_exp_smooth` but does not inherit `stand_support_height_margin_l2` or `stand_support_height_margin`.

Fresh verification:

```text
uv run pytest tests/config/test_reward_injection.py -q
-> 14 passed, 2 warnings

uv run pytest tests/envs/locomotion/g1/test_gait_constraint.py -q
-> 59 passed

uv run pytest tests/scripts/test_start_sh.py -q
-> 5 passed

uv run pytest tests/envs/locomotion/g1/test_symmetry_contract.py -q
-> 4 passed, 8 warnings

uv run ruff check src/unilab/envs/locomotion/g1/joystick.py tests/envs/locomotion/g1/test_gait_constraint.py tests/config/test_reward_injection.py
-> All checks passed

uv run python -m py_compile src/unilab/envs/locomotion/g1/joystick.py tests/envs/locomotion/g1/test_gait_constraint.py tests/config/test_reward_injection.py
-> pass

uv run python -c "import json; from pathlib import Path; [json.loads(p.read_text()) for p in Path('note/architecture').glob('**/*.data.json')]; print('architecture json ok')"
-> architecture json ok
```

Evidence ledger:

| Claim | Evidence | S/T | Result | Limitation |
| --- | --- | --- | --- | --- |
| `G1StandStill` owns the high-support height bundle. | `test_offpolicy_g1_stand_still_is_explicit_expert_contract`; `uv run pytest tests/config/test_reward_injection.py -q`. | S0/S2, T-connect/T-oracle | contract-confirmed | Does not prove live standing height. |
| `G1WalkFlat` remains pure upstream-style Walking. | `test_offpolicy_g1_env_override_preserves_upstream_walking_contract`; same config test command. | S0/S2, T-diff/T-oracle | contract-confirmed | Does not prove trained walking quality. |
| `G1WalkHeight` keeps height tracking without Standing high-support pollution. | `test_g1_height_sac_config_exposes_explicit_height_fields`; same config test command. | S0/S2, T-diff/T-connect | contract-confirmed | Live height-tracking policy quality remains separate. |

Remaining gap:

- Step 4 proves config isolation only. Step 5 live sentinel is still required before using the reward change as a training-quality claim.

### Step 5: Live Sentinel Before Training

Scope:

- Compare zero action, searched support action, and untrained/current policy action under the new reward log.

Required S/T:

- S4.
- T-live, T-diff, T-scale, T-oracle.

Command shape:

```bash
uv run scripts/deploy/check_unilab_g1_stand_still_live_sentinel.py --num-envs 8 --steps 128 --probe-mode support-policy-diff
```

Expected facts:

- high-support action gets better high-support reward than low crouch;
- low height is visible as the failing term;
- no walking/gait terms are active.

Stop condition:

- Live diagnostic proves the reward distinguishes high loaded stand from low loaded crouch before paying for a long training run.

Status, 2026-07-09:

- Executed the S4 live sentinel with `support-policy-diff`.
- The live path is confirmed to use `G1StandStill`, zero commands, no mode observation, no height-command observation, and no Walking/gait/height-tracking reward keys.
- The 64-step sentinel proves a high-support nonzero action is physically reachable and reward-visible:
  - `searched_support_action`: height min `0.7219`, height deficit max `0.0321`, tilt max `11.49 deg`, base-over-feet x abs max `0.0172`, both feet contact `1.0`, reward mean about `-0.0234`.
  - `zero_action`: height min `0.5792`, height deficit max `0.1748`, tilt max `38.59 deg`, base-over-feet x abs max `0.4801`, both feet contact `0.0`, reward mean about `-2.2283`.
  - `current_trained_policy_action`: height min `0.6331`, tilt max `1.17 deg`, both feet contact `1.0`, reward mean about `0.1378`.
- The 128-step sentinel exposes a longer-horizon gap:
  - `searched_support_action` is not durable as a constant action and reaches low height at step `65`.
  - `current_trained_policy_action` remains dynamically stable but still sits near `0.6330`, below the desired high-support target `0.754`.
  - `zero_action` remains unstable, with tilt around `42 deg` and base-over-feet x around `0.505`.

Fresh verification:

```text
uv run scripts/deploy/check_unilab_g1_stand_still_live_sentinel.py --num-envs 8 --steps 128 --probe-mode support-policy-diff
-> exit 1, because a role terminates at step 65 with low_height; useful live details printed.

uv run python -c "... run_check(num_envs=8, steps=128, probe_mode='support-policy-diff') ..."
-> completed_steps=65, forbidden_reward_keys_present=[], current_trained_policy_action height_min=0.6330, zero_action height_min=0.5616, searched_support_action height_min=0.2726.

uv run python -c "... run_check(num_envs=8, steps=64, probe_mode='support-policy-diff') ..."
-> completed_steps=64, forbidden_reward_keys_present=[], searched_support_action height_min=0.7219, zero_action height_min=0.5792, current_trained_policy_action height_min=0.6331.
```

Evidence ledger:

| Claim | Evidence | S/T | Result | Limitation |
| --- | --- | --- | --- | --- |
| The live sentinel reaches the standalone `G1StandStill` route. | `support-policy-diff` live run: task `G1StandStill`, commands max `0`, gait enabled `0`, forbidden reward keys `[]`. | S4, T-live/T-oracle | runtime-confirmed | Does not prove training convergence. |
| The high-support reward term is visible in real MuJoCo. | 64-step per-term role summary: `stand_support_height_margin_l2` is near zero for high-support searched action and more negative for low zero action. | S4, T-live/T-role/T-scale | runtime-confirmed | The searched action is a diagnostic constant action, not a learned policy. |
| A high-support loaded pose is physically reachable for at least a short horizon. | 64-step `searched_support_action`: height min `0.7219`, tilt max `11.49 deg`, both feet contact `1.0`. | S4, T-diff/T-oracle | runtime-confirmed | It fails as a constant action over 128 steps. |
| The current trained policy improves stability over zero action but remains low. | 128-step role summary: policy tilt max `1.11 deg`, height min `0.6330`; zero tilt max `42.07 deg`, height min `0.5616`. | S4, T-diff/T-scale | runtime-confirmed | Policy is still below the intended `0.754` high-support target. |

Remaining gap:

- Step 5 does not yet justify a long training claim. It proves the reward/live route and short-horizon high-support feasibility, but Step 6 must judge whether learning can turn that short-horizon support into a durable policy.

Post-log regression, 2026-07-09:

- User playback log `log.txt` showed the policy still settles around `base_height=0.676-0.677`, with low tilt and both feet in contact.
- Root cause in reward ordering: the earlier reward lab checked severe low crouches (`0.625` and below), but not the actual trained low equilibrium around `0.677`.
- Added `test_g1_stand_still_high_support_bundle_rejects_trained_low_equilibrium`.
- Before the fix, the `0.677` low equilibrium scored `0.2148` versus clean `0.2800`, ratio `0.767`, so it could remain attractive.
- Increased only the standalone `G1StandStill` high-support scale:
  - `stand_support_height_margin_l2: -300.0 -> -1500.0`.
- Also changed `G1ActionTrace` to print all `reward/*` keys, because the previous first-24-key truncation hid `reward/stand_support_height_margin_l2` from the log.

Fresh verification after the post-log fix:

```text
uv run pytest tests/envs/locomotion/g1/test_gait_constraint.py::test_g1_stand_still_high_support_bundle_rejects_trained_low_equilibrium -q
-> 1 passed

uv run pytest tests/envs/locomotion/g1/test_gait_constraint.py -q
-> 60 passed

uv run pytest tests/config/test_reward_injection.py -q
-> 14 passed, 2 warnings

uv run ruff check src/unilab/envs/locomotion/g1/joystick.py tests/envs/locomotion/g1/test_gait_constraint.py tests/config/test_reward_injection.py
-> All checks passed

uv run python -m py_compile src/unilab/envs/locomotion/g1/joystick.py tests/envs/locomotion/g1/test_gait_constraint.py tests/config/test_reward_injection.py
-> pass
```

Live caveat:

- Existing checkpoints will not stand higher automatically, because their actor was trained under the weaker reward.
- A 64-step live summary with the old checkpoint and new reward shows `current_trained_policy_action` still around `0.674-0.678`, but now receives `stand_support_height_margin_l2=-0.0628` per step, while `searched_support_action` near `0.722` receives only `-0.00037`.
- Therefore the next meaningful evidence is retraining or short resume under the strengthened reward, then playback.

Second post-log regression, 2026-07-09:

- User playback log `log.txt` showed retraining improved the policy height to around `base_height=0.708-0.709`, but the stance remained visibly lower than the Walking-compatible target `0.754`.
- Runtime log fact:
  - `height min/mean/max = 0.708413 / 0.716899 / 0.754`.
  - post-reset settled steps around `0.709` show `height_deficit≈0.045`.
- Reward-lab probe before the fix:
  - `clean=0.280000`, `trained_0709=0.245936`, ratio `0.878`.
  - `base_height` contribution was only `-0.008100`.
  - `stand_support_height_margin_l2` contribution was only `-0.018750` when feet are at terrain height in the offline oracle.
- Root cause in reward ordering: the support-relative term was necessary, but it did not directly make a visible `0.709m` base-height equilibrium unattractive enough.
- Added a Standing-only visible low-height hinge:
  - `stand_base_height_deficit_l1: -240.0`.
  - formula path: `base_height -> max(base_height_target - stand_support_height_margin - base_height, 0) -> stand mask`.
  - This stays in `G1StandStill`; `G1WalkFlat` and `G1WalkHeight` remain isolated by config tests.
- Added `test_g1_stand_still_height_bundle_rejects_trained_0709_equilibrium`.
- Reward-lab probe after the fix:
  - `clean=0.280000`, `trained_0709=0.125936`, ratio `0.450`.
  - `stand_base_height_deficit_l1` contribution is `-0.120000` on the `0.709m` row.

Fresh verification after the 0.709m fix:

```text
uv run pytest tests/envs/locomotion/g1/test_gait_constraint.py::test_g1_stand_still_height_bundle_rejects_trained_0709_equilibrium -q
-> 1 passed

uv run pytest tests/envs/locomotion/g1/test_gait_constraint.py -q
-> 61 passed

uv run pytest tests/config/test_reward_injection.py -q
-> 14 passed, 2 warnings
```

Remaining gap:

- This is S1/S2 reward-ordering evidence and S0/S2 config-route evidence. It still requires retraining or resume training before the actor behavior can be expected to change in playback.

### Step 6: Short Training And Playback Gate

Scope:

- Train `task=g1_stand_still` for a short checkpoint interval.
- Inspect `base_height`, `height_deficit`, `tilt`, `both_feet_contact`, `feet_slide`, and per-term reward in playback.

Required S/T:

- S4.
- T-live, T-persist, T-diff.

Stop condition:

- If height remains low, inspect whether the actor action is trying to raise height and failing, or whether reward still makes crouch locally attractive.

## Open Risk

- A support-relative height term can be gamed if foot contact or foot geometry is weak. Therefore it must be bundled with contact and slip checks.
- A too-strict low-height penalty can make early training unstable. Prefer reward lab and live sentinel before full training.
- This plan does not change Walking. Any Walking term change must be treated as a separate migration.

## Step 1 Execution Status

Status: executed as design/docs/test-governance only.

Execution contract:

| Field | Value |
| --- | --- |
| Scope | Freeze the reference-derived Standing high-support reward design and align repo docs/test matrix. |
| Non-scope | No reward implementation, no config weight change, no training launch, no playback run. |
| Owner module | Current active owner: `src/unilab/envs/locomotion/g1/joystick.py`; planned extraction owner: `standing_rewards.py`. |
| Core parameter path | `G1StandStill` reward concept -> reward term names/config scales -> reward lab ranking -> live sentinel per-term logs -> later training/playback. |
| Test class | Secondary contract path for docs/test governance; future core param path starts in Step 2. |
| Stop condition | Docs identify reference concepts, UniLab owners, required S/T tiers, missing tests, and live-only boundaries. |

Evidence ledger:

| Claim | Evidence | S/T | Result | Limitation |
| --- | --- | --- | --- | --- |
| The design is recorded from mature references, not invented as a one-off patch. | This file, `Reference Evidence` and `UniLab Adaptation Contract`. | S0, T-oracle | note-confirmed | External code was inspected by URL/search, not vendored into UniLab. |
| Architecture ownership is aligned. | `note/g1_locomotion_modularization_atlas.md`, `Standing High-Support Reward Migration`. | S0, T-connect | note-confirmed | Planned `standing_rewards.py` is not implemented yet. |
| Test governance knows the missing reward-lab and live-sentinel boundaries. | `note/testing/test_control_board.md` rows `G1LOC-STAND-002` and `G1LOC-STAND-003`. | S0, T-oracle | note-confirmed | Tests are proposed, not implemented. |
| Inventory lists the next smallest tests. | `note/testing/test_inventory.md`, `Proposed Reward Lab Tests` and live sentinel entry. | S0, T-connect | note-confirmed | No S1/S2 reward-lab execution yet. |
| Semantic-object impact is registered. | `note/testing/semantic_objects.md` and `note/testing/impact_rules.md` rows for `G1 stand high-support height bundle`. | S0, T-connect | note-confirmed | Live physics and training quality remain S4 gaps. |

Fresh verification:

```bash
uv run python -c "import json; from pathlib import Path; [json.loads(p.read_text()) for p in Path('note/architecture').glob('**/*.data.json')]; print('architecture json ok')"
```

Expected result:

- architecture JSON parse remains valid.

Next safest action:

- Step 2: write the reward lab that must fail or expose the current low-crouch ordering before any reward implementation.
