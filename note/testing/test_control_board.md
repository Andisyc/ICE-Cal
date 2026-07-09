# UniLab Test Control Board

Scope: all atlas modules in `note/architecture/architecture/01_unilab_repo_architecture.data.json` and `note/architecture/architecture/02_g1_locomotion_modularization.data.json`.

Evidence rule: this board assigns required S tiers and T kinds. It is not a claim that all rows are covered. `covered-but-live-gap` means offline/contract tests exist but the live simulator/training boundary is still unconfirmed.

## S/T Legend

- S0 Static: compile/import/config/schema/stale-text checks.
- S1 Module Semantic: deterministic tiny/golden module behavior.
- S2 Offline Connectivity: fake route across owners, no physics claim.
- S3 Persistence / Semantic Object: checkpoint/resume/export/play/eval/storage/normalizer lifecycle.
- S4 Live Sentinel: real env/reset/rollout/update/play boundary with compact runtime facts.

## Coverage Matrix

| Module | Owner | Module type | Required S | Required T | Existing tests / evidence | Result | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| U-E-01 | `src/unilab/cli.py` | Entrypoint / CLI / config | S0, S2 | T-connect, T-oracle | `tests/test_cli.py`; atlas `U-E-01` | covered | Need update when new feature flag is added. |
| U-E-02 | `scripts/train_*.py` | Runner / orchestration | S0, S2, S3, S4 for live play/train | T-connect, T-oracle, T-persist, T-live | `tests/scripts/test_train_scripts.py`; `tests/scripts/test_train_script_configs.py` | covered-but-live-gap | Real long train/play lifecycle not covered by quick tests. |
| U-C-01 | `conf/*/config.yaml` | Entrypoint / CLI / config | S0, S2 | T-connect, T-oracle | `tests/scripts/test_train_scripts.py`; `tests/config/test_config_system.py` | covered | Add per-algo config row when adding an algo root. |
| U-C-02 | `conf/*/task/<task>/<backend>.yaml` | Entrypoint / CLI / config | S0, S2 | T-connect, T-oracle | `tests/scripts/test_train_scripts.py`; `tests/config/test_reward_injection.py`; `tests/config/test_locomotion_params.py` | covered | Backend-specific runtime behavior remains S4. |
| U-R-01 | `src/unilab/base/registry.py` | Env command / reset / lifecycle adapter | S0, S1, S2 | T-connect, T-oracle, T-shape | `tests/base/test_registry.py`; `tests/envs/test_env_configs.py` | covered | Unknown future env variants need new registry rows. |
| U-R-02 | `src/unilab/base/registry.py` | Tensor/config adapter | S0, S1, S2 | T-shape, T-order, T-transform, T-connect | `tests/base/test_reward_override.py`; `tests/config/test_reward_injection.py` | covered | Nested dataclass edge cases should add golden overrides. |
| U-O-01 | `src/unilab/base/{np_env,base,scene,observations}.py` | Tensor layout / adapter | S0, S1, S2 | T-shape, T-order, T-mask, T-transform | `tests/base/test_np_env.py`; `tests/utils/test_final_observation.py`; `tests/utils/test_obs_utils.py` | covered | Live env reset semantics remain per-task S4. |
| U-B-01 | `src/unilab/base/backend/base.py` | Backend abstraction | S0, S1, S2, S4 for backend changes | T-connect, T-oracle, T-live | `tests/base/test_sim_backend.py`; `tests/base/test_sim_backend_smoke.py` | covered-but-live-gap | Real backend timing/render is live-only. |
| U-B-02 | `src/unilab/base/backend/mujoco/*` | Backend adapter | S0, S1, S2, S4 | T-shape, T-connect, T-oracle, T-live | `tests/base/test_mujoco_batch_env_jacobian.py`; `tests/base/test_mujoco_batch_env_randomization.py`; `tests/base/test_backend_pre_step_control.py` | covered-but-live-gap | G1 falling requires task-level S4, not only backend smoke. |
| U-B-03 | `src/unilab/base/backend/motrix/*` | Backend adapter | S0, S1, S2, S4 | T-shape, T-connect, T-oracle, T-live | `tests/base/test_motrix_backend_options.py`; `tests/base/test_backend_imports.py` | covered-but-live-gap | Motrix native runtime availability is environment-dependent. |
| U-T-01 | `src/unilab/envs/locomotion/*` | Env command / reset / lifecycle adapter | S1, S2, S4 | T-connect, T-oracle, T-live, T-shape | `tests/envs/locomotion/*`; `tests/envs/test_g1_obs_noise.py`; G1 rows below | covered-but-live-gap | Current G1 policy collapse is an S4 gap. |
| U-T-02 | `src/unilab/envs/manipulation/*` | Env command / reset / lifecycle adapter | S1, S2, S4 | T-connect, T-oracle, T-live | `tests/envs/test_allegro_domain_randomization.py`; `tests/envs/test_sharpa.py`; `tests/envs/test_stewart.py` | covered-but-live-gap | Live rollout per manipulation task not proven here. |
| U-T-03 | `src/unilab/envs/motion_tracking/*` | Env command / reset / lifecycle adapter | S1, S2, S4 | T-connect, T-oracle, T-live, T-shape | `tests/envs/test_motion_loader.py`; `tests/scripts/test_train_scripts.py` motion-tracking entries | covered-but-live-gap | Full motion playback/eval remains S4. |
| U-A-01 | `src/unilab/algos/torch/*` | Algorithm / loss / optimizer | S1, S2, S3 for checkpoint | T-mask, T-value, T-grad, T-connect, T-detach, T-state | `tests/algos/test_*`; `tests/ipc/test_*`; `tests/training/test_resume_logger_state.py` | covered | Add S4 only when debugging live learner/collector drift. |
| U-A-02 | `src/unilab/algos/mlx/*` | Algorithm / loss / optimizer | S1, S2, S3 | T-value, T-grad, T-connect, T-state | `tests/algos/test_mlx_ppo.py` | covered | Apple Silicon runtime performance is not asserted. |
| U-I-01 | `src/unilab/ipc/async_runner.py` | Runner / orchestration | S2, S4 for process lifecycle | T-connect, T-oracle, T-live | `tests/ipc/test_async_runner.py`; `tests/ipc/test_nan_guard_spawn_pickle.py` | covered-but-live-gap | Real collector subprocess failure timing is S4. |
| U-I-02 | `src/unilab/ipc/{shared_buffer,replay_buffer,rollout_ring_buffer,weight_sync}.py` | Storage / batch tuple | S1, S2, S3 | T-shape, T-order, T-mask, T-connect, T-persist | `tests/ipc/test_replay_buffer.py`; `test_rollout_ring_buffer.py`; `test_shared_obs_stats.py`; `test_shared_weight_sync.py` | covered | Add semantic object row when tuple layout changes. |
| U-S-01 | `src/unilab/training/*` | Checkpoint / config / orchestration support | S0, S2, S3 | T-connect, T-oracle, T-persist, T-diff | `tests/training/test_sim2sim_resolver.py`; `test_seed_contract.py`; `test_training_helpers.py`; `tests/utils/test_experiment_tracking.py` | covered | Any new checkpoint field requires S3 row. |
| U-X-01 | `src/unilab/assets/* / tools/* / visualization/* / demo.py` | Checkpoint / export / play / diagnostics | S0, S2, S3, S4 for GUI | T-connect, T-persist, T-diff, T-live | `tests/assets/test_hub.py`; `tests/visualization/test_interactive_playback.py`; `tests/test_export_scene.py`; `tests/test_import_robot.py` | covered-but-live-gap | GUI playback and visual correctness are live/manual. |
| U-V-01 | `tests/ benchmark/ docs/sphinx/source/adr/*` | Diagnostics / notes / atlas | S0, S1 | T-connect, T-oracle | `tests/scripts/test_repo_hygiene.py`; `tests/scripts/doc_checks.py`; atlas parse command | covered | This board must be updated when atlas IDs change. |
| G1LOC-E-001 | `start.sh` | Entrypoint / CLI / config | S0, S2 | T-connect, T-oracle | `tests/scripts/test_start_sh.py`; dry-run confirms walking default, explicit/shorthand `g1_stand_still`, checkpoint shortcut, keyboard default, and extra Hydra override preservation | covered | GUI playback remains G1LOC-P-002/P-003 live/manual. |
| G1LOC-E-002 | `src/unilab/cli.py` | Entrypoint / CLI / config | S0, S2 | T-connect, T-oracle | `tests/test_cli.py`; `tests/config/test_reward_injection.py` | covered | Add rows when flag semantics change. |
| G1LOC-C-001 | `conf/offpolicy/task/sac/g1_walk_flat/mujoco.yaml` | Entrypoint / CLI / config | S0, S2 | T-connect, T-oracle, T-diff | `test_offpolicy_g1_env_override_preserves_upstream_walking_contract` | covered-but-live-gap | Does not prove trained policy stability. |
| G1LOC-C-002 | `conf/offpolicy/g1_walk_stage/mixed_mode.yaml` | Entrypoint / CLI / config | S0, S2 | T-connect, T-oracle, T-diff | `test_offpolicy_g1_standing_reward_is_explicit_stage_contract`; stage config tests | covered-but-live-gap | Standing quality still S4. |
| G1LOC-C-003 | `conf/offpolicy/task/sac/g1_walk_height/mujoco.yaml` | Entrypoint / CLI / config | S0, S2, S3 | T-connect, T-oracle, T-shape, T-persist, T-scale | `test_g1_height_sac_config_*`; height range/survival and reward-scale tests in `test_g1_height_tracking_contract.py` | covered-but-live-gap | Offline contract confirms no Standing path and no below-survival target; live height tracking quality unconfirmed. |
| G1LOC-C-004 | `conf/offpolicy/task/sac/g1_stand_still/mujoco.yaml` | Entrypoint / CLI / config | S0, S2, S3 | T-connect, T-oracle, T-shape, T-persist, T-diff | `test_offpolicy_g1_stand_still_is_explicit_expert_contract`; `test_offpolicy_g1_stand_still_pose_anchor_matches_upstream_walking_reward`; `test_g1_stand_still_symmetry_keeps_walking_actor_obs_dim` | covered-but-live-gap | Offline contract confirms no Walking/height/mode route, 98-dim actor obs, upstream G1WalkFlat posture anchor, and agile-style static support reward keys; live standing stability still needs S4 after training. |
| G1LOC-CMD-001 | `src/unilab/envs/locomotion/common/commands.py` | Sampler / curriculum / priority | S1, S2 | T-dist, T-meta, T-role | `test_common_small_xy_threshold_zeroes_low_speed_xy_commands`; height tests | covered | More distribution tests needed if sampling rules change. |
| G1LOC-GAIT-001 | `g1/joystick.py` gait pure helpers | Pure helper / math | S1 | T-value, T-meta, T-scale | `test_gait_phase_violation_zero_when_feet_match_generator`; phase target tests | covered | Should move with helpers when modularized. |
| G1LOC-RWD-001 | `g1/joystick.py` reward config dataclasses | Entrypoint / CLI / config | S0, S1 | T-connect, T-oracle, T-value | `test_reward_config_converts_gait_constraint_dict`; `test_reward_config_converts_reward_mode_dict` | covered | Config schema gaps if new fields added. |
| G1LOC-CMD-002 | `g1/joystick.py` G1 command distribution | Sampler / curriculum / priority | S1, S2, S4 for live reset | T-dist, T-meta, T-role, T-live | `test_g1_reset_info_writes_gait_enabled_from_sampled_command`; transition/resampling tests | covered-but-live-gap | Need live command histogram in falling run. |
| G1LOC-ENV-001 | `g1/joystick.py` env lifecycle | Env command / reset / lifecycle adapter | S2, S4 | T-connect, T-oracle, T-live | fake env tests in `test_gait_constraint.py` | covered-but-live-gap | Need real reset pose/base qvel/termination sentinel. |
| G1LOC-OBS-001 | `g1/joystick.py` observation contract | Tensor layout / adapter | S1, S2, S3 | T-shape, T-order, T-mask, T-transform, T-persist | `test_mode_observation_*`; `test_g1_height_tracking_contract.py`; playback dim tests | covered | Add normalizer/checkpoint test if obs normalization changes. |
| G1LOC-RWD-002 | `g1/joystick.py` reward dispatch | Reward / metric / evaluator | S1, S2, S4 for live reward health | T-value, T-role, T-oracle, T-meta, T-diff, T-scale, T-live | reward mode dispatch/gating/log tests in `test_gait_constraint.py`; height-vs-walking reward scale tests in `test_g1_height_tracking_contract.py` | covered-but-live-gap | Need per-term live reward snapshot for direct falling or height-tracking gait quality. |
| G1LOC-GAIT-002 | `g1/joystick.py` gait constraint bridge | Reward / metric / evaluator | S1, S2 | T-value, T-role, T-oracle, T-mask | gait constraint component/gate tests | covered | Live gait quality is covered by G1LOC-RWD-002/S4 gap. |
| G1LOC-GAIT-003 | `g1/joystick.py` walking gait rewards | Reward / metric / evaluator | S1, S2, S4 for policy behavior | T-value, T-role, T-oracle, T-diff, T-live | `test_feet_phase_reward_*`; walk-only/gated tests | covered-but-live-gap | Need low-speed walking live probe if shuffling returns. |
| G1LOC-STAND-001 | `g1/joystick.py` standing behavior | Reward / metric / evaluator | S1, S2, S4 | T-value, T-role, T-oracle, T-diff, T-live | stand geometry/recovery/action-authority tests; static-support tests; `test_g1_stand_still_reward_lab_prefers_clean_stand_over_pose_failures`; `scripts/deploy/check_unilab_g1_stand_still_live_sentinel.py` | covered-with-live-differential | Offline clean stand now outranks bad pose and bad support rows: rear lean, wide feet, staggered x, moving base, toe-in, low crouch, missing contact, unbalanced contact, sliding feet. Support-aware live sentinel confirms the support sensors/reward terms are wired, but 64-step zero-action fails stability gates: tilt max 43.09 deg, base-over-feet x up to 0.4961m, height min 0.5669m, both-feet contact mean 0.0. 64-step static-anchor search finds a nonzero action with tilt 3.43 deg, height 0.6598m, base-over-feet x -0.0254m, no termination. Trained policy quality remains unproven. |
| G1LOC-ACT-001 | `g1/joystick.py` action and phase execution | Env command / reset / lifecycle adapter | S1, S2, S4 | T-transform, T-oracle, T-live, T-scale | apply_action phase/action authority tests; live sentinel `--probe-mode anchor-diff`; live sentinel `--probe-mode anchor-search` | covered-with-live-failure | First-step probe shows `reset_dof_pos == default_angles`, action_scale 1.0, zero base qvel, and zero/hold-reset actions are identical. Failure boundary is missing static support-action anchor, not default/reset mismatch. |
| G1LOC-REG-001 | `g1/joystick.py` env cfg/registry | Entrypoint / CLI / config | S0, S2 | T-connect, T-oracle | registry/config tests and G1 config composition tests | covered | Add row when new G1 task variant registers. |
| G1LOC-T-001 | `scripts/train_offpolicy.py` training dims | Runner / orchestration | S2, S3, S4 for live train | T-connect, T-oracle, T-persist, T-live | `test_offpolicy_*`; symmetry contract tests | covered-but-live-gap | Need short live rollout/update sentinel for falling checkpoint family. |
| G1LOC-T-002 | `scripts/train_offpolicy.py` non-interactive play | Checkpoint / resume / export / play | S3, S4 | T-persist, T-order, T-diff, T-live | `test_play_offpolicy_*`; sim2sim tests | covered-but-live-gap | GUI/live playback quality not proven. |
| G1LOC-P-001 | `scripts/play_interactive.py` checkpoint env contract | Checkpoint / resume / export / play | S3, S4 for GUI | T-persist, T-order, T-diff, T-live | `test_play_interactive_*checkpoint_env_contract*`; latest 98/99/100 tests | covered-but-live-gap | It fixes dimension recovery only, not policy stability. |
| G1LOC-P-002 | `scripts/play_interactive.py` interactive env creation | Runner / orchestration | S2, S3, S4 | T-connect, T-oracle, T-persist, T-live | `test_play_interactive_*`; wrapper reset tests | covered-but-live-gap | Real `mjpython` viewer path remains live/manual. |
| G1LOC-P-003 | `src/unilab/visualization/interactive_playback.py` SAC playback session | Checkpoint / resume / export / play | S3, S4 | T-persist, T-order, T-diff, T-live | `tests/visualization/test_interactive_playback.py`; script tests | covered-but-live-gap | Live MPS/mjpython policy behavior unconfirmed. |
| G1LOC-V-001 | G1 tests | Diagnostics / notes / atlas | S0, S1 | T-connect, T-oracle | This board plus existing G1 test files | covered | Must remain updated after modularization split. |

## Missing Tests

| Module | Missing S/T | Proposed smallest test | Evidence target |
| --- | --- | --- | --- |
| G1LOC-CMD-002 | S4: T-live, T-dist | Short no-viewer G1 reset probe prints command histogram, gait_enabled fraction, standing/transition/walk fractions. | prove training distribution actually matches selected profile. |
| G1LOC-ENV-001 | S4: T-live, T-oracle | Short rollout probe prints reset qpos/qvel, base height, tilt, termination reason at step 0/1. | covered by stand-still live sentinel: reset qvel zero and reset/default posture aligned. |
| G1LOC-RWD-002 | S4: T-live, T-role | Per-term live reward snapshot plus foot contact, contact balance, and contacted-foot slide metrics. | distinguish bad reward shaping, bad support contact, and action/model issue. |
| G1LOC-ACT-001 | S4: T-live, T-scale | First-step action probe plus static lower-body anchor search. | covered for current bug: action scale/default pose are aligned; nonzero static support action is required. |
| G1LOC-STAND-001 | S4: T-live, T-diff | Zero-action vs static-anchor standing recovery differential rollout, then trained-policy rollout. | next: train/evaluate the support-aware `G1StandStill` expert. |

## Next Baseline Commands

Full quick baseline after code edits:

```bash
uv run pytest tests/test_cli.py tests/config/test_reward_injection.py tests/envs/locomotion/g1/test_gait_constraint.py tests/envs/locomotion/g1/test_g1_height_tracking_contract.py tests/scripts/test_train_scripts.py tests/visualization/test_interactive_playback.py -q
uv run ruff check src/unilab/cli.py scripts/train_offpolicy.py scripts/play_interactive.py src/unilab/envs/locomotion/g1/joystick.py tests/config/test_reward_injection.py tests/envs/locomotion/g1/test_gait_constraint.py tests/scripts/test_train_scripts.py
uv run python -m py_compile src/unilab/cli.py scripts/train_offpolicy.py scripts/play_interactive.py src/unilab/envs/locomotion/g1/joystick.py
```

Live baseline for the current direct-fall bug:

```bash
uv run scripts/deploy/check_unilab_g1_policy_live_sentinel.py --load-run <run> --checkpoint model_5000.pt --steps 16
```

Status: proposed, not implemented yet.
