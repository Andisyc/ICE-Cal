# G1 Stand-Still Task Chain Audit

Date: 2026-07-09

Scope: `uv run train --algo sac --task g1_stand_still --sim mujoco`.

Non-scope: trained policy quality, long training convergence, GUI playback visual quality, and remote GitHub push authentication.

Core parameter path:

```text
CLI --task g1_stand_still
-> task=sac/g1_stand_still/mujoco
-> conf/offpolicy/task/sac/g1_stand_still/mujoco.yaml
-> training.task_name=G1StandStill
-> registry envcfg G1StandStillCfg
-> G1WalkEnv with zero-command stand-still reward config
-> MuJoCo scene_flat.xml + stand_support_task.xml
-> 98-dim actor obs / 101-dim critic obs
```

## Evidence Ledger

| Claim | Evidence | Observed fact | S/T | Result | Limitation |
| --- | --- | --- | --- | --- | --- |
| CLI routes `--task g1_stand_still` to the owner YAML. | `uv run python -c ... cli.build_command(...)` | command tail is `scripts/train_offpolicy.py`, `algo=sac`, `task=sac/g1_stand_still/mujoco`. | S0/S2, T-connect/T-oracle | covered | Does not launch training. |
| Owner config activates standalone stand-still task. | `tests/config/test_reward_injection.py::test_offpolicy_g1_stand_still_is_explicit_expert_contract` | `task_name=G1StandStill`, zero velocity limits, `rel_standing_envs=1.0`, `stand_action_authority=false`. | S0/S2, T-connect/T-oracle | covered | Offline config only. |
| Standing path does not inherit Walking/height/mixed-mode reward keys. | Parameter probe and reward injection test | `forbidden_walk_terms=[]`; no `mode_observation`, no `observe_height_command`, no `track_base_height_exp_smooth`, no `reward.mode`, no `gait_constraint`. | S0/S2, T-diff/T-oracle | covered | Does not prove learned behavior. |
| Stand posture anchor matches upstream G1WalkFlat shaping. | `test_offpolicy_g1_stand_still_pose_anchor_matches_upstream_walking_reward` | shared anchor terms, `base_height_target`, `min_base_height`, `max_tilt_deg`, and `pose_weights` match walking config. | S0/S2, T-diff/T-value | covered | Weight equality is not a physical stability proof. |
| G1StandStill envcfg only adds the support sensor fragment. | CodeGraph `G1StandStillCfg` and `tests/envs/test_env_configs.py::test_g1_stand_still_cfg_adds_support_sensor_fragment_without_polluting_walk` | `G1StandStillCfg(G1WalkFlatCfg)` with `stand_support_task.xml`; `G1WalkFlatCfg.scene.fragment_files == []`. | S0/S2, T-connect/T-oracle | covered | MuJoCo sensor runtime is covered by live sentinel below. |
| Actor observation dimension remains checkpoint-compatible with walking. | `tests/envs/locomotion/g1/test_symmetry_contract.py::test_g1_stand_still_symmetry_keeps_walking_actor_obs_dim` and live sentinel | actor obs is 98, critic obs is 101. | S1/S2/S3/S4, T-shape/T-order/T-persist/T-live | covered | Existing checkpoints still require matching run_config at playback. |
| Stand support rewards penalize the intended bad cases. | `tests/envs/locomotion/g1/test_gait_constraint.py` | clean stand outranks rear lean, wide feet, staggered x, moving base, toe-in, low crouch, missing contact, unbalanced contact, and sliding feet. | S1/S2, T-value/T-role/T-diff/T-oracle | covered | Fake backend, not physics. |
| Real MuJoCo stand-still route reaches reset/action/reward/sensor boundaries. | `uv run scripts/deploy/check_unilab_g1_stand_still_live_sentinel.py --num-envs 8 --steps 32 --seed 11 --probe-mode anchor-search --anchor-search-candidates 24 --anchor-search-iterations 3 --anchor-search-elites 6` | PASS; commands max abs 0, gait_enabled mean 0, both-feet contact 1.0, contact balance 0, obs `{"obs": 98, "critic": 101}`, no forbidden reward keys, completed 32 steps without termination. | S4, T-live/T-oracle/T-value/T-role/T-scale/T-diff | covered-with-live-sentinel | Zero-action sentinel is not a trained-policy evaluation. |
| Static nonzero support action exists and zero action should not be forced as the only anchor. | Same live sentinel | best nonzero constant action: tilt 4.09 deg, height 0.736 m, no termination; zero action at final probe: tilt 10.47 deg, height 0.729 m, no termination. | S4, T-live/T-diff/T-scale | covered-with-live-sentinel | Search is local and short, not an optimal controller. |

## Module Coverage Matrix

| Module | Owner | Module type | Required S | Required T | Existing tests / probes | Evidence | Result | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLI route | `src/unilab/cli.py` | Entrypoint / config | S0, S2 | T-connect, T-oracle | `tests/test_cli.py::test_offpolicy_g1_stand_still_task_routes_to_owner_config`; parameter probe | contract-confirmed | covered | No gap for route construction. |
| Owner YAML | `conf/offpolicy/task/sac/g1_stand_still/mujoco.yaml` | Entrypoint / config | S0, S2, S3 | T-connect, T-oracle, T-diff, T-shape, T-persist | reward injection tests; symmetry test; parameter probe | contract-confirmed | covered | Long training config quality remains empirical. |
| Registry / envcfg | `src/unilab/envs/locomotion/g1/joystick.py::G1StandStillCfg` | Env config adapter | S0, S2 | T-connect, T-oracle | CodeGraph; env config test | static-confirmed / contract-confirmed | covered | None for config wiring. |
| Support sensor asset | `src/unilab/assets/robots/g1/stand_support_task.xml` | Asset / backend input | S0, S2, S4 | T-connect, T-live | env config test; live sentinel foot linear velocity and contact metrics | runtime-confirmed | covered | Broader asset import suite not rerun here. |
| Command distribution | `sample_g1_walk_commands`, `G1WalkDomainRandomizationProvider` | Sampler / curriculum | S1, S2, S4 | T-dist, T-role, T-oracle, T-live | reward injection test; gait tests; live sentinel | contract-confirmed / runtime-confirmed | covered | Long-run distribution under training workers not separately sampled. |
| Observation / symmetry | `G1WalkEnv.obs_groups_spec`, `_compute_obs`, `symmetry.py` | Tensor layout / adapter | S1, S2, S3, S4 | T-shape, T-order, T-transform, T-persist, T-live | symmetry contract; env config tests; live sentinel | contract-confirmed / runtime-confirmed | covered | Normalizer checkpoint restore not exercised without a trained run. |
| Reward dispatch | `G1WalkEnv._init_reward_functions`, `_compute_mode_reward` | Reward / metric | S1, S2, S4 | T-value, T-role, T-oracle, T-diff, T-live | gait constraint reward lab; reward injection test; live per-term snapshot | contract-confirmed / runtime-confirmed | covered | Reward weights may still require empirical tuning after training. |
| Stand support terms | `_reward_stand_both_feet_contact`, `_reward_stand_foot_contact_balance`, `_reward_stand_feet_slide_l2` | Reward / metric | S1, S2, S4 | T-value, T-role, T-diff, T-live | support reward tiny fixture; reward lab; live sentinel contact metrics | contract-confirmed / runtime-confirmed | covered | Contact solver behavior over long training remains S4. |
| Reset lifecycle | `G1WalkDomainRandomizationProvider.build_reset_plan`, backend reset | Env lifecycle | S2, S4 | T-connect, T-oracle, T-live | live sentinel | runtime-confirmed | covered | Only 32-step short rollout. |
| Action execution | `G1WalkEnv.apply_action` | Env lifecycle / action transform | S1, S2, S4 | T-transform, T-oracle, T-scale, T-live | action authority tests; live first-step action probe; static anchor search | contract-confirmed / runtime-confirmed | covered | No trained actor action distribution yet. |
| Off-policy train route | `scripts/train_offpolicy.py`, SAC config | Runner / orchestration | S2, S3, S4 | T-connect, T-oracle, T-persist, T-live | train script tests; CLI probe | connectivity-confirmed | covered-but-live-gap | No short SAC update was run in this audit. |
| Checkpoint / playback | `scripts/play_interactive.py`, `interactive_playback.py` | Checkpoint / play | S3, S4 | T-persist, T-order, T-diff, T-live | visualization playback tests; obs dim contracts | persistence-confirmed | covered-but-live-gap | Needs a real `G1StandStill` checkpoint for playback verification. |
| Test knowledge base | `note/testing/*` | Diagnostics / notes | S0, S1 | T-connect, T-oracle | this audit note; control board; inventory | note-confirmed | covered | Must be updated after future reward/config changes. |

## Commands Run

```bash
uv run pytest tests/test_cli.py tests/config/test_reward_injection.py tests/envs/locomotion/g1/test_gait_constraint.py tests/envs/locomotion/g1/test_symmetry_contract.py tests/envs/test_env_configs.py tests/scripts/test_train_scripts.py tests/visualization/test_interactive_playback.py -q
```

Result: `325 passed, 3 skipped, 28 warnings`.

```bash
uv run ruff check src/unilab/cli.py scripts/train_offpolicy.py scripts/play_interactive.py src/unilab/envs/locomotion/g1/joystick.py tests/test_cli.py tests/config/test_reward_injection.py tests/envs/locomotion/g1/test_gait_constraint.py tests/envs/locomotion/g1/test_symmetry_contract.py tests/envs/test_env_configs.py tests/scripts/test_train_scripts.py tests/visualization/test_interactive_playback.py
```

Result: `All checks passed!`.

```bash
uv run python -m py_compile src/unilab/cli.py scripts/train_offpolicy.py scripts/play_interactive.py src/unilab/envs/locomotion/g1/joystick.py tests/test_cli.py tests/config/test_reward_injection.py tests/envs/locomotion/g1/test_gait_constraint.py tests/envs/locomotion/g1/test_symmetry_contract.py tests/envs/test_env_configs.py tests/scripts/test_train_scripts.py tests/visualization/test_interactive_playback.py
```

Result: exit code 0.

```bash
uv run scripts/deploy/check_unilab_g1_stand_still_live_sentinel.py --num-envs 8 --steps 32 --seed 11 --probe-mode anchor-search --anchor-search-candidates 24 --anchor-search-iterations 3 --anchor-search-elites 6
```

Result: `[PASS] G1StandStill real MuJoCo zero-action sentinel completed without termination`.

## Remaining Gaps

1. Trained policy quality is unconfirmed. This audit proves route wiring, reward semantics, observation shape, and short live physics reachability, not SAC convergence.
2. Checkpoint/playback for a future `G1StandStill` checkpoint is unconfirmed until a real run writes `run_config.json` and `model_*.pt`.
3. Long training distribution is unconfirmed at worker scale. The owner config sets zero-command distribution, and live reset sees zero commands, but this audit did not run a learner update.
4. Remote push is intentionally out of scope; local `main` is ahead of `origin/main` by one commit.

