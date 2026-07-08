# UniLab Semantic Objects

Semantic objects are values whose meaning crosses module boundaries. Search by all aliases before changing one.

| Object | Aliases / fields | Owners | Persistence/play risk | Required S/T |
| --- | --- | --- | --- | --- |
| G1 actor observation layout | `obs`, `obs_groups_spec`, actor obs, `mode_observation`, `observe_height_command`, command dim | `G1WalkEnv`, `train_offpolicy.py`, `play_interactive.py`, `interactive_playback.py` | High: checkpoint actor input dim must match playback env obs | S1/S2/S3; T-shape, T-order, T-persist |
| G1 command distribution | `commands`, `vel_limit`, `rel_standing_envs`, `rel_transition_envs`, `small_xy_threshold`, `gait_enabled` | `Commands`, `sample_g1_walk_commands`, DR provider | Medium: run_config restores command profile; live distribution affects policy | S1/S2/S4; T-dist, T-role, T-live |
| G1 height command | `height_commands`, `commands_height`, `height_range`, `default_height`, `observe_height_command`, `min_base_height` | `Commands`, G1 height config, `_height_command_column`, reward config | High: adds actor obs dimension and can conflict with survival height | S1/S2/S3; T-shape, T-order, T-persist, T-scale, T-oracle |
| G1 height reward scale | `track_base_height_exp_smooth`, `tracking_sigma`, `tracking_lin_vel`, `feet_phase`, `alive` | G1 height config, reward dispatch, common reward functions | Medium: can bias policy toward crouching or no-motion if scale/range dominates walking rewards | S1/S2/S4; T-value, T-meta, T-scale, T-live |
| G1 reward mode | `reward.mode`, `standing_enabled`, `stand_terms`, `walk_terms`, `stand_recovery_terms`, overrides | `G1RewardConfig`, `_compute_mode_reward` | Medium: run_config restores reward_config in playback; training distribution sensitive | S1/S2/S4; T-role, T-diff, T-live |
| G1 gait phase | `gait_phase`, `stand_phase`, `feet_phase`, `feet_phase_contrast`, `feet_phase_contact` | gait helpers, DR provider, `apply_action` | Medium: observation and reward share the same phase | S1/S2/S4; T-value, T-transform, T-live |
| G1 action execution | `current_actions`, `executed_actions`, `stand_action_authority`, `action_scale`, `default_angles`, PD target | `G1WalkEnv.apply_action` | High: direct-fall bug may live here | S1/S2/S4; T-transform, T-scale, T-live |
| G1 reset physical state | qpos, qvel, `reset_base_qvel_limit`, `standing_reset_base_qvel_limit`, base height, tilt | DR provider, backend reset, env update_state | High: direct-fall bug may live here | S2/S4; T-oracle, T-live |
| Checkpoint contract | `run_config.json`, actor state dict, actor input dim, obs_normalizer, `algo.load_run`, `algo.checkpoint` | training tracker, play scripts, visualization playback | High: old/new checkpoint compatibility | S3/S4; T-persist, T-order, T-diff, T-live |
| Replay/storage tuple | obs, actions, rewards, dones, next obs, info/log, stats | IPC and offpolicy runner | High when storage fields change | S1/S2/S3; T-shape, T-order, T-mask, T-persist |
| Normalizer / obs stats | `obs_normalization`, `obs_normalizer`, shared obs stats, mean/std | algorithms, IPC, checkpoint/play | High for export/play/eval | S1/S2/S3; T-shape, T-persist, T-diff |

## Search Rule

When debugging a semantic object, search aliases across:

```text
conf/
scripts/
src/unilab/
tests/
note/architecture/
note/testing/
```

Then assign required S/T before editing.
