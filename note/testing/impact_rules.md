# UniLab Impact Rules

Use these rules before choosing tests. Never test only the edited file when the semantic object crosses config, env, runner, checkpoint, or playback boundaries.

## G1 Locomotion

| Changed object | Impact expansion | Required S/T |
| --- | --- | --- |
| `env.mode_observation` | config -> `G1WalkEnv.obs_groups_spec` -> `_compute_obs` -> symmetry layout -> learner obs_dim -> checkpoint/playback dim guard | S0/S2/S3; T-shape, T-order, T-connect, T-persist |
| `commands.observe_height_command` | height config -> command sampler -> obs dim -> symmetry command dim -> checkpoint/playback dim guard | S1/S2/S3; T-shape, T-order, T-connect, T-persist |
| `commands.height_range` / `reward.scales.track_base_height_exp_smooth` | height config -> height command sampler -> reward target -> `min_base_height` survival bound -> walking reward scale budget -> live posture/gait quality | S1/S2/S4; T-value, T-meta, T-scale, T-oracle, T-live |
| `commands.rel_standing_envs` / `rel_transition_envs` | command sampler -> `gait_enabled` -> reward masks -> gait phase freeze -> action authority -> training distribution | S1/S2/S4; T-dist, T-role, T-oracle, T-live |
| `reward.mode` | config -> mode masks -> stand/walk/recovery dispatch -> reward logs -> policy objective | S1/S2/S4; T-value, T-role, T-diff, T-live |
| `reward.gait_constraint` | config -> gait helper components -> gate mask -> reward subtraction -> phase freeze | S1/S2/S4 when live gait changes; T-value, T-mask, T-oracle, T-live |
| `feet_phase` rewards | gait phase -> foot sensor positions/contacts -> reward term -> low-speed walking behavior | S1/S2/S4; T-value, T-role, T-diff, T-live |
| `stand_action_authority` | command mode -> action execution -> PD target -> stability/recovery | S1/S2/S4; T-transform, T-oracle, T-scale, T-live |
| `action_scale` / default pose | config -> actor output -> `apply_action` -> PD target -> first-step physics | S2/S4; T-transform, T-scale, T-live |
| checkpoint/run_config playback | run_config -> `_apply_checkpoint_env_contract` -> env obs_dim -> actor load -> interactive command probe | S3/S4; T-persist, T-order, T-diff, T-live |

## Cross-Repo Runtime

| Changed object | Impact expansion | Required S/T |
| --- | --- | --- |
| Backend interface | `SimBackend` -> MuJoCo/Motrix impls -> env call sites -> playback render plans | S0/S2/S4; T-connect, T-oracle, T-live |
| Registry override merge | Hydra owner YAML -> dataclass config -> env constructor -> reward/DR fields | S0/S2; T-connect, T-oracle, T-transform |
| Storage tuple / replay buffer | collector write -> shared buffer -> minibatch -> learner update -> checkpoint/resume if persisted | S1/S2/S3; T-shape, T-order, T-mask, T-persist |
| Normalizer / obs stats | observation layout -> normalizer shape/state -> checkpoint save/load -> export/play/eval | S1/S2/S3; T-shape, T-order, T-persist, T-diff |
| Runner helper | CLI/config -> env lifecycle -> storage -> learner -> checkpoint/logging | S2/S4; T-connect, T-oracle, T-live |
| Atlas / notes | data JSON -> viewer route -> index/README link -> parse check | S0/S1; T-connect, T-oracle |

## Current Direct-Fall Bug Rule

Do not adjust reward weights again until an S4 live sentinel has printed:

- resolved config owner and feature flags;
- actor obs_dim, checkpoint actor input dim, and optional observation flags;
- reset qpos/qvel, base height, tilt, and termination threshold;
- command histogram and `gait_enabled` fractions;
- first-step raw action, executed action, PD target, and action_scale;
- reward mode flags and per-term reward snapshot;
- termination reason at step 0/1.

The first failing row determines the owner module to change.
