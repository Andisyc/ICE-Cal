# UniLab Test Inventory

This file maps test commands to S tiers and T kinds. It is intentionally module-oriented, not just a list of pytest files.

## Core Commands

| Command | S tiers | T kinds | Main modules | Evidence class |
| --- | --- | --- | --- | --- |
| `uv run pytest tests/test_cli.py -q` | S0, S2 | T-connect, T-oracle | U-E-01, G1LOC-E-002, G1LOC-C-004; includes `--task g1_stand_still` -> `task=sac/g1_stand_still/mujoco` CLI route | contract-confirmed |
| `uv run pytest tests/scripts/test_train_scripts.py -q` | S0, S2, S3 | T-connect, T-oracle, T-persist, T-diff | U-E-02, U-C-01, U-C-02, G1LOC-T-001, G1LOC-T-002, G1LOC-P-001, G1LOC-P-002 | contract-confirmed |
| `uv run pytest tests/scripts/test_start_sh.py -q` | S0, S2 | T-connect, T-oracle | G1LOC-E-001; confirms `start.sh` walking default, explicit/shorthand `g1_stand_still`, checkpoint shortcut, keyboard default, and extra Hydra override preservation without launching viewer | contract-confirmed |
| `uv run pytest tests/config/test_reward_injection.py -q` | S0, S2 | T-connect, T-oracle, T-diff | U-C-02, U-R-02, G1LOC-C-001, G1LOC-C-002, G1LOC-C-003, G1LOC-C-004; includes upstream posture-anchor equality and static-support key isolation for G1StandStill | contract-confirmed |
| `uv run pytest tests/envs/locomotion/g1/test_gait_constraint.py -q` | S1, S2 | T-value, T-role, T-oracle, T-meta, T-diff, T-mask, T-transform | G1LOC-CMD-002, G1LOC-GAIT-001, G1LOC-RWD-001, G1LOC-OBS-001, G1LOC-RWD-002, G1LOC-GAIT-002, G1LOC-GAIT-003, G1LOC-STAND-001, G1LOC-ACT-001; includes stand-still pose/support reward lab ranking | contract-confirmed |
| `uv run pytest tests/envs/locomotion/g1/test_g1_height_tracking_contract.py -q` | S1, S2, S3 | T-shape, T-order, T-connect, T-persist, T-value, T-meta, T-scale, T-oracle | G1LOC-C-003, G1LOC-OBS-001, G1LOC-RWD-002 | contract-confirmed |
| `uv run pytest tests/envs/locomotion/g1/test_symmetry_contract.py -q` | S1, S2, S3 | T-shape, T-order, T-transform, T-persist | G1LOC-C-004, G1LOC-OBS-001, G1LOC-T-001 | contract-confirmed |
| `uv run scripts/deploy/check_unilab_g1_stand_still_live_sentinel.py --num-envs 16 --steps 64 --seed 11 --probe-mode anchor-search --anchor-search-candidates 48 --anchor-search-iterations 4 --anchor-search-elites 8` | S4 | T-live, T-oracle, T-value, T-role, T-scale, T-diff | G1LOC-C-004, G1LOC-STAND-001, G1LOC-ACT-001 | runtime-confirmed differential: zero-action fails stability gates, nonzero static anchor exists; trained-policy gap remains |
| `uv run pytest tests/visualization/test_interactive_playback.py -q` | S2, S3 | T-connect, T-persist, T-diff | U-X-01, G1LOC-P-003 | persistence-confirmed |
| `uv run pytest tests/base/test_registry.py tests/envs/test_env_configs.py -q` | S0, S2 | T-connect, T-oracle | U-R-01, G1LOC-REG-001 | contract-confirmed |
| `uv run pytest tests/base/test_np_env.py tests/utils/test_obs_utils.py tests/utils/test_final_observation.py -q` | S1, S2 | T-shape, T-order, T-mask, T-transform | U-O-01 | contract-confirmed |
| `uv run pytest tests/base/test_sim_backend.py tests/base/test_sim_backend_smoke.py -q` | S1, S2 | T-connect, T-oracle, T-shape | U-B-01 | contract-confirmed |
| `uv run pytest tests/base/test_mujoco_batch_env_jacobian.py tests/base/test_mujoco_batch_env_randomization.py tests/base/test_backend_pre_step_control.py -q` | S1, S2 | T-shape, T-connect, T-oracle | U-B-02 | contract-confirmed |
| `uv run pytest tests/base/test_motrix_backend_options.py tests/base/test_backend_imports.py -q` | S0, S2 | T-connect, T-oracle | U-B-03 | contract-confirmed |
| `uv run pytest tests/algos -q` | S1, S2, S3 | T-mask, T-value, T-grad, T-connect, T-detach, T-state | U-A-01, U-A-02 | contract-confirmed |
| `uv run pytest tests/ipc -q` | S1, S2, S3 | T-shape, T-order, T-mask, T-connect, T-persist | U-I-01, U-I-02 | connectivity-confirmed |
| `uv run pytest tests/training tests/utils/test_experiment_tracking.py -q` | S0, S2, S3 | T-connect, T-oracle, T-persist, T-diff | U-S-01 | persistence-confirmed |
| `uv run pytest tests/assets tests/test_export_scene.py tests/test_import_robot.py -q` | S0, S2, S3 | T-connect, T-persist, T-diff | U-X-01 | contract-confirmed |
| `uv run python -c "import json; from pathlib import Path; [json.loads(p.read_text()) for p in Path('note/architecture').glob('**/*.data.json')]"` | S0 | T-connect, T-oracle | U-V-01, G1LOC-V-001 | static-confirmed |
| `uv run ruff check <changed files>` | S0 | T-oracle | changed Python modules | static-confirmed |
| `uv run python -m py_compile <changed Python files>` | S0 | T-oracle | changed Python modules | static-confirmed |

## Proposed Live Sentinels

| Proposed command | S tiers | T kinds | Main modules | Status |
| --- | --- | --- | --- | --- |
| `uv run scripts/deploy/check_unilab_g1_policy_live_sentinel.py --load-run <run> --checkpoint model_5000.pt --steps 16` | S4 | T-live, T-oracle, T-scale, T-role | G1LOC-CMD-002, G1LOC-ENV-001, G1LOC-RWD-002, G1LOC-ACT-001 | missing |
| `uv run scripts/deploy/check_unilab_g1_standing_mode_dynamics.py --num-envs 16 --steps 16` | S4 | T-live, T-diff, T-oracle | G1LOC-STAND-001, G1LOC-ACT-001 | existing mixed-stage script, not the standalone G1StandStill owner route |

## Inventory Rule

When a new test is added, update this file with:

```text
command -> S tiers -> T kinds -> modules -> evidence class -> remaining gap
```
