# UniLab Planner-IDM Integration Evidence

Date: 2026-08-04

Scope: default-off FADA training route through source Planner-IDM construction.

## Evidence

- E6: `uv run --frozen --no-sync pytest tests/algos/test_fada_planner_idm.py tests/algos/test_fada_unilab_training.py tests/scripts/test_train_scripts.py::test_distill_main_routes_enabled_single_entry_workflow -q` -> `13 passed`.
- E7: `uv run --frozen --no-sync python scripts/train_distill.py task=g1_walk_flat/mujoco training.fada.enabled=true --cfg job` -> successful compose with `obs_dim=98`, `action_dim=29`, `command_dim=3`, `H=30`, `K=6`, and the full FADA parameter family.
- E8: focused `ruff check` -> `All checks passed`; focused `pyright` -> `0 errors, 0 warnings, 0 informations`.
- E9: Architecture Atlas `npm ... run check` -> viewer/data contracts OK.
- E10: bounded real-owner sentinel stopped before environment creation with `FileNotFoundError: No SAC Oracle checkpoint resolved for FADA training`; repository and `/Users/sss9999/locomotion` search found no `2026-07-09_02-48-58_mujoco` or checkpoint 5000 artifact.
- E11: SSH read-only audit on `chengyuxuan@10.16.84.87` confirmed `/ssd1/cyx/UniLab/model/G1WalkFlat/model_5000.pt`, SHA-256 `db2f536f1391f7bdd92d22afe065170e32239834e398b685488bcb5ba5b63291`, SAC `actor` input 98 and action 29; GPU 0 is an idle RTX 4090 D with 24564 MiB.
- E12: downloaded Oracle hash matched E11. A real local G1WalkFlat MuJoCo two-iteration sentinel completed: iteration 0 `oracle`, iteration 1 `planner_idm`, one causal window per iteration, no rejected boundaries, final IDM loss `0.4862082303`, final Planner loss `0.4444436729`, paired checkpoint written successfully.
- E13: isolated remote repository `/ssd1/cyx/FADA_runs/20260804_planner_idm_v1/repo` was materialized from local base `5fff4b33` plus the six FADA owner/config files. Remote CUDA sentinel passed with Oracle then Planner-IDM rollout before the formal launch.
- E14: the initial formal run completed iterations 0-2, then failed closed during iteration 3 because the automatic `10275` step limit yielded only `42046/65536` valid windows under frequent Planner-IDM episode termination. Checkpoint remained at iteration 3 / 196,608 samples. Resume changed only `training.fada.max_env_steps` to `50000`; architecture, window count, loss, update count, and command contract were unchanged.
- E15: resumed remote campaign completed 8/8 iterations and 524,288 total windows. Final checkpoint `/ssd1/cyx/FADA_runs/20260804_planner_idm_v1/planner_idm.pt`, SHA-256 `5ebb8f79a58af2ab40ba1bf3da707fe79969290b65ec29bfac42ad4dc51aa398`, size 22 MB. Final recorded losses: IDM `0.0006427853`, Planner `0.0136708245`. Strict load restored both modules and optimizers; a zero-input probe produced finite Planner future `(2,6,98)` and first action `(2,29)`. Training process exited normally.
- E16: local artifact `/Users/sss9999/planner_idm.pt` matched E15 SHA-256 exactly. The inference-only loader reconstructed the checkpoint-owned architecture and produced finite `(1,29)` actions. Focused playback/model tests passed (`8 passed` plus `25 passed` across the playback CLI subsets), and focused Ruff checks passed.
- E17: a real local `G1WalkFlat` MuJoCo playback sentinel composed `--algo fada`, built one environment, loaded the final checkpoint, reset the FADA history owner, and executed one policy step. Observed `policy_obs_mode=actor`, action shape `(1,29)`, finite actions, and `step_count=1`.

## Confirmed facts

- `training.fada.enabled=false` preserves legacy workflow dispatch.
- ON routing executes Oracle bootstrap in iteration 0 and Planner-IDM rollout thereafter.
- IDM batches bind realized future to actually executed actions; Planner labels remain separate same-state Oracle first actions.
- Windows crossing episode termination or a future command change are rejected.
- All IDM updates precede all fixed-IDM Planner updates and each update draws a fresh replay mini-batch.
- Checkpoints atomically contain both modules, both optimizers, architecture, runtime config, cursor, and sample count.
- `--algo fada` is an isolated playback composition path. It defaults to policy actions, reads the complete 3-D command from `env.state.info['commands']`, executes only the first IDM action, and resets history rows at episode boundaries without changing generic `distill` routing.

## Open boundary

The formal source campaign used paper architecture `H=30`, `K=6`, hidden 128, 3-layer Planner,
3-layer IDM encoder, and 2-layer IDM decoder. Completion and finite losses prove training and
persistence, not closed-loop policy quality. Oracle-shadow augmentation and evaluation remain
outside the authorized construction boundary. Because replay is not checkpoint-persisted, the
resume retained module/optimizer/cursor state but rebuilt replay from iterations 3-7; this run is
not bitwise equivalent to an uninterrupted eight-iteration replay history.
