# Paper-Exact FADA Source Repair Evidence

Date: 2026-08-05

Scope: FADA Appendix B.2 source-domain Planner-IDM construction only.

## Diagnosis

- E18: the prior schema-v1/v002 checkpoint strictly loaded as 98-D observation, 29-D action,
  3-D command, 8 iterations, and 524,288 samples. Real MuJoCo comparison measured true-future IDM
  action MSE `0.00283`, but Planner-IDM to Oracle action MSE `0.03252` and Planner future to realized
  future MSE about `0.536`; the robot's unstable playback is therefore consistent with failed
  Planner-to-IDM distillation rather than an inability to execute the checkpoint.
- E19: collector reading found the first invalid lifecycle boundary: `init_state()` was followed by
  a detached direct all-row `reset()`, so external observation/command data could differ from the
  internal `NpEnv._state` consumed by `step()`.
- E20: local extraction of `FADA.pdf`, Appendix B.2, requires same-state final-Oracle K-step shadow
  pairs and exactly 20 intermediate Oracle checkpoints with suboptimal data twice the optimal data.
  Local and remote artifact audit found only final `model_5000.pt`.

## Implemented Evidence

- E21: `NpEnv.reset_all()` now routes full reset through the authoritative state carrier.
- E22: the public backend/environment snapshot transaction restores MuJoCo physics, sensor cache,
  pending forces, `NpEnvState`, counters, scratch buffers, NumPy/Python RNG, autoreset, and G1
  curriculum/reward-scale extension state on success or exception.
- E23: every accepted visited state can produce a K-step final-Oracle shadow pair; terminated or
  command-crossing shadow rows are masked out. IDM loss uses realized and valid shadow causal rows.
- E24: the FADA training owner seals exactly 20 unique checkpoint identities, the literal paper
  count of 20, strict pre-environment checkpoint loading, and an exact 2:1 allocation. Intermediate
  rollout actions remain separate from final-Oracle Planner labels.
- E25: schema-v2 checkpoints persist trajectory-IDM, shadow-IDM, Planner-IDM, Planner-future,
  shadow-validity, termination-rejection, and command-rejection metrics.

## Verification

- E26: focused FADA/env/G1 suite: `126 passed`.
- E27: expanded FADA/backend suite: `158 passed, 11 skipped, 218 deselected`.
- E28: train-entry/playback/visualization suite excluding one known unrelated stale-message test:
  `245 passed, 8 skipped, 1 deselected`.
- E29: the excluded test expects `G1WalkFlat/G1StandStill`, while the current height route correctly
  fails earlier on its 99-D observed-height contract. This failure is outside the FADA path.
- E30: Ruff passed for all changed FADA/reset/backend/G1 owners; focused Pyright returned zero errors.
- E31: Architecture Atlas checks passed: viewer import, data contracts, runtime modules, method
  modules, and concept nodes.
- E32: formal CLI with the exact final Oracle fails before environment creation with
  `paper-exact FADA requires exactly 20 unique intermediate Oracle checkpoints, got 0`.
- E33: final real MuJoCo sentinel used 8 envs and collected 32 Oracle windows in 38 env steps with
  no rejected rows and shadow-valid fraction `1.0`. It saved
  `/private/tmp/fada_v003_sentinel_final.pt` with trajectory/shadow IDM MSE `0.1198243`,
  Planner-IDM Oracle-action MSE `0.1417196`, Planner-future realized MSE `0.4345777`, and both
  rejection counts `0.0`. These are fresh one-update connectivity metrics, not policy acceptance.

## Status

The v003 implementation and bounded real path are verified. Formal paper-exact training is blocked,
not completed, until 20 compatible intermediate Oracle checkpoints are recovered or produced.
