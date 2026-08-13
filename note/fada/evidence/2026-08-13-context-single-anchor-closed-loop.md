---
status: negative-runtime-evidence
date: 2026-08-13
method_contract: FADA-CONTEXT-METHOD-v004
training_contract: FADA-CONTEXT-TRAIN-v003
checkpoint_sha256: 35049f87fc39e73a56df6e7d1f06bf30b2c8028a7d9e0b66704a1a2b1b5ab18b
---

# Single-Anchor Context Closed-Loop Result

This evidence closes the implemented single-anchor Query route. Both evaluations used fixed
left-knee strength `0.7`, command `[0.4, 0, 0]`, seeds `101/102/103`, `32` environments per seed,
and `200` evaluated steps. Healthy, fault-zero, and fault-Context branches shared exact paired
initial states. No branch fell or truncated.

## Stored validation Support

Artifact:
`artifacts/fada_context/support_query_left_knee_070_run_002/closed_loop_healthy_reference_context_500.json`

SHA-256: `af0492534fc81d47de7318e4b2cbec7eff0982c426cf0c31dfd5b5d76bc360c6`

Context worsened all `7/7` healthy-reference distance metrics. Actor-observation MSE changed from
`0.0011199288` to `0.0014466355` (`29.17%` more error). Base-position, yaw, local-velocity,
joint-position, joint-velocity, and action MSE also worsened.

## Online no-Context Support

Artifact:
`artifacts/fada_context/support_query_left_knee_070_run_002/closed_loop_online_support_context_500.json`

SHA-256: `0f08ed16196b4d88d8706d926b1c687b44408c5526f89ef0c592776370d22751`

The same fault environment first collected a fresh `60`-step no-Context Support, encoded it once,
restored the evaluation snapshot, and reused the resulting fixed `delta_z` in the repaired rollout.
Context again worsened all `7/7` metrics. Actor-observation MSE changed from `0.0011344016` to
`0.0014350880` (`26.51%` more error).

## Conclusion and limitation

The negative result rules out stale stored Support as the sole explanation for failure. It does not
prove that increasing Query temporal coverage will improve or fail. The accepted v005/v004
multi-window route changes only that coverage and must be evaluated separately. Its fault-action
labels still do not guarantee healthy-trajectory repair.
