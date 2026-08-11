# Context Phase-1 Privileged Teacher Evidence

Date: 2026-08-10

Contracts: `FADA-CONTEXT-PHASE1-METHOD-v001`, `FADA-CONTEXT-PHASE1-TRAIN-v001`

## Implemented boundary

- G1 actuator strength remains disabled by default. The fixed-vector path used for the earlier
  left-knee playback probe remains compatible.
- The Phase-1 profile samples nominal, left-knee, and right-knee 29D `g` rows at reset and appends
  `g` only to the critic observation.
- `PrivilegedResidualSACActor` owns frozen nominal SAC inference, privileged residual inference,
  residual scaling, and final action clipping.
- `PrivilegedResidualSACLearner` reads `g` only from the final critic 29D, trains only the residual
  branch, and records the nominal checkpoint SHA-256 in the teacher checkpoint.
- Collector, standard playback, interactive playback, and ONNX export share the same privileged
  actor routing rule.

## Contract evidence

The selected nominal checkpoint was
`checkpoints/oracles/G1WalkFlat/model_5000.pt`, SHA-256
`db2f536f1391f7bdd92d22afe065170e32239834e398b685488bcb5ba5b63291`.

A real 64-environment MuJoCo reset produced actor/critic shapes `(64, 98)` and `(64, 130)`, with
`12` nominal, `25` left-knee, and `27` right-knee rows. The maximum absolute difference between the
runtime Kp pool and `base_kp * g` was `0.0`.

The production runner resolved to:

```text
DoubleBufferOffPolicyRunner
PrivilegedResidualSACLearner
PrivilegedResidualSACActor
obs=98, critic=130, g=29, action=29
```

The final bounded sentinel ran one environment for one learner iteration and four collected steps.
The collector remained alive, critic and actor updates completed with finite metrics, weight sync
completed, and the checkpoint was written to
`/private/tmp/fada-context-phase1-sentinel2-20260810/model_1.pt`. Its metadata uses schema
`unilab_privileged_residual_teacher_v1`, the nominal SHA above, and residual scale `0.2`.

## Verification

- Focused contract suite: `24 passed`.
- G1, residual SAC, off-policy runtime/worker, HORA, and interactive playback regression:
  `177 passed`.
- Existing script playback selection regression: `3 passed`.
- Ruff on all touched Python files: passed.
- Pyright on all touched runtime Python files: `0 errors, 0 warnings, 0 informations`.
- `git diff --check`: passed.

Gymnasium emitted its existing float-bound cast warnings in MuJoCo environment construction. They did
not affect the tested observation, gain, action, loss, or checkpoint finiteness contracts.

## Not established

This evidence does not show trajectory improvement, Context Encoder trainability, real motor-torque
fidelity, sim-to-real repair, or publication novelty. The intervention scales position-servo Kp/Kd;
it is a controlled actuator-effectiveness approximation, not a measured motor-strength model.
