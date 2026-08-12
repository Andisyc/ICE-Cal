---
status: superseded-history
updated_date: 2026-08-11
contracts: FADA-CONTEXT-PHASE1-METHOD-v006, FADA-CONTEXT-PHASE1-TRAIN-v006
superseded_by: FADA-CONTEXT-METHOD-v003, FADA-CONTEXT-TRAIN-v002
---

# Context Phase-1 Privileged Teacher Plan

Steps 1-12 below are retained as completed history for the rejected residual-teacher path.

1. Extend the default-off G1 actuator-strength owner with per-reset nominal/left-knee/right-knee
   sampling and optional critic-only 29D observation exposure.
2. Add a custom off-policy actor that owns the frozen nominal SAC actor, trainable privileged
   residual actor, bounded additive fusion, and nominal checkpoint identity.
3. Add the learner/runtime, collector, playback, and export connectivity required by that actor.
4. Prove OFF compatibility, ON observation/randomization semantics, residual fusion, frozen gradient
   ownership, checkpoint round-trip, and Hydra routing.
5. Run focused offline tests and one bounded MuJoCo training sentinel. Do not start formal training or
   claim Context repair quality in this closure unit.
6. Add a separate same-snapshot paired evaluator that records straight-line, termination, residual,
   and clipping metrics by nominal/left-knee/right-knee stratum without imposing quality thresholds.

Steps 1-6 have bounded engineering evidence. Formal teacher training remains gated on a human-owned
launch decision; the numeric quality contract is accepted in v002, while the one-update checkpoint
and paired sentinel remain non-acceptance diagnostics.

7. Encode the accepted five-seed formal protocol and conjunctive per-stratum gate as testable owners.
8. Freeze the inherited G1 SAC hyperparameters in the Phase-1 task profile and run a no-training
   preflight that validates config, runtime, dimensions, and nominal checkpoint identity.
9. Stop before `runner.learn`; formal launch requires an explicit target device and log directory.

Steps 7-9 produced the completed v002 run on 2026-08-11. Formal evaluation rejected it because the
teacher improved velocity/yaw but worsened lateral displacement.

10. Replace the broad bilateral continuous distribution with balanced nominal `1.0` and fixed left-
    knee `0.9` strata; reject right-knee and non-`0.9` rows in formal evaluation.
11. Add default-off initial-frame lateral-displacement and yaw-drift penalties so the training
    objective directly owns the declared straight-line measurements.
12. Run focused tests and target CUDA preflight, then launch an isolated v003 formal retry. Stop at
    runtime-confirmed collector startup; checkpoint quality remains a separate later gate.

## v004 full-action retry

13. Add a privileged full-action SAC actor/learner that consumes the exact 29D strength tail and
    directly emits all 29 actions, with no nominal actor in its forward path.
14. Warm-start the full-action actor from the original SAC where interfaces match, while keeping all
    teacher parameters trainable and recording source-checkpoint identity.
15. Add a fixed-left-knee-0.9-only task profile, formal no-training preflight, checkpoint loader, and
    same-snapshot evaluator against the original actor under the same 0.9 physics.
16. Run focused contract tests and one bounded MuJoCo update sentinel. Stop before formal SSH
    training and hand the exact launch command to the human.

Steps 13-16 completed. The v004 formal run completed but failed quality: it reduced lateral/yaw
error by remaining nearly stationary and failed the forward-velocity gate.

## v005 forward-progress retry

17. Preserve the v004 full-action actor, fixed-left-knee-0.9 intervention, rewards, SAC learner, and
    paired quality gate; add only a default-off G1 forward-progress failure termination.
18. After 50 completed steps, terminate commanded-forward episodes whose reset-yaw-frame average
    forward speed is below `0.20 m/s`; exclude commands below `0.1 m/s`.
19. Calibrate the threshold against the original actor, prove the rejected stationary v004 teacher
    terminates at the exact boundary, and run local/remote regressions plus CUDA no-training preflight.
20. Launch an isolated formal v005 run from the original actor and stop monitoring after one
    collector-startup confirmation. Training completion and paired quality remain later gates.

Steps 17-19 are complete. The original actor's minimum step-50 average speed across seeds 101-105
was `0.226277 m/s`. In the remote 64-environment discriminator it survived 60/60 steps with zero
failures, while the rejected v004 stationary teacher terminated at exactly step 50 in every row.
Step 20 completed `5000/5000` with `10,262,528` environment steps. The formal paired gate failed:
the teacher survived exactly 50 steps in every evaluated row, reached only `0.0282 m` forward
progress, and had `100%` failure rate. Checkpoints 1000 through 5000 all showed the same step-50
failure in a bounded sweep, so the collapse occurred before the first saved checkpoint rather than
as late-training regression. No further retry is authorized until the training mechanism changes.

## v006 behavior-anchored retry

21. Preserve complete-action inference, fixed left-knee `0.9`, v005 physics/rewards/termination, and
    the unchanged paired quality gate.
22. Add a separately frozen original actor as a training-only action anchor; optimize only the
    privileged teacher with `L_SAC + 10 * MSE(a_teacher, stopgrad(a_nominal))`.
23. Reduce actor learning rate to `3e-5`, save every 100 iterations, and cap the first run at 1000
    iterations so collapse is screened before another full run completes.
24. Pass owner tests, one-environment MuJoCo update, full regression, and remote CUDA no-training
    preflight. Launch only after those gates; reject the run immediately if checkpoint 100 does not
    preserve forward survival/progress.

Steps 21-24 completed. Remote CUDA preflight passed and training completed `1000/1000`. The behavior
anchor repaired stationary collapse: model 100 walked for the full formal horizon and advanced
`2.8192 m`. Formal quality still failed because maximum lateral displacement and yaw drift were worse
than the original actor. No v006 checkpoint is eligible for Context Encoder supervision.
