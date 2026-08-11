---
date: 2026-08-11
evidence_class: runtime-confirmed
contracts: FADA-CONTEXT-PHASE1-METHOD-v005, FADA-CONTEXT-PHASE1-TRAIN-v005
status: quality-failed
---

# Context Full-Action Teacher v005 Evaluation

## Training artifact

- Remote root: `/ssd1/cyx/FADA_runs/20260811_context_teacher_full_action_v005`
- Training completed `5000/5000` in 9m59s with `10,262,528` environment steps.
- Final checkpoint: `training/model_5000.pt`, 19,489,301 bytes.
- SHA-256: `24722004a22a318fafa867b9a4c05265c129ab3381a64f55a8521dbd26cc0e70`.
- Strict loader confirmed schema `unilab_privileged_full_action_teacher_v1`, dimensions
  `(obs=98, g=29, action=29)`, update count 5000, and original initialization SHA
  `db2f536f1391f7bdd92d22afe065170e32239834e398b685488bcb5ba5b63291`.

## Formal paired evaluation

The emitted report is
`/ssd1/cyx/FADA_runs/20260811_context_teacher_full_action_v005/evaluation/formal_paired.json`.
Protocol validation passed with exact same-snapshot pairing, seeds 101-105, 256 environments per
seed, 400 requested steps, command `(0.4, 0.0, 0.0)`, and fixed left-knee strength `0.9`.

| Metric | Original policy | v005 teacher |
|---|---:|---:|
| Maximum lateral displacement | 0.122964 m | 0.006020 m |
| Maximum yaw drift | 0.142355 rad | 0.010231 rad |
| Forward-velocity MAE | 0.075804 m/s | 0.376011 m/s |
| Forward progress | 2.580561 m | 0.028192 m |
| Mean survival | 400 steps | 50 steps |
| Failure rate | 0.0 | 1.0 |
| Action saturation step rate | 0.0 | 0.0 |

Lateral reduction `95.10%` and yaw reduction `92.81%` passed. Forward-velocity non-degradation and
failure-rate checks failed, so the conjunctive quality status is `failed`.

## Collapse localization

A bounded seed-101 sweep used 64 environments and 60 requested steps for every saved checkpoint:

| Iteration | Survival | Failure rate | Forward progress | Forward MAE |
|---:|---:|---:|---:|---:|
| 1000 | 50 | 1.0 | 0.027081 m | 0.371395 m/s |
| 2000 | 50 | 1.0 | 0.025863 m | 0.376699 m/s |
| 3000 | 50 | 1.0 | 0.020996 m | 0.381546 m/s |
| 4000 | 50 | 1.0 | 0.034868 m | 0.370171 m/s |
| 5000 | 50 | 1.0 | 0.028515 m | 0.375664 m/s |

The first saved checkpoint had already collapsed. The environment correctly detects insufficient
progress at step 50, but termination alone did not teach forward motion. This evidence does not
authorize another threshold or reward change; the next training mechanism is a human-owned method
decision. Context Encoder distillation remains blocked because no teacher checkpoint has passed the
paired quality gate.
