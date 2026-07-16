# Distillation Contract Registry

This registry is the only default contract entrypoint.

## Active

| Contract | Status | Scope | Supersedes |
| --- | --- | --- | --- |
| [DISTILL-METHOD-v001](active/method/DISTILL-METHOD-v001.md) | active | G1 multi-teacher command-intent MoE distillation | none |
| [DISTILL-TRAIN-v002](active/training/DISTILL-TRAIN-v002.md) | active | Single-entry multi-role workflow with transition-state evidence | DISTILL-TRAIN-v001 |

## History

| Contract | Status | Scope |
| --- | --- | --- |
| [DISTILL-TRAIN-v001](history/training/DISTILL-TRAIN-v001.md) | superseded | Single-entry, resumable multi-role training workflow |

## Recall Rule

Read only the active contract required by the task. Do not scan
`contracts/history/` unless an active contract cites a historical item or the
human explicitly requests historical comparison.

Draft method changes belong under `../plans/`, not in this registry.
