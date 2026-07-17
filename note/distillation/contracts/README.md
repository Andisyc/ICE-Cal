# Distillation Contract Registry

This registry is the only default contract entrypoint.

## Active

| Contract | Status | Scope | Supersedes |
| --- | --- | --- | --- |
| [DISTILL-METHOD-v001](active/method/DISTILL-METHOD-v001.md) | active | G1 multi-teacher command-intent MoE distillation | none |
| [DISTILL-TRAIN-v003](active/training/DISTILL-TRAIN-v003.md) | active | Integrated persistent DAgger runtime; promotion deferred; legacy default | DISTILL-TRAIN-v002 |

## History

| Contract | Status | Scope |
| --- | --- | --- |
| [DISTILL-TRAIN-v001](history/training/DISTILL-TRAIN-v001.md) | superseded | Single-entry, resumable multi-role training workflow |
| [DISTILL-TRAIN-v002](history/training/DISTILL-TRAIN-v002.md) | superseded | Transition-state scenario and schema contract |

## Recall Rule

Read only the active contract required by the task. Do not scan
`contracts/history/` unless an active contract cites a historical item or the
human explicitly requests historical comparison.

Draft method changes belong under `../plans/`, not in this registry.
