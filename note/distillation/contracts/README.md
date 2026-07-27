# Distillation Contract Registry

This registry is the only default contract entrypoint.

## Active

| Contract | Status | Scope | Supersedes |
| --- | --- | --- | --- |
| [DISTILL-METHOD-v003](active/method/DISTILL-METHOD-v003.md) | active | G1 ordered Walk -> nominal settle -> StandHeight tracking distillation | DISTILL-METHOD-v002 |
| [DISTILL-TRAIN-v003](active/training/DISTILL-TRAIN-v003.md) | active | Integrated persistent DAgger runtime; promotion deferred; legacy default | DISTILL-TRAIN-v002 |

## History

| Contract | Status | Scope |
| --- | --- | --- |
| [DISTILL-METHOD-v001](history/method/DISTILL-METHOD-v001.md) | superseded | G1 standing, walking, and future height-control multi-teacher distillation |
| [DISTILL-METHOD-v002](history/method/DISTILL-METHOD-v002.md) | superseded | G1 StandHeight and Walk two-teacher command-intent MoE distillation with atomic stop/height switching |
| [DISTILL-TRAIN-v001](history/training/DISTILL-TRAIN-v001.md) | superseded | Single-entry, resumable multi-role training workflow |
| [DISTILL-TRAIN-v002](history/training/DISTILL-TRAIN-v002.md) | superseded | Transition-state scenario and schema contract |

## Recall Rule

Read only the active contract required by the task. Do not scan
`contracts/history/` unless an active contract cites a historical item or the
human explicitly requests historical comparison.

Draft method changes belong under `../plans/`, not in this registry.
