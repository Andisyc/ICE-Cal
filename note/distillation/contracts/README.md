# Distillation Contract Registry

This registry is the only default contract entrypoint.

## Active

| Contract | Status | Scope | Supersedes |
| --- | --- | --- | --- |
| [DISTILL-METHOD-v001](active/method/DISTILL-METHOD-v001.md) | active | G1 multi-teacher command-intent MoE distillation | none |
| [DISTILL-TRAIN-v001](active/training/DISTILL-TRAIN-v001.md) | active | Single-entry, resumable multi-role training workflow | none |

## Recall Rule

Read only the active contract required by the task. Do not scan
`contracts/history/` unless an active contract cites a historical item or the
human explicitly requests historical comparison.

Draft method changes belong under `../plans/`, not in this registry.
