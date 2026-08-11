# UniLab Planner-IDM Training Integration Plan

Status: completed on 2026-08-04; implementation, formal connectivity, and bounded real MuJoCo DAgger sentinel pass

Terminal outcome: UniLab can bootstrap, iteratively collect, train, checkpoint, and resume the
paper-defined IDM and Planner through one default-off FADA route.

Scope: FADA trajectory-window data, bounded replay, alternating trainer, checkpoint contract,
live UniLab collector, `train_distill.py` composition, config, focused OFF/ON tests, governance,
and Architecture projections.

Non-scope: oracle-shadow future augmentation, target adaptation, LoRA, deployment, evaluation,
or a long GPU training campaign.

## Old chain and feature boundary

| Concern | Existing owner | FADA target owner | Isolation rule |
|---|---|---|---|
| Entrypoint routing | `scripts/train_distill.py:main` | FADA branch before legacy routing | only when `training.fada.enabled=true` |
| Environment creation | `create_env` via `BackendAdapter` | same public UniLab owners | no private backend access |
| Oracle inference | `LoadedTeacherPolicy` | reused unchanged | frozen and queried at visited state |
| Generic collector | flat single-step dataset | FADA causal window collector | separate dataset type, no schema overloading |
| Generic trainer | MLP/MoE behavior loss | ordered IDM then Planner passes | separate optimizers and gradient owners |
| Generic checkpoint | one student model | FADA paired module/optimizer checkpoint | separate schema and loader |

## Parameter inventory

| Owner field | Purpose | Required when ON | OFF behavior |
|---|---|---:|---|
| `training.fada.enabled` | route owner | yes | `false`, legacy route unchanged |
| `history_length`, `prediction_horizon` | paper windows | yes | ignored |
| architecture fields | Planner/IDM dimensions | yes | ignored |
| `iterations`, `windows_per_iteration`, `num_envs` | DAgger collection | yes | ignored |
| `idm_updates`, `planner_updates`, `batch_size` | ordered optimization | yes | ignored |
| `replay_capacity`, `checkpoint_path`, `resume_path` | persistence | yes | ignored |
| observation/command projection fields | task adapter | yes | ignored |

## Step 1 / 1

Objective: add the complete default-off UniLab FADA training path and close its deterministic
implementation and formal-entrypoint connectivity evidence.

Owner files/modules:

- `src/unilab/algos/torch/distill/fada_training.py`
- `src/unilab/algos/torch/distill/fada_collector.py`
- `src/unilab/algos/torch/distill/__init__.py`
- `scripts/train_distill.py`
- `conf/distill/config.yaml`
- focused FADA tests and FADA governance/Architecture files

Expected evidence: module tests for causal windows, command/episode rejection, alternating gradient
ownership, checkpoint round-trip; OFF routing regression; ON entrypoint connectivity with a fake
UniLab environment; lint/type/import checks; bounded real-owner config compose smoke.

Stop condition: all deterministic and formal-route checks pass, or the first real simulator or
checkpoint dependency is recorded without claiming a completed live campaign.

Why one step: all changes are reversible local code/config/document edits under one already
confirmed training objective and one default-off route boundary.
