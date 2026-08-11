# Planner–IDM Design Discussion

Status: proposal under human discussion

Scope: pretrained Oracle trajectory -> IDM distillation -> Planner distillation

Non-scope: target adaptation, LoRA, target deployment, evaluation, and later training stages

## FADA-DP-CMD-01 Command Coverage

### Confirmed intent

The command input must cover the complete task-intent space needed by the
Oracle. The purpose is to make the collected Oracle rollouts sufficiently
complete for both IDM state-to-action distillation and Planner
command-to-state distillation.

Usable rollout coverage must consider all of the following together:

- every task-relevant command dimension;
- the legal range and coverage resolution of each dimension;
- the sampling distribution over valid command combinations;
- temporal transitions and command sequences, not only static values;
- an observable coverage gate for the resulting rollout evidence.

Omitting any one of these aspects is not sufficient for calling the rollout
set complete or usable.

### Current semantic interpretation

- Command coverage is an upstream data-support decision, not only an input-format decision.
- A large number of commands is not sufficient by itself; the resulting rollouts must cover the task-relevant state-action regions.
- IDM and Planner must be trained from command-conditioned evidence drawn from the same declared task space.

### Open parameterization questions

1. What is the exact command schema for the selected task family?
2. What numerical range and resolution does each command dimension use?
3. What concrete mixture of static values, valid combinations, and transition sequences should be collected?
4. What quantitative or structural criterion proves that the resulting rollout set is sufficiently complete?

### Current risk

Independent uniform sampling can produce invalid or low-value combinations and
can still miss transition behavior. A small set of scripted command sequences
can cover transitions but may leave large parts of the command space unseen.
The accepted design therefore needs both a legal command domain and an explicit
coverage rule before rollout collection is considered complete.
