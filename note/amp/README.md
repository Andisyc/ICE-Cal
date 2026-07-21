# AMP-Only Async Walking Control Room

Status: `Stage 1 contract proposal`; no active AMP contract or AMP Concept Figure exists yet.

This directory governs the proposed migration of walk-only AMP training from
`/Users/chengyuxuan/ArtiIntComVis/AMP_mjlab` into UniLab's asynchronous APPO
runtime. It is separate from `note/distillation/`; AMP is not a distillation
role or a hidden extension of the active multi-teacher method.

## Confirmed Human Scope

- migrate human-like **walking** AMP only;
- use UniLab's asynchronous collector/learner architecture;
- exclude running, fall recovery, delayed termination, recovery reset, and
  motion-reset curriculum;
- exclude explicit gait-phase observation, gait-phase reward, contact schedule,
  and other gait-controller ownership;
- retain only the minimum task reward needed to specify commanded locomotion and
  physical viability;
- train a deployable actor; the discriminator remains training-only.

## Default Read Path

1. [Contract and migration proposal](plans/amp_async_walk_migration_proposal.md)
2. [Static migration evidence](evidence/2026-07-21-static-migration-baseline.md)
3. [Current task canvas](task_canvas.md)

## Governance State

The active UniLab Concept Figure is currently the G1 multi-teacher distillation
method. It contains no AMP design point. This is a `figure-mismatch`, not a
reason to reuse distillation IDs or contracts.

After the human confirms the proposed AMP semantics, the next governed actions
are:

1. create an active `AMP-WALK` method/training contract;
2. add a separate AMP Concept Figure and Design Point Register;
3. refresh the repository/runtime Architecture with planned owner boundaries;
4. promote the provisional step map into the current engineering plan and
   checklist;
5. only then begin implementation.

No speedup or policy-quality claim is active. The target of reaching a useful
walking policy within roughly 10-20 minutes is a live acceptance hypothesis.

