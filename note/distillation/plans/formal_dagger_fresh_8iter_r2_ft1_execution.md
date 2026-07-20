# Formal Fresh r2 FT-1 Execution Plan

Status: authorized for exactly one frozen supervisor invocation; awaiting its
live result.
Date: 2026-07-20.

## Objective

Execute exactly the authenticated Gate 0 identity
`20260720-140520_g1_walk_stand_formal_fresh_8iter_oom_r2` once, then invoke
its matching frozen postflight oracle only after supervisor success.

## White-Box Runtime Chain

| Node | Owner | Input -> output | Evidence before FT-1 |
| --- | --- | --- | --- |
| Frozen identity | Gate 0 materializer | r2 spec -> freeze, supervisor, oracle, absent outputs | runtime-confirmed E111 |
| One-shot launch | frozen supervisor | absence checks -> telemetry/time/log -> frozen train argv | code-confirmed; live unconfirmed |
| Training workflow | `train_distill.py` / workflow owner | fresh bootstrap -> 8 DAgger aggregates/checkpoints/metrics | live unconfirmed |
| Frozen acceptance | matching `.oracle.py` | freeze + manifest + metrics -> one acceptance JSON | code-confirmed; live unconfirmed |

## Exact Authorized Command

Run only from the authenticated server SSH session:

```bash
cd /ssd1/cyx/UniLab && bash 20260720-140520_g1_walk_stand_formal_fresh_8iter_oom_r2.supervisor.sh && uv run --no-sync 20260720-140520_g1_walk_stand_formal_fresh_8iter_oom_r2.oracle.py --freeze 20260720-140520_g1_walk_stand_formal_fresh_8iter_oom_r2.freeze.json --result 20260720-140520_g1_walk_stand_formal_fresh_8iter_oom_r2.acceptance.json
```

The `&&` is deliberate: a nonzero supervisor exit, including CUDA OOM,
prevents oracle execution. The frozen supervisor itself fails if any frozen
output path already exists, so it cannot resume or overwrite r2.

## Scope And Stop Condition

Scope: one supervisor execution, its frozen log/time/GPU telemetry, generated
run artifacts, and one matching frozen postflight oracle result.

Non-scope: retry, resume, rerun Gate 0, a second r2 identity, any source/config
change, batch adjustment, v2 replacement oracle, RT-10 physical evaluation,
promotion, or default-mode change.

Stop on the first nonzero exit. Preserve the existing r2 identity and report
the supervisor exit, tail of its frozen log, frozen `.time`, frozen NVIDIA CSV,
and any acceptance JSON if produced. Do not launch a second command.

## Acceptance

| Gate | Required result | Status |
| --- | --- | --- |
| FT-1 command | supervisor exits 0 exactly once | IN PROGRESS |
| FT-1 artifact path | fresh manifest, metrics, eight completed iterations, final checkpoint | IN PROGRESS |
| Frozen postflight | `.acceptance.json` has `accepted=true` | IN PROGRESS |
| OOM/cleanup evidence | no CUDA OOM; frozen log/time/telemetry exist | IN PROGRESS |
| RT-10 / promotion | separate, not implied | PENDING |
