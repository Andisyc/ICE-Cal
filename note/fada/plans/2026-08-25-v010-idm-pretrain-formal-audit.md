# FADA v010 IDM-pretrain formal runtime audit

Status: proposed formal-audit plan for the current `main@98cc802f` production
route. Scope is only the first long-training candidate, `idm_pretrain`; Planner
training remains a later unit bound to a real completed IDM checkpoint.

## Authority and claim boundary

- Active authority: `FADA-METHOD-v010` and `FADA-TRAIN-v010`.
- Current module receipt:
  `2026-08-25-v010-idm-pretrain-current-module-test.json` (`ADMITTED-OFFLINE`).
- Production entrypoint: `scripts/train_distill.py:main`, FADA branch,
  `persistent_async` composition root.
- Allowed: local test/evidence writes and deterministic external-adapter fakes.
- Forbidden: production semantic changes, simulator execution, server mutation,
  long training, Planner training, commit, and push.
- PASS may establish official offline composition and persistence for a bounded
  IDM-pretrain transaction. It cannot establish server asset availability,
  simulator behavior, convergence, walking quality, or Planner readiness.

## Critical design-point matrix

| Design point | Production owners | Minimum complete witness | Falsifier |
|---|---|---|---|
| DP-TS-01 one phase per invocation | phase resolver, workflow, trainer | result and checkpoint identify `idm_pretrain`; IDM loss is present and Planner loss is absent | any Planner update/loss or mixed optimizer owner |
| DP-TS-02 Oracle-only IDM source | source plan, persistent runtime, replay | final Oracle, standing Oracle, and all 20 intermediate Oracle identities reach the runtime; three collections activate | student rollout, missing Oracle identity, or zero source activation |
| DP-TS-03/05 freeze boundary | trainer and checkpoint | module receipt proves phase-specific gradient ownership; official result carries only the IDM update effect | mixed update ownership on the official route |
| DP-TS-04 Planner source | not in this audit | deferred until a real completed IDM checkpoint exists | any claim that this audit admits Planner training |
| DP-TS-06 persistence | persistent workflow and schema-4 checkpoint owner | completed phase, `optimizer_owner=idm`, one optimizer payload, strict reload, first action consumer, no temp residue | incomplete/mixed checkpoint, reload mismatch, missing first consumer, or residue |
| Persistent replay/lifecycle | runtime, collection IO, replay, workflow | three iterations, historical replay consumption, expected role counts, close exactly once | zero replay activation, quota mismatch, leak, or duplicate close |

## Global Simplified Formal Test

Reuse `tests/algos/test_fada_formal_runtime.py` as an independent test driver.
It calls the unchanged production `train_distill.main` composition root with the
active FADA config branch. Only two external boundaries are replaced:

1. Oracle checkpoint loading returns a deterministic external policy stand-in;
2. the simulator-backed persistent collector returns deterministic, contract-valid
   source artifacts through the production collection/replay admission path.

All semantic owners remain production owners: phase resolution, source plan,
replay, trainer loss/update, checkpoint writer/reader, and the first policy
consumer. Cost is reduced to one environment, three iterations, twelve windows
per main collection, one IDM update per iteration, and fixed external inputs.
Three iterations are retained so replay state is created and consumed later.

The test will add only missing observations of existing public result/checkpoint
fields: phase identity, mutually exclusive losses, phase completion, optimizer
owner, and single optimizer payload. No production hook or alternate algorithm
is introduced.

## R1/R2 edge receipts

- EDGE-01 config/phase → workflow: exact `idm_pretrain` branch.
- EDGE-02 Oracle source plan → persistent collector: final, standing, and 20
  intermediate checkpoint identities.
- EDGE-03 collected artifacts → replay → trainer: role/scenario counts and three
  active update iterations.
- EDGE-04 trainer → checkpoint: IDM-only loss/update and schema-4 phase identity.
- EDGE-05 checkpoint → strict reload → first action consumer: finite `[1,29]`
  action and no temporary residue.
- EDGE-06 lifecycle: runtime closes exactly once on success.

## Stop conditions

Stop on the first branch mismatch, missing source identity, mixed optimizer/loss,
replay non-activation, schema/persistence mismatch, cleanup failure, or test-only
semantic owner. If the offline transaction passes, server path and asset
resolution remains one explicit external-state question; it is not silently
promoted into training readiness.
