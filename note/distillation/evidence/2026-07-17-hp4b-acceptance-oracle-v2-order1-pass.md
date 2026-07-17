# HP-4b Acceptance Oracle v2 And Order 1 Pass

Date: 2026-07-17

Status: `PASS` for oracle repair and existing r7 order 1 acceptance. Orders 2-8
remain unexecuted.

## Scope

Repair the external experiment acceptance owner without changing UniLab source,
r7 execution identity, datasets, manifest, metrics, checkpoint, workload, or
training. Apply the frozen oracle to existing order-1 artifacts and stop before
order 2.

## Frozen oracle identity

- Oracle:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r7/hp4b_acceptance_oracle_v2.py`.
- Oracle SHA-256:
  `9e62b678eb02d792c587b2a46ecc7fae1e000b9376d5bfbc229683170fedb631`.
- Contract:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r7/acceptance_oracle_contract.json`.
- Contract SHA-256:
  `d46aefd4e089ad90423f45f69fa292b3bc60563a6923428945ee8781d6d2a6a2`.
- Bound execution identity SHA-256:
  `9b180b464433e0f29e59060c9245e9fbcd1879d988eeab802cee67be22f59718`.

The contract records order-1 hashes for manifest, metrics, aggregate,
checkpoint, and three scenario datasets before oracle execution.

## Corrected semantic boundaries

- Raw role artifact: validate hash, 128 rows, and the single expected role;
  require scenario/transition fields to remain absent. Scenario identity belongs
  to `WorkflowDatasetSource`.
- Raw transition artifact: validate native `walk_to_stop` scenario labels,
  both roles, transition ages, and before/after commands.
- Aggregate: require complete scenario/transition schema over 1024 rows and
  counts `walk_flat=384`, `static_stand=384`, `walk_to_stop=256`.
- Run: validate complete manifest, positive actual updates, checkpoint hash,
  21 request + 6 workflow + 1 cleanup records, route identity, and cleanup
  lifecycle. Persistent runs additionally require one weight version and
  resource-owner evidence.

## Existing order-1 acceptance

- Acceptance:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r7/execution_logs/order_01_acceptance.json`.
- Acceptance SHA-256:
  `5096ac20a558af0b07fea57686ce4d2ca525a1e2dd5ba1ec50298ba2dc55adcf`.
- Result: `accepted=true`, oracle schema v2.
- Formal run: legacy repetition 1, 1024 aggregate rows, 16 actual updates,
  complete checkpoint/28-record metrics/cleanup.

Attestation:

`/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r7/execution_logs/order_01_acceptance_attestation.json`.

It proves:

- oracle and contract hashes match the frozen values;
- all seven tracked training artifacts are byte-identical before/after;
- `training_rerun=false`;
- order 2 output/log/acceptance remain absent.

## Decision

The acceptance-oracle owner repair passes, and the existing r7 order 1 is now
formally accepted without rerunning training. Gate 0B's execution-plus-oracle
identity chain is complete for resumption. HP-4b remains partial because orders
2-8 have not run; no route comparison or speedup claim is accepted.

Control returns before order 2. A new human authorization is required to resume
r7 at order 2 using oracle v2 after each run; order 1 must not be rerun.
