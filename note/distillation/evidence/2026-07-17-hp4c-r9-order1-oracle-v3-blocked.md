# HP-4c r9 Order 1 Oracle v3 Blocked

Date: 2026-07-17

Status: `BLOCKED` at order-1 acceptance. The legacy two-iteration training
command exited zero; persistent order 2 did not start. No training rerun or
formal acceptance was performed after the first oracle failure.

## Frozen identity

- Identity:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4c_discriminator_20260717_r9/discriminator_identity_manifest.json`.
- Identity SHA-256:
  `894e1d30f424a4e8329fabcf2d011bbe6eced9e82351a7c08e6f70f18ba183f7`.
- Oracle v3 SHA-256:
  `44175d63524f90ab75017b04f7700f4d77d5b2334e01343bbd3b928e6fa8d821`.
- Oracle contract SHA-256:
  `49256b4d0719996c4e41733840ede5c85d7af61edfdcb74221c7c7266127a700`.
- Source bundle SHA-256 remains
  `ea1d4f7a6acc3a35f9669bbc55c3df681e48100bdbe72e0880def609f5d5b25e`.

The exact no-training preflight passed source, asset, MuJoCo, Hydra compose,
98-D teacher, nested frozen-source import/help, and normalized 171-entry
dependency-provider identity. The initial provider comparison failure was only
`uv pip freeze` line ordering; package names and versions were identical.

## Order 1 runtime facts

- Train log:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4c_discriminator_20260717_r9/execution_logs/order_01_legacy.log`.
- Subprocess elapsed: `2.971892750 s`.
- Manifest stage: `DAGGER_ITERATION_2_COMPLETE`; completed iterations: `2`.
- Metrics: `55` records = two times `(21 request + 6 workflow)` plus one
  final cleanup record.
- Aggregate rows: iteration 1 `1024`; iteration 2 `1408`.
- Checkpoint lineage:
  `96aaecf...ae35 -> d0bd6dca...ae93 -> b4d44cd5...e6b0`.
- Actual updates: iteration 1 `16`; iteration 2 `24`.
- Legacy cleanup: complete, per-request scope, `0.0 s` composite final cleanup.

## Oracle-owner defect

Oracle v3 stopped with `KeyError: input_weight_version` at the legacy branch.
The frozen source proves this field is optional by contract: `workflow.py`
adds `collection_execution_mode` and `input_weight_version` only when the
persistent collector returns a non-null version. Legacy iteration and scenario
records therefore omit the key; their timing identities correctly carry null.

The same audit found a second incorrect oracle assumption before formal retry:
the contract froze `updates_per_iteration=16`. `scripts/train_distill.py`
passes the configured eight updates as a minimum and enables
`auto_expand_replay_budget`; the offline owner raises the effective budget to
satisfy eight transition replay passes. With cumulative aggregates of 1024 and
1408 rows, the observed valid counts are `16` and `24`.

A diagnostic-only oracle copy under `/private/tmp` changed only these two
acceptance assumptions: optional legacy version keys use `.get(...)`, and
actual update counts are `[16, 24]`. It accepted all existing order-1 semantic,
lineage, aggregate, timing, and cleanup artifacts. This is diagnostic evidence,
not formal acceptance; the frozen v3 oracle, r9 identity, and order-1 artifacts
remain unchanged.

## Stop boundary

Persistent order 2, its output directory, its raw log, and both formal
acceptance files remain absent. Do not rerun legacy order 1 and do not mutate
oracle v3 in place.

The smallest next step requires separate authorization: freeze a versioned
oracle v4 plus an acceptance amendment bound to the immutable r9 identity;
record the corrected optional-field and replay-budget contracts; apply it to
existing order 1 without rerunning training; attest artifact hashes unchanged;
then return control before persistent order 2.
