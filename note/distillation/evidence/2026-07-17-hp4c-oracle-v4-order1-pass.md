# HP-4c Oracle v4 And Existing Order 1 Pass

Date: 2026-07-17

Status: `PASS` for the acceptance-owner repair and existing r9 legacy order-1
acceptance. Persistent order 2 remains unexecuted and unauthorized.

## Scope

Freeze a versioned oracle v4/amendment, apply it to the existing completed
order-1 artifacts without invoking training, prove every training artifact is
byte-identical before/after acceptance, and stop before order 2.

## Frozen acceptance identity

- r9 execution identity SHA-256:
  `894e1d30f424a4e8329fabcf2d011bbe6eced9e82351a7c08e6f70f18ba183f7`.
- Superseded oracle v3 SHA-256:
  `44175d63524f90ab75017b04f7700f4d77d5b2334e01343bbd3b928e6fa8d821`.
- Superseded v3 contract SHA-256:
  `49256b4d0719996c4e41733840ede5c85d7af61edfdcb74221c7c7266127a700`.
- Oracle v4:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4c_discriminator_20260717_r9/hp4c_iteration_oracle_v4.py`.
- Oracle v4 SHA-256:
  `9acbbef280203ad1b3fce686a01dbea9aeb77462cc6a765d50405f75d09683a0`.
- Amendment:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4c_discriminator_20260717_r9/iteration_oracle_v4_amendment.json`.
- Amendment SHA-256:
  `3d58eb5d7558fa179e9de5f16deeaf08374ac5f59bc805ca37c588522b3a4c7c`.

The amendment binds v4 to the immutable r9 identity and records that order 2
is not authorized. Oracle v3, its contract, and the r9 identity were not edited.

## Corrected owner contracts

- Legacy iterations and scenario artifacts omit the persistent-only
  `input_weight_version` field. Timing identity retains null. Oracle v4 accepts
  absent optional fields with `.get(...)` while still requiring persistent
  versions when that route is used.
- `training.workflow.dagger_updates_per_iteration=8` is a configured floor.
  `auto_expand_replay_budget` raises actual updates to satisfy the frozen eight
  transition replay passes. For cumulative aggregates `1024 -> 1408`, the
  accepted actual counts are `16 -> 24`.

These are acceptance-owner corrections only; UniLab source/config/runtime
semantics did not change.

## Existing order-1 acceptance

- Acceptance:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4c_discriminator_20260717_r9/execution_logs/order_01_acceptance.json`.
- Acceptance SHA-256:
  `166623c8e047e1ed33ec4ca1abdf0be665ca2bac5fd0acb1c71fe98029240a9f`.
- Result: `accepted=true`, oracle schema v4, two iterations, `55` timing
  records, aggregate rows `1024 -> 1408`, updates `16 -> 24`, complete
  checkpoint lineage, role/transition schemas, and legacy cleanup.

## Immutability attestation

- Before snapshot SHA-256:
  `048923de1b81d7ec543967a611c20b358c1bdae026b8192b7372ddd3400441c2`.
- After snapshot SHA-256:
  `048923de1b81d7ec543967a611c20b358c1bdae026b8192b7372ddd3400441c2`.
- Coverage: every one of the `16` files under the legacy order-1 training
  directory, recursively, with relative path, size, and SHA-256.
- Attestation:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4c_discriminator_20260717_r9/order_01_acceptance_v4_attestation.json`.
- Attestation SHA-256:
  `b10549970e274c200e1986d55bbd4147c3b4db82da2b1dc38df8fed2fc28bd3b`.
- Facts: `training_rerun=false`; all 16 training files byte-identical;
  persistent output/log/acceptance absent.

## Decision

The oracle v4 owner repair passes and existing legacy order 1 is formally
accepted without rerunning training. The two-route discriminator remains
partial because persistent order 2 has not run. Architecture and UniLab source
do not require refresh because ownership/runtime routing did not change.

Control returns before persistent order 2. Resuming order 2 with the frozen v4
amendment requires separate authorization; legacy order 1 must not be rerun.
