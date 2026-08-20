---
contract_id: FADA-CONTEXT-TRAIN-v005
status: superseded
effective_date: 2026-08-14
supersedes: FADA-CONTEXT-TRAIN-v004
method_contract: FADA-CONTEXT-METHOD-v006
scope: query-conditioned delta_z training with complete-Support pair ownership
implementation_status: implemented-module-correct-formal-offline
superseded_by: FADA-CONTEXT-TRAIN-v006
superseded_date: 2026-08-19
---

# FADA Query-Conditioned Context Training Contract

This active training Contract retains complete Support ownership and conditions every Context forward
on the current Query histories. Module correctness, formal-route connectivity, and training authority
remain separate engineering and evidence gates.

## Dataset ownership

The dataset retains one complete Support trajectory per Support–Query pair. For every valid Query
time `t`, it derives the current `H=30` histories, following `K=6` realized future, and executed
Action. Pair and Query-sample axes remain explicit:

```text
Support Planner Intent  [P,S,K,O]
Support realized State  [P,S,O]
Support executed Action [P,S,A]
Query State History     [P,N,H,O]
Query Action History    [P,N,H,A]
Query realized Future   [P,N,K,O]
Query executed Action   [P,N,A]
Query time/mask          [P,N]
```

The implementation may expand a complete Support row across its owning Query rows for vectorized
execution, but it must not create or persist Support subsequences.

## Forward and loss contract

For every valid Query timestep, Context Encoder receives the complete owning Support and that
timestep's current histories. The frozen Tracker Encoder receives the Query histories and its training
Future Motion. Tracker Decoder emits six Actions. Only Action zero participates in the loss.

Pair weighting remains explicit: first average valid Query samples within each pair, then average
pairs. Padded samples and the five nonexecuted Action entries contribute neither value nor gradient.

## Gradient, mode, and persistence contract

- The optimizer owns exactly Context Encoder parameters.
- Planner and complete Tracker stay in evaluation mode and remain bitwise unchanged across every
  optimizer and checkpoint sentinel.
- Checkpoint schema must bind `FADA-CONTEXT-METHOD-v006`, Context architecture, `H`, `K`, Support
  length, dataset/split digests, and the healthy source checkpoint digest.
- Fixed-residual checkpoint schemas are incompatible by default. Any inspection-only legacy loader
  must remain outside the active training/evaluation/playback route.
- Existing fixed-residual training and evaluation evidence becomes historical on activation; it
  cannot establish correctness or learning quality for query-conditioned `delta_z_t`.

## Admission and stop conditions

Stop before optimizer construction on a Contract/schema mismatch, old Planner signature, missing
Query histories, Support/Query ownership mismatch, non-finite input, rollout overlap, or any frozen
parameter entering the optimizer. Stop training on non-finite loss/gradient, zero Context gradient,
or any frozen-parameter mutation.

Module Test Cards are confirmed, the implementation is `MODULE-CORRECT`, and the official offline
route has a current `formal-runtime-audit` receipt. Long training nevertheless remains forbidden
because human long-run authority was not requested. The current receipts do not establish simulator
behavior, convergence, policy quality, robustness, or deployment admission; no Context training or
live simulation was executed in this implementation unit. Exact evidence and identities are recorded
in `note/fada/evidence/2026-08-15-ica-dp08-query-conditioned-execution.md`.
