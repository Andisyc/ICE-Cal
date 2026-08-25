---
contract_id: FADA-METHOD-v009
status: superseded
effective_date: 2026-08-24
superseded_by: FADA-METHOD-v010
scope: paper-exact Planner-IDM source-role routing with explicit schema-4 provenance
---

# FADA Planner-IDM Method Contract v009

Authority: FADA Appendix B.2 and user-confirmed solution A on 2026-08-24.

The deployable interface remains exactly `66/29/3`, `H=30`, `K=6`; Planner receives observation
history plus command, IDM receives observation/action history plus a future-state chunk, and only
the first action executes. Planner loss, IDM architecture, optimizer order, replay quotas, and
persistent-async lifecycle are unchanged.

## IDM source-role contract

Every persisted source row carries `idm_source_role` with one of:

- `trajectory`: realized rollout future/action pair only;
- `oracle_shadow`: valid final-Oracle shadow future/action pair only.

Final-Oracle bootstrap rows are `oracle_shadow`. Planner-IDM and intermediate-Oracle rollout rows
are `trajectory`. One row contributes at most one IDM first-action loss term; invalid shadow rows are
not admissible IDM terms. Unknown or missing role identity fails before replay mutation.

`planner_eligible` remains Planner replay admission and `oracle_shadow_valid` remains shadow validity;
neither may reconstruct IDM source role.

## Persistence

Source artifacts use schema 4 and require the explicit role field. Schema 2/3 source artifacts are
incompatible and rejected before batch construction/replay. Checkpoint schema remains 3. v007 and
v007r1 student checkpoints/artifacts are immutable negative evidence and cannot be resumed or used
as initial weights for the v009 campaign.

## Evidence boundary

FADA module tests and the official offline persistent pseudo-transaction must pass before training
readiness. This contract does not claim convergence, closed-loop policy quality, simulator quality,
or authorize a training launch.
