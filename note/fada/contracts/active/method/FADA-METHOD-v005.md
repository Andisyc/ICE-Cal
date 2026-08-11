---
contract_id: FADA-METHOD-v005
status: active
effective_date: 2026-08-05
updated_date: 2026-08-05
supersedes: FADA-METHOD-v004
scope: paper-aligned Planner-IDM source training with exact cold-start and scenario-preserving replay
---

# FADA Planner-IDM Method Contract

Authority: `FADA.pdf`, Sections 3, 4.1, Appendix B.1-B.2. The network factorization and Eq. 4.2/4.3
semantics remain unchanged from v004. v005 changes source support, replay admission, and evidence.

## FADA-DP-CMD-01 / CMD-COVERAGE / Command Coverage

Optimal/current-policy windows remain `walk=0.50`, `static_stand=0.25`, and
`walk_to_stand=0.25`. Static standing remains owned by `G1StandStill`; walking and active-to-zero
transitions remain owned by `G1WalkFlat`. The standing Oracle owns static and post-switch labels.

Half of every static-standing allocation is an exact deployment cold-start window. Its Planner input
is the reset observation repeated `H=30` times, an all-zero executed-action history, and zero command.
Its realized and Oracle-shadow targets begin at reset time and contain `K=6` causal transitions. The
other half remains steady-state static-standing coverage. A later steady-state window cannot be used
as a substitute for cold start.

Every persisted row carries `command_scenario`, `planner_eligible`, and `cold_start` identity. These
fields are provenance only and never enter deployable Planner or IDM observations.

## FADA-DP-PLAN-01 / PLANNER-INTERFACE / Planner Interface

`P_phi(O_t^H, c_t) -> Y_hat_t^K` is unchanged: deployable proprioception history plus complete command
produce `K=6` residual future observations through the paper-aligned Transformer.

## FADA-DP-IDM-01 / IDM-INTERFACE / IDM Interface

`I_psi(O_t^H, A_t^H, Y_t^K) -> U_hat_t^K` is unchanged. Only the first predicted action executes.

## FADA-DP-IDM-02 / CAUSAL-IDM-SUPERVISION / Causal IDM Supervision

IDM replay consumes all valid realized and Oracle-shadow causal pairs, including the paper-required
20 intermediate walking Oracle sources at the 2:1 budget. Intermediate rows remain
`planner_eligible=false`; they expand inverse-dynamics support and never supervise Planner.

## FADA-DP-PLAN-02 / ACTIONABLE-PLANNER-SUPERVISION / Actionable Planner Supervision

Planner replay samples only `planner_eligible=true` rows and constructs every batch at the fixed
`50/25/25` scenario ratio. Within the static-standing share, `50%` is exact cold start and `50%` is
steady state. Sampling is with replacement inside each stratum and fails closed when any required
stratum is absent. Planner future still trains only through fixed-IDM first-action Oracle loss.

## FADA-DP-EXEC-01 / RECEDING-HORIZON / Receding-Horizon Execution

Deployment remains receding-horizon: predict `K`, execute the first action, update histories, replan.

## Quality contract

Checkpoint quality evidence must retain the aggregate source metrics and add per-scenario Planner-IDM
Oracle-action MSE, row fractions, and separate static cold-start/steady-state Planner MSE. Missing or
non-finite required strata fail before checkpoint acceptance. Closed-loop acceptance remains three
separate 3-seed evaluations for walking, static standing, and walk-to-stand.

## Required evidence

- Exact cold-start history/action/command and causal future construction.
- OFF equivalence when the v005 repair flag is disabled.
- Row identity survives artifact, replay, sampling, device transfer, and checkpoint metrics.
- Planner batches are exact `50/25/25`, static is exact `50/50`, and intermediate rows are excluded.
- IDM batches continue to admit intermediate Oracle rows.
- Per-scenario and cold-start metrics are serialized by the production checkpoint path.
