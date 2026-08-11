---
contract_id: FADA-METHOD-v003
status: superseded
effective_date: 2026-08-05
updated_date: 2026-08-05
supersedes: FADA-METHOD-v002
scope: paper-aligned Planner and inverse-dynamics construction through source training
---

# FADA Planner-IDM Method Contract

Superseded by `FADA-METHOD-v004`, which adds explicit standing and walk-to-stand command coverage
and dual-Oracle authority. The text below preserves the v003 semantic contract.

Authority: `FADA.pdf`, Sections 3, 4.1, Appendix B.1-B.2. Paper semantics take precedence.

## FADA-DP-CMD-01 / CMD-COVERAGE / Command Coverage

The task adapter owns the complete deployable command vector `c_t`. It includes every task-relevant
component and uses one declared task space for rollout collection, final-Oracle relabeling, and
Planner supervision. An admitted `K`-step trajectory or Oracle-shadow pair must retain the same
command; command-crossing windows are rejected.

## FADA-DP-PLAN-01 / PLANNER-INTERFACE / Planner Interface

- Learned object: `P_phi(O_t^H, c_t) -> Y_hat_t^K`.
- Input: deployable proprioceptive history and complete command, without privileged state.
- Output: `K=6` future proprioceptive observations reconstructed as residuals from `o_t`.
- Architecture: `H=30`, 3-layer, 4-head, hidden-128 Transformer encoder.
- Boundary: Planner owns command-to-intent and never directly outputs actions.

## FADA-DP-IDM-01 / IDM-INTERFACE / IDM Interface

- Learned object: `I_psi(O_t^H, A_t^H, Y_t^K) -> U_hat_t^K`.
- Input: `H=30` observation/action history and a `K=6` future chunk.
- Output: `K=6` parallel actions; deployment executes only the first.
- Architecture: 3-layer history encoder and 2-layer non-causal future decoder, 4 heads and hidden 128.
- Boundary: IDM owns intent-to-action realization and never interprets raw commands.

## FADA-DP-IDM-02 / CAUSAL-IDM-SUPERVISION / Causal IDM Supervision

The paper source buffer contains two physically causal future/action pairs for student visited states:

- trajectory source `(Y_traj^K, U_traj^K)` from the actually executed rollout;
- final-Oracle shadow source `(Y_orac^K, U_orac^K)` obtained by restoring the complete same-state
  simulator snapshot and rolling the final Oracle for `K` steps under the same command.

IDM training consumes both sources with Eq. 4.2 first-action supervision. A physics-only reset,
student future paired with an unrelated Oracle action, or same-state first-action query without the
full causal shadow chunk is forbidden.

To broaden inverse-dynamics coverage, source data also includes rollouts from exactly 20 intermediate
Oracle checkpoints. Their total data budget is twice the optimal-data budget. This suboptimal data
changes only IDM support coverage; final-Oracle labels remain the Planner authority.

## FADA-DP-PLAN-02 / ACTIONABLE-PLANNER-SUPERVISION / Actionable Planner Supervision

At each visited state, the final Oracle is rolled from the restored same-state snapshot under the same
command. Its first shadow action is `a_t*` for Eq. 4.3. Planner future passes through the fixed current
IDM; only Planner parameters update, and Planner future is not observation-regressed to the shadow
trajectory.

## FADA-DP-EXEC-01 / RECEDING-HORIZON / Receding-Horizon Execution

Planner and IDM return `K`-step chunks, but only `Pi_1(U_hat_t^K)` executes. Histories update and
both modules are queried again at the next control step. Source losses supervise the first action.

## Local implementation choices not fixed by the paper

The repository uses learned bounded positional embeddings, feed-forward width `4 * hidden_dim`,
GELU, zero dropout, and the last encoded history token for the Planner head.

## Required evidence

- Exact snapshot/restore equivalence including physics, observation/info, counters, RNG, and autoreset.
- Causal trajectory and final-Oracle-shadow provenance tests.
- Exactly 20 unique intermediate Oracle identities and 2:1 source-budget validation.
- Separate IDM/Planner gradient ownership and first-action semantics.
- Source quality report separating true-future IDM error from Planner-IDM Oracle-action error.
