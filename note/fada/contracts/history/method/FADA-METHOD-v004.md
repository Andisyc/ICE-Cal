---
contract_id: FADA-METHOD-v004
status: superseded
effective_date: 2026-08-05
updated_date: 2026-08-05
supersedes: FADA-METHOD-v003
scope: paper-aligned Planner and inverse-dynamics construction with standing and walk-to-stand source coverage
---

Superseded by `FADA-METHOD-v005`, which adds exact deployment cold-start windows, row-level source
identity, Planner-only scenario-balanced replay, and scenario-resolved quality evidence.

# FADA Planner-IDM Method Contract

Authority: `FADA.pdf`, Sections 3, 4.1, Appendix B.1-B.2. Paper semantics take precedence.
The standing curriculum below is the user-confirmed source-domain command-coverage specialization.

## FADA-DP-CMD-01 / CMD-COVERAGE / Command Coverage

The task adapter owns the complete deployable command vector `c_t`. It includes every task-relevant
component and uses one declared task space for rollout collection, Oracle relabeling, and Planner
supervision. An admitted `K`-step future trajectory or Oracle-shadow pair must retain the same
command; command-crossing future windows are rejected.

When `stand_transition_curriculum` is enabled, every optimal/current-policy collection budget is
partitioned into three explicit scenarios:

- `walk`: the existing active-command source distribution;
- `static_stand`: zero command from the dedicated `G1StandStill` owner environment, labeled and
  shadow-rolled by the standing Oracle;
- `walk_to_stand`: at least `H` active-command steps followed by an atomic zero-command switch in
  the `G1WalkFlat` owner environment.

For `walk_to_stand`, the accepted anchor has zero command, its `H`-step history retains active-command
states, and its full `K`-step future remains at zero command. This captures braking and balance after
walking without admitting a command-crossing future chunk. The walking Oracle owns pre-switch labels;
the standing Oracle owns static-standing and post-switch labels. The curriculum must never silently
use the walking Oracle as the standing authority.

Intermediate Oracle checkpoints remain walking-source support only unless a separately contracted
standing intermediate-Oracle set exists. Their combined 2:1 paper budget is unchanged.

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
- same-state Oracle shadow source `(Y_orac^K, U_orac^K)` obtained by restoring the complete same-state
  simulator snapshot and rolling the scenario-authoritative Oracle for `K` steps under the same command.

IDM training consumes both sources with Eq. 4.2 first-action supervision. A physics-only reset,
student future paired with an unrelated Oracle action, or same-state first-action query without the
full causal shadow chunk is forbidden.

To broaden inverse-dynamics coverage, source data also includes rollouts from exactly 20 intermediate
walking Oracle checkpoints. Their total data budget is twice the optimal/current-policy budget. This
suboptimal data changes only IDM support coverage; the scenario-authoritative final Oracle remains
the Planner label authority.

## FADA-DP-PLAN-02 / ACTIONABLE-PLANNER-SUPERVISION / Actionable Planner Supervision

At each visited state, the scenario-authoritative Oracle is rolled from the restored same-state
snapshot under the same command. Its first shadow action is `a_t*` for Eq. 4.3. Planner future passes
through the fixed current IDM; only Planner parameters update, and Planner future is not
observation-regressed to the shadow trajectory.

## FADA-DP-EXEC-01 / RECEDING-HORIZON / Receding-Horizon Execution

Planner and IDM return `K`-step chunks, but only `Pi_1(U_hat_t^K)` executes. Histories update and
both modules are queried again at the next control step. Source losses supervise the first action.

## Local implementation choices not fixed by the paper

The repository uses learned bounded positional embeddings, feed-forward width `4 * hidden_dim`,
GELU, zero dropout, and the last encoded history token for the Planner head.

## Required evidence

- Exact snapshot/restore equivalence including physics, observation/info, counters, RNG, and autoreset.
- Causal trajectory and Oracle-shadow provenance tests.
- OFF equivalence with the standing curriculum disabled.
- Exact scenario quotas, zero-command static-standing windows, and active-history-to-zero-future
  walk-to-stand windows.
- Dedicated `G1StandStill` reset/state distribution for static standing and `G1WalkFlat` ownership
  for walking-to-standing transitions.
- Walking/standing Oracle authority separation and missing-standing-checkpoint rejection.
- Exactly 20 unique intermediate walking Oracle identities and unchanged 2:1 source-budget validation.
- Separate IDM/Planner gradient ownership and first-action semantics.
- Source quality report separating true-future IDM error from Planner-IDM Oracle-action error.
