---
contract_id: FADA-CONTEXT-METHOD-v006
status: proposal
proposed_date: 2026-08-14
supersedes_on_activation: FADA-CONTEXT-METHOD-v005
prerequisite: FADA-METHOD-v005
design_point: ICA-DP-08
scope: query-conditioned per-control-cycle latent calibration from complete Support
implementation_status: not-implemented
---

# FADA Query-Conditioned Context Method Contract Proposal

This proposal records the human-confirmed `ICA-DP-08 Frozen Query Execution` semantics. It is not
active engineering authority. Until Contract activation, Module Test Card confirmation, and plan
review are complete, the current implementation remains the
fixed-`delta_z` lineage described by `FADA-CONTEXT-METHOD-v005`.

## Accepted deployment lifecycle

1. The first rollout does not invoke Context Encoder. It only collects one complete Support
   trajectory.
2. Complete Support contains every recorded Planner Intent, realized State, and executed Action from
   that rollout. Support remains one complete trajectory and is never subdivided.
3. Planner retains the existing `Planner(State History, Command) -> Intent` interface. Both rollouts
   use the same command, so the task objective is unchanged; the Planner Intent is recomputed and may
   vary with the current State History.
4. During every control cycle of the second rollout:
   - Tracker Encoder consumes the current `H=30` State History, current `H=30` Action History, and
     Planner Intent to produce `z_t`;
   - Context Encoder consumes the complete first-rollout Support plus the current second-rollout
     State/Action histories to produce one `delta_z_t`;
   - Tracker Decoder consumes `z_t + delta_z_t`, predicts a `K=6` Action chunk, and only its first
     Action is executed.
5. State/Action histories are updated after execution and all three frozen modules run inference
   again. No online parameter update is permitted.

## Public tensor contract

```text
command                               [B,C]
Support Planner Intent                [B,S,K,O]
Support realized State                [B,S,O]
Support executed Action               [B,S,A]
current second-rollout State History  [B,H,O]
current second-rollout Action History [B,H,A]
current deployment Planner Intent     [B,K,O]
Tracker latent z_t                    [B,K,D]
Context residual delta_z_t            [B,D]
decoded Action chunk                  [B,K,A]
executed Action                       [B,A] = Action chunk[:,0]
```

The Context residual is broadcast only across the `K` latent tokens of the current control cycle.
It is not reused as one condition-level residual across the complete second rollout.

## Calibration Learning projection

During Calibration Learning, one complete Support is reused across multiple valid Query timesteps.
Each timestep supplies its current State/Action histories and therefore receives its own
`delta_z_(p,t)`:

```text
delta_z_(p,t) = ContextEncoder(Support_p, StateHistory_(p,t), ActionHistory_(p,t))
z_(p,t) = FrozenTrackerEncoder(StateHistory_(p,t), ActionHistory_(p,t), Y_realized_(p,t))
A_hat_(p,t) = FrozenTrackerDecoder(z_(p,t) + delta_z_(p,t))
L_context = pair_mean(sample_mean(MSE(A_hat_(p,t)[0], A_executed_(p,t))))
```

Support is shared as one complete trajectory by pair ownership.

## Ownership and forbidden behavior

- Planner owns Intent and retains its existing State History plus command input contract.
- Tracker Encoder owns `z_t`; Context Encoder may only produce `delta_z_t` in the same latent space.
- Tracker Decoder owns Action generation. Context Encoder must not emit Action directly.
- Planner, Tracker Encoder, and Tracker Decoder remain frozen in Calibration Learning and deployment.
- Only Context Encoder parameters may enter the Calibration Learning optimizer.
- Loss supervises only the first decoded Action. The remaining five predicted Actions are neither
  executed nor supervised in that control cycle.
- Do not aggregate several Support residuals, subdivide Support, or silently load a fixed-`delta_z`
  checkpoint into this route.

## Semantic migration table

| Semantic object | Proposed owner | Current legacy path | Retirement/isolation rule |
|---|---|---|---|
| Context residual | `FADASupportContextEncoder.forward(Support, StateHistory, ActionHistory)` | One Support-only `[B,D]` residual | Fixed-residual checkpoints remain historical and fail closed on the new schema |
| Query calibration | `FrozenIDMSupportQueryPolicy` | Broadcast one residual across Query samples/rollout | Each Query timestep/control cycle invokes Context with its own histories |
| Deployment | evaluator/playback policy wrapper | Precompute one residual before rollout | Complete Support is retained; `delta_z_t` is recomputed inside every policy forward |

## Activation and evidence gates

- Keep the synchronized Concept Figure and Design Inspector hash-bound in the governance manifest.
- Confirm Module Test Cards for Planner, Context Encoder, policy orchestration, and first-action loss.
- Obtain a `code-review-expert: READY` plan-review receipt.
- Demonstrate RED evidence for both rejected new public interfaces on the current checkout.
- Pass owner and consumer tests after implementation; old fixed-residual routes must be explicitly
  isolated or rejected.
- Do not start training, simulation, or policy-quality evaluation before later formal admission.
