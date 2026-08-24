---
contract_id: FADA-CONTEXT-TRAIN-v008
status: active
effective_date: 2026-08-24
supersedes: FADA-CONTEXT-TRAIN-v007
method_contract: FADA-CONTEXT-METHOD-v009
scope: serial construction of data-driven correction components, coefficient readout, and scale evidence
implementation_status: engineering-proposal
---
# ICE-Cal Data-Driven Calibration Training Contract

## Transaction identity

Each dataset seals one ordered non-empty list of admitted component IDs and its catalog/provenance.
The width is `m=len(component_ids)` for every stage and artifact. The first authorized transaction is
straight-walk commands with one Gain component (`m=1`); joint DR and multi-component held-out evidence
are deferred, not silently claimed.

## Stage 0 / Basis Discovery

Freeze Planner and Tracker. Collect joint DR rollouts for the declared transaction and retain the
causal tuple. Solve one minimum-norm `delta_z_star` per valid row from zero, regress onto the declared
carrier library, and run SVD on the state-independent operator matrices. Record singular spectrum,
projection residuals, split identity, and component-to-`theta` correlation. No component enters later
stages until the admission record is complete.

## Stage 1 / Operator

Fit one typed operator per admitted component, with only that operator mutable. Multiplicative Gain
uses `Delta(z)=M z`; additive and shift forms are allowed only when the discovery/identification
record supports them. Freeze and publish only if the shared-state compensation ratio is at least `0.9`
and held-out action-family checks pass. No threshold relaxation is allowed.

## Stage 2 / Coefficient Encoder

Freeze all prior owners. The 2-layer `d_model=128` Transformer reads the same 30-frame State/Action
history and outputs `[B,m]` coefficients. The target is the least-squares projection of `delta_z_star`
onto the frozen operator bank. The loss is:

```text
MSE(c_hat, c_true) + 0.1 * MSE(action(c_hat), action_star)
```

The coefficient term owns semantics; the Action term is a safety net and gradients never update the
frozen Planner, Tracker, operators or scale curves. Admit only if validation `|c_hat-c_true| <= 0.05`.

## Stage 3 / Scale evidence

Train no network. Scan normalized `c in [-1,1]` at 21 points with 32 rollouts per point, fit a monotone
PCHIP scale curve per component, and require monotonicity plus `R^2 >= 0.95`. Preserve raw readings,
saturate at endpoints, and emit an explicit out-of-range event.

## Persistence and retirement

Every dataset, stage artifact and scale evidence binds Contract v009, component order, `H/K/D/m`,
source checkpoint, split and parent digests. A later stage requires a freshly loaded admitted parent.
v008/v007 analytic-target datasets and trained artifacts fail closed; only a separately validated raw
donor adapter may reseal data without migrating trained state.
