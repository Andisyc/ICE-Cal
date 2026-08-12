---
contract_id: FADA-CONTEXT-TRAIN-v001
status: superseded
effective_date: 2026-08-11
updated_date: 2026-08-11
method_contract: FADA-CONTEXT-METHOD-v002
scope: direct supervised Context Encoder training from tracking-expert complete-action labels
implementation_status: design-only
superseded_by: FADA-CONTEXT-TRAIN-v002
---

# FADA Context Direct-Supervision Training Contract

## Parameter boundary

The optimizer owns only Context Encoder `C`. Tracker Encoder `E` and Decoder `D` are strict-loaded
from the accepted nominal checkpoint, remain in evaluation mode, and have `requires_grad=false`.
There is no privileged-teacher model or teacher optimizer.

## Supervised row

```text
inputs:  causal deployable history H, current deployable Tracker input x
audit:   reference/fault/snapshot/time provenance
target:  same-state tracking-expert complete 29D action a_expert
output:  delta_z = C(H)
action:  a_context = D(E(x) + delta_z)
```

The primary loss compares `a_context` with `a_expert`. Future-state prediction is optional auxiliary
supervision only. A direct latent target is not required and is not accepted as primary supervision.

## Pre-training gates

1. A paired baseline proves the selected fault measurably degrades the frozen nominal controller.
2. The tracking expert passes the same reference/safety quality gate.
3. Same-state dataset provenance and Context-input privilege exclusion pass contract tests.
4. A bounded free-`delta_z` probe shows the frozen Decoder can express the corrective actions.
5. Zero repair is numerically equivalent to the nominal Tracker-Decoder action path.

Failure of any gate stops before optimizer construction.

## Training evidence

- Only Context Encoder parameters change after an update.
- Action loss and optional auxiliary losses are logged separately.
- Held-out rows split by fault identity and reference identity.
- Context-policy visited states are expert-relabeled before aggregation into later supervised rounds.
- Checkpoints bind the nominal Tracker/Decoder identity and the supervised dataset identity.

## Post-training gates

- Held-out complete-action imitation passes its accepted threshold.
- Closed-loop trajectory and safety metrics pass conjunctively against the fault baseline.
- History mask/shuffle degrades performance, ruling out a constant correction.
- Deployment loads only frozen `E`, `C`, and `D`; expert and privileged fields are absent.

Exact architecture, hyperparameters, commands, thresholds, and runtime owners remain open and must be
accepted before this design-only contract authorizes implementation or training.
