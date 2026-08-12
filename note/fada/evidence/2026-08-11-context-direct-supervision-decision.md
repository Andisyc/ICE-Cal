---
date: 2026-08-11
evidence_class: note-confirmed
contract: FADA-CONTEXT-METHOD-v002
status: human-accepted-design
---

# Context Direct-Supervision Decision

## Human decision

The Context phase does not train a privileged-teacher network. A fault-condition trajectory-tracking
expert generates same-state complete 29D corrective-action labels directly, and supervised learning
updates only the Context Encoder. Tracker Encoder and Decoder remain frozen.

## Accepted supervision boundary

```text
delta_z = ContextEncoder(causal deployable history)
a_context = FrozenDecoder(FrozenTrackerEncoder(x) + delta_z)
L_primary = distance(a_context, a_expert)
```

- Complete corrective action is the primary label.
- Future state/trajectory may be an auxiliary target and remains a closed-loop evaluation object.
- `delta_z` is not directly supervised as semantic ground truth because it need not be unique under
  the frozen Decoder.
- A free-latent optimization probe is required before training to test Decoder reachability; its
  optimized latent is diagnostic, not the Context label.

## Evidence limit

This is a human-confirmed method decision, not implementation or runtime evidence. The fault,
reference, tracking-expert provider, dataset, Decoder reachability, Context architecture, training,
and closed-loop quality remain unverified.

