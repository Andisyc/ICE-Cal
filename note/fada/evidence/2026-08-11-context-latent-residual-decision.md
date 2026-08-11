---
date: 2026-08-11
evidence_class: note-confirmed
authority: human decision
contract: FADA-CONTEXT-METHOD-v001
---

# Context Latent-Residual Decision

The human confirmed that the Context Encoder follows the first proposed fusion design:

```text
delta_z = ContextEncoder(rollout_history)
z = FrozenTrackerEncoder(current_input)
z_repaired = z + delta_z
action = FrozenDecoder(z_repaired)
```

This resolves the previous architectural ambiguity. The Context Encoder output is a latent residual
inserted immediately before the existing Decoder. It is not a replacement latent and it is not an
action residual after the Decoder.

The complete accepted training relationship is:

- ideal information first trains Tracker Encoder and Decoder;
- a separate privileged teacher under the accepted anomaly outputs a complete 29D action;
- Tracker Encoder, Decoder, and teacher are frozen during Context distillation;
- only Context Encoder is trained, using the decoded student action against the teacher's complete
  action as the primary supervision;
- deployment uses frozen Tracker Encoder, Context Encoder, and Decoder without `g`, teacher, or
  online parameter updates.

This note does not confirm the anomaly model, teacher architecture, Context history window, latent
constraints, exact loss, or evaluation threshold. The existing fixed-left-knee gain-scaling
experiment failed intervention validity and therefore cannot supply the teacher prerequisite.
