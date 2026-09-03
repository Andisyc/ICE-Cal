# FADA-ADAPT-TRAIN-v002

Status: superseded on 2026-09-03 by `FADA-ADAPT-TRAIN-v003`, which uses schema-5 source checkpoints,
Q/V-only IDM attention adapters, and `fada-adapted/v3` persistence.

Inputs were described as a schema-3 source checkpoint and `fada-target-batch/v2`, both carrying the
exact `g1_fada_state_v2` architecture and source SHA-256. Output was `fada-adapted/v2`. The
deterministic episode split, LoRA-only update transaction, frozen parameter ownership, first-action
loss, atomic persistence, and repository-owned schedule remained unchanged from v001.

Generic readers could retain explicit legacy playback support. The official v2 Stage-C and Stage-D
composition roots had to reject legacy state before environment reset, split, optimizer
construction, or output write.

This historical contract does not authorize current work.
