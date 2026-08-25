# FADA-ADAPT-TRAIN-v002

Status: active offline construction contract; target collection and optimizer training remain
separately authorized long-run operations.

Inputs are a schema-3 source checkpoint and `fada-target-batch/v2`, both carrying the exact
`g1_fada_state_v2` architecture and source SHA-256. Output is `fada-adapted/v2`. The deterministic
episode split, LoRA-only update transaction, frozen parameter ownership, first-action loss, atomic
persistence, and repository-owned schedule remain unchanged from v001.

Generic readers may retain explicit legacy playback support. The official v2 Stage-C and Stage-D
composition roots must reject legacy state before environment reset, split, optimizer construction,
or output write.
