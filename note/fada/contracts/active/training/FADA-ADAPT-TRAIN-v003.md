# FADA-ADAPT-TRAIN-v003

Status: active offline construction contract; target collection and optimizer training remain
separately authorized long-run operations.
Supersedes: `FADA-ADAPT-TRAIN-v002`.

Inputs are a schema-5 source checkpoint and `fada-target-batch/v2`, both carrying the exact
`g1_fada_state_v2` architecture and source SHA-256. Output is `fada-adapted/v3`, whose manifest seals
the Q/V-only attention adapter type, projection set, exact IDM attention module names, LoRA
hyperparameters, architecture, optimizer state, source identity, target identity, and update cursor.

The deterministic episode/time split, LoRA-only update transaction, frozen Planner/base-IDM
ownership, first-action loss, and atomic persistence remain unchanged. The optimizer owns every and
only Q/V LoRA parameter exactly once. A zero adapter must reproduce the source policy before any
optimizer step.

The official Stage-D source reader admits only schema 5. The official Stage-C collector admits
schema 5 and `fada-adapted/v3`; it rejects historical adapted schemas before environment creation.
The generic deployable reader may load `fada-adapted/v1` and `fada-adapted/v2` through an isolated
legacy injector and must never reinterpret their all-Linear target manifest as the v3 Q/V design.
