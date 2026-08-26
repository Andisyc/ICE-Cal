---
contract_id: FADA-TRAIN-v010
status: superseded
effective_date: 2026-08-25
superseded_by: FADA-TRAIN-v011
scope: historical fresh persistent-async two-stage training
---

# FADA Two-Stage Source Training Contract v010 (superseded)

v010 required separate `idm_pretrain` and `planner` launches, transfer through
`pretrained_idm_path`, and schema-4 phase identity. v011 removes those controls and replaces them
with one fresh alternating campaign and a schema-5 checkpoint containing both optimizer states.

This document is historical evidence only and must not configure a new run.
