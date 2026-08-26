---
contract_id: FADA-METHOD-v010
status: superseded
effective_date: 2026-08-25
superseded_by: FADA-METHOD-v011
scope: historical two-command IDM-pretrain then Planner-only training
---

# FADA Planner-IDM Method Contract v010 (superseded)

v010 split training into an IDM-only command followed by a Planner-only command with a permanently
frozen IDM. It used phase-owned schema-4 checkpoints and separate walking and standing Oracle
authorities. v011 retires this arrangement because the approved unified distillation Oracle already
owns both behaviours and the paper-style training order is intra-iteration, not two independent
campaigns.

This document is historical evidence only and must not configure a new run.
