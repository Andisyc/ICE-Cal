---
contract_id: FADA-METHOD-v001
status: superseded
effective_date: 2026-08-04
superseded_by: FADA-METHOD-v002
scope: Planner and inverse-dynamics module construction through source training
---

# FADA Planner-IDM Method Contract v001

This version established the Planner-IDM factorization and causal target separation. It treated a
complete Oracle shadow rollout as mandatory for every Planner label, although Eq. 4.3 requires only
the same-state Oracle first action and UniLab has no complete public environment snapshot contract.
`FADA-METHOD-v002` retains Oracle shadow as an optional causally matched IDM augmentation and makes
the realized-trajectory path explicit.
