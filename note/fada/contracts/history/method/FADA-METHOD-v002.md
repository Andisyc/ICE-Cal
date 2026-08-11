---
contract_id: FADA-METHOD-v002
status: superseded
effective_date: 2026-08-04
updated_date: 2026-08-05
supersedes: FADA-METHOD-v001
superseded_by: FADA-METHOD-v003
scope: Planner and inverse-dynamics construction through source training
---

# FADA Planner-IDM Method Contract v002

This version admitted realized trajectory pairs as the complete UniLab IDM source and treated
Oracle-shadow rollout as optional. Runtime quality evidence on 2026-08-05 contradicted that
reduction: Planner futures left the IDM support domain and closed-loop execution was unstable.
Appendix B.2 of `FADA.pdf` requires same-state final-Oracle shadow rollout for student visited
states and 20 intermediate Oracle checkpoints with a 2:1 suboptimal-to-optimal data budget.
`FADA-METHOD-v003` restores those paper-defined source-data semantics.
