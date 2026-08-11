---
contract_id: FADA-TRAIN-v002
status: superseded
effective_date: 2026-08-04
updated_date: 2026-08-05
supersedes: FADA-TRAIN-v001
superseded_by: FADA-TRAIN-v003
method_contract: FADA-METHOD-v002
scope: UniLab source-domain Planner-IDM training through reduced iterative DAgger
---

# FADA UniLab Source Training Contract v002

This version implemented final-Oracle bootstrap plus Planner-IDM DAgger using only realized
trajectory pairs for IDM supervision. It explicitly excluded Oracle-shadow rollout and
intermediate Oracle checkpoints. The completed checkpoint was loadable but unstable in closed
loop. `FADA-TRAIN-v003` supersedes this reduced route with the full Appendix B.2 data contract
and quantitative source-quality evidence.
