---
contract_id: FADA-METHOD-v017
status: historical
effective_date: 2026-08-27
supersedes: FADA-METHOD-v016
superseded_by: FADA-METHOD-v022
scope: nominal privileged-Oracle Planner-IDM source training with downstream-only failure injection
---

# FADA Planner–IDM Method Contract v017 — Historical

## Historical decision

v017 required the privileged Oracle to train under strictly nominal dynamics. Its Actor consumed
only the deployable 98-D observation, while its Critic retained the typed privileged state. Gain,
delay, bias, friction, mass, COM, Kp/Kd variation, pushes, and every other perturbation were reserved
for downstream failed-rollout collection.

The intended lineage contained intermediate checkpoints `240…4800` and final checkpoint `5000`
from one nominal run. Planner–IDM tensor semantics remained state66 + previous-action29 + command3,
with an action-free K×66 future and first-action receding-horizon execution.

## Supersession reason

The later v022 experiments established a different active source-teacher route: the Actor consumes
normalized live privileged information and the environment introduces grouped domain randomization
through an iteration curriculum. The v017 nominal-only ownership is therefore preserved here as a
historical decision and must not be used to reject or describe the v022 training lineage.

No v017 checkpoint or runtime observation is promoted into v022 evidence.
