# FADA-ADAPT-METHOD-v002

Status: superseded on 2026-09-03 by `FADA-ADAPT-METHOD-v003`, which makes the original-LoRA
Q/V-only IDM attention target explicit and binds current adaptation to schema-5 source checkpoints.

The LoRA target, frozen Planner/base IDM, rank `8`, alpha `16`, dropout `0.05`, first-action MSE, and
receding-horizon execution remained unchanged from v001.

Every target window used `g1_fada_state_v2`: 66-D observation history and 66-D realized future,
with 29-D action history/action target and 3-D command carried separately. Action and command fields
from the environment's 98-D actor observation were forbidden in `Y_executed`. A legacy 98-D source
or target artifact had to reject before split or optimizer construction. No legacy weight
conversion was permitted.

This historical contract does not authorize current work.
