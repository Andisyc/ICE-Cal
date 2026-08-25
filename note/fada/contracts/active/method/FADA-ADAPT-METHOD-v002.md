# FADA-ADAPT-METHOD-v002

Status: active for paper Figure 3(d) reproduction after the input-v2 source campaign.

The LoRA target, frozen Planner/base IDM, rank `8`, alpha `16`, dropout `0.05`, first-action MSE, and
receding-horizon execution remain unchanged from v001.

Every target window must use `g1_fada_state_v2`: 66-D observation history and 66-D realized future,
with 29-D action history/action target and 3-D command carried separately. Action and command fields
from the environment's 98-D actor observation are forbidden in `Y_executed`. A legacy 98-D source or
target artifact must reject before split or optimizer construction. No legacy weight conversion is
permitted.

This contract does not authorize Stage-C collection, LoRA training, simulation evaluation, or
deployment.
