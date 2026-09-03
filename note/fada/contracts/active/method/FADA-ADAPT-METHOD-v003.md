# FADA-ADAPT-METHOD-v003

Status: active for paper Figure 3(d) reproduction from the current schema-5 Planner-IDM source.
Supersedes: `FADA-ADAPT-METHOD-v002`.

Target adaptation freezes the complete Planner and pretrained IDM. Rank-`8`, alpha-`16`,
dropout-`0.05` LoRA branches update only the query and value projections of every attention module
inside the IDM: encoder self-attention, decoder self-attention, and decoder cross-attention. Key and
output projections, embeddings, feed-forward layers, normalization, positional parameters, and the
action head remain frozen. The loss remains first-action MSE and deployment remains receding-horizon
first-action execution.

Every target window uses `g1_fada_state_v2`: 66-D observation history and 66-D realized future,
with 29-D action history/action target and 3-D command carried separately. Action and command fields
from the environment's 98-D actor observation are forbidden in `Y_executed`. Legacy 98-D source or
target artifacts reject before split or optimizer construction. No legacy weight conversion is
permitted.

The Stage-C target budget is expressed as approximately `6000` executed control steps. The target
collector derives the usable-window count from the checkpoint's `H`/`K` architecture and the
configured command ramp and settling interval; it does not encode the derived count as a second
method constant. Sim-to-sim slope collection uses a fixed-seed stratified sample of forward-speed
commands while keeping the target dynamics, reset pose, observation noise, pushes, and actuator
properties fixed. Each represented episode owns one unique command trial; the schedule fails closed
instead of cycling and silently duplicating deterministic trajectories.

Stage-D validation holds out complete command groups. No command value may appear in both training
and validation, and slope preflight rejects artifacts that reuse one command across represented
episodes. This separation diagnoses command interpolation; it does not replace closed-loop slope and
flat-ground evaluation.

Closed-loop evaluation uses the first `20` unique commands from the same fixed-seed stratified
Stage-C command sequence. Zero-shot and adapted policies restore the same complete simulator
snapshot for every command. Each policy is measured only on causal windows from its own rollout:
first-action IDM MSE measures realized-dynamics fit, while the first-action RMSE between the IDM
outputs for Planner-predicted and realized futures is the consistency gap. Metrics report paired
means, sample standard deviations, and win fractions; one command nearest `0.8 m/s` owns the
representative videos. These diagnostics do not alter training, action execution, or checkpoints.

Historical `fada-adapted/v1` and `fada-adapted/v2` policies retain explicit generic playback support
but are excluded from current Stage-C collection and Stage-D training.

This contract does not authorize Stage-C collection, LoRA training, simulation evaluation, or
deployment.
