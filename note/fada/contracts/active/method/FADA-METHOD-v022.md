---
contract_id: FADA-METHOD-v022
status: active
effective_date: 2026-08-29
supersedes: FADA-METHOD-v017
scope: live-privileged grouped-DR SAC teacher and Planner-IDM source construction
---

# FADA Planner–IDM Method Contract v022

## Source task and Reward

ICE-Cal owns one `G1WalkFlat/MuJoCo` source task and one locomotion Reward for every command. Gait
phase has no behavioral authority: its two compatibility positions remain zero, phase Reward terms
remain zero, and the Planner–IDM split remains state66 + previous-action29 + command3.

## Privileged teacher

The source Oracle is a training-only SAC teacher. Its policy trunk consumes the 98-D task observation
plus a learned embedding of the typed `g1_fada_privileged_v1` tail. The privileged tail is derived
from the Critic observation contract, normalized by one persisted empirical normalizer, and consumed
consistently by Collector and Learner. The Critic consumes the same normalized privileged tail.

The fixed-zero privileged-input profile is diagnostic only. The active teacher uses live privileged
values; deployability is recovered by downstream Planner–IDM distillation rather than by pretending
that the teacher Actor is privilege-free.

## Grouped perturbation curriculum

Teacher training begins near nominal dynamics and expands one grouped perturbation distribution by
training iteration. Levels change at iterations `0, 500, 1200, 2000, 3000, 4000`, with group scales
`0.0, 0.2, 0.4, 0.6, 0.8, 1.0`. The left-knee actuator-strength lower bound progresses through
`1.0, 0.98, 0.95, 0.9, 0.85, 0.8`, while its nominal probability progresses through
`1.0, 0.8, 0.7, 0.5, 0.4, 0.3`.

The final group includes Kp/Kd multipliers `[0.9,1.1]`, friction `[0.8,1.2]`, added base mass
`[-1.5,1.5]`, body-mass scale `[0.9,1.1]`, COM offsets `[-0.05,0.05]`, and DoF position bias
`[-0.025,0.025]`. Control delay and external pushes remain disabled. Episode termination may brake
curriculum advancement; training quality does not choose the nominal iteration boundaries.

## Planner–IDM and persistence

Planner–IDM preserves the 98→66/29/3 split, H=30 history, K=6 action-free future, causal
future–action pairing, IDM-before-Planner ordering, frozen-IDM Planner gradients, and first-action
receding-horizon execution.

Planner–IDM source replay treats command-speed coverage as a training-distribution identity rather
than a model input change. Walking steady-state rows are stratified by planar command norm at
`0.25` and `0.60 m/s`, with `0.10/0.30/0.60` slow/medium/high sampling. Planner and both IDM source
roles consume the same speed definition, while standing and walk-to-stand retain their scenario
identities. All required strata are admitted before optimizer mutation; replay coverage changes the
update budget but not the first-action objectives or frozen-IDM gradient boundary.

IDM coverage still requires one teacher lineage with exactly 20 intermediate checkpoints
`240…4800` and final checkpoint `5000`. A policy-quality validation run that saves only at 1000-step
intervals is not an admissible IDM lineage even when its Reward and episode length are high.

## Evidence boundary

The observed `G1WalkFlat_live_priv_grouped_dr_v022` run is qualitative policy-quality evidence for
the live-privileged curriculum route. Its exact remote metrics and artifacts are volatile. The
current validation profile uses `checkpoint_mode=validation` and `save_interval=1000`, so the run
does not satisfy the 20+1 persistence contract and cannot yet authorize IDM training.
