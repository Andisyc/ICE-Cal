# FADA G1 15-Degree Slope Traversal Design

## Status

Human-approved design direction: reproduce the G1 narrow-slope straight-line
adaptation demonstration with a 15-degree incline. This document defines the
implementation boundary; it does not authorize training or simulation runs.

## Goal

Add a deterministic MuJoCo target domain in which the existing flat-ground G1
Planner-IDM policy walks uphill on a narrow ramp, then adapt only the IDM LoRA
parameters from target-domain rollouts and compare zero-shot against adapted
straight-line traversal under identical conditions.

The first version covers one uphill pass. It deliberately excludes the hardware
demonstration's 180-degree turn and return pass so that turning skill does not
confound the execution-alignment result.

## Paper Boundary

The paper's G1 hardware task uses two 15-degree slopes on an approximately
0.8 m-wide ramp and judges success by completing the route without falling or
leaving the ramp. Its simulation benchmark is different: G1 follows randomized
velocity commands on 20-degree slopes. This design selects the former visual
scenario, reduced to one uphill pass.

The FADA adaptation contract remains unchanged:

- freeze the Planner;
- freeze the pretrained IDM weights;
- optimize only IDM LoRA parameters from observed target-domain
  observation-action rollout windows;
- execute only the first action from each predicted action chunk.

Flat-ground and zero-shot rollouts are evaluation evidence, not LoRA training
examples.

## Non-Goals

- Do not retrain the Oracle, Planner, or base IDM.
- Do not add a slope-specific policy head, reward, action bound, clip, clamp,
  min/max rule, or controller correction.
- Do not use domain randomization, observation noise, pushes, or actuator faults
  in slope collection or evaluation.
- Do not train on nominal-versus-fault subtraction.
- Do not implement the 180-degree turn, downhill return, or real-robot control.
- Do not change the checkpoint policy I/O contract.

## Selected Architecture

### 1. Target-domain ownership

Replace the Stage C owner's assumption that every target domain is an actuator
fault with a typed target-domain configuration. The initial supported kinds are:

- `actuator_gain`, preserving the existing knee experiments;
- `slope`, owning scene selection and slope traversal boundaries.

Hydra selects the condition through a `target_domain` config group. The new
default demonstration config is `target_domain=slope_15`. Existing knee configs
remain available for reproducibility, but the slope route does not populate or
validate actuator indices and multipliers.

The scripts remain thin composition entrypoints. Target-domain parsing,
validation, environment overrides, collection acceptance, and artifact identity
belong to the FADA target-domain owner modules.

### 2. Environment and scene

Add a task-level MuJoCo scene for the target domain. The scene contains:

- a short flat approach used only to establish a stable gait;
- one 15-degree uphill ramp;
- a 1.5 m flat approach followed by an 8.0 m ramp measured along its surface;
- 0.8 m of traversable width;
- no supporting floor beside the ramp that could hide a step-off failure.

The keyframe stays in the task-level XML. `robot.xml` remains a pure robot asset.
The task reuses the existing G1 locomotion environment and policy observation and
action layout. A dedicated Hydra task owner selects the slope scene instead of
switching the backend or mutating the scene after environment construction.

The environment config sets all domain randomization, observation noise, pushes,
action latency randomization, and actuator faults off. A preflight assertion
checks these fields before collection begins. The assertion fails closed and
names any non-nominal field.

The slope scene keeps the existing foot-contact sensor names, but each sensor
filters only by its foot geom or foot body instead of naming one ground geom.
This lets the same observation contract report contact on both the flat approach
and slope without adding a second sensor family or changing policy inputs.

### 3. Command and collection lifecycle

The default evaluation command is `[0.8, 0.0, 0.0]`: forward speed only, with no
lateral or yaw command. The existing short startup profile is retained:

- ramp from zero to the target command over 25 control steps;
- allow 50 additional control steps for gait stabilization;
- begin accepting training records only after the robot crosses the configured
  slope-entry boundary.

Time alone does not authorize collection. The slope-entry condition is computed
from world-frame base and foot state exposed through the public G1 task boundary;
that accessor uses the backend's declared base-state and sensor APIs. Startup
steps on flat ground are retained for video context but excluded from the
target-domain training batch.

Stage C aggregates 6000 accepted target-domain control steps across episodes.
An episode ends on a fall, ramp exit, environment termination, or reaching the
uphill finish boundary. Partial causal windows are discarded rather than padded
or joined across resets. Collection resets the existing environment through its
normal lifecycle and continues until the accepted-step target is met or a
24,000 total-step budget is exhausted.

Slope collection disables autoreset before stepping so terminal body state is
observable before reset. Each explicit reset uses the environment's normal
reset lifecycle, then clears policy history and causal records as one
transaction. A transition is accepted only when its pre-action state already
satisfies the slope-entry gate.

The ramp-local surface coordinate is `s=0` at the bottom edge. Collection starts
only when the pelvis reaches `s>=0.25 m` and both foot bodies have entered the
ramp. The uphill pass finishes at `s>=7.5 m`, leaving 0.5 m before the physical
edge.

This replaces the current `single_trajectory=true` requirement for the slope
condition. Actuator-gain collection retains its existing single-trajectory
lifecycle through the same typed owner.

To avoid collecting the same deterministic attempt repeatedly, slope collection
cycles one forward command per episode through `0.75`, `0.80`, and `0.85 m/s`,
while keeping lateral and yaw commands at zero. The sequence is part of artifact
identity, not domain randomization. Evaluation always uses the fixed 0.8 m/s
command.

### 4. Data and artifact contract

The slope Stage C bundle contains:

```text
artifacts/fada_target/g1_slope_15_mujoco/
  target.pt
  collection.mp4
  collection_summary.json
  manifest.json
```

`target.pt` uses target-artifact schema `fada-target-batch/v3`, succeeding the
current actuator-fault schema `fada-target-batch/v2`. It stores only accepted
target-domain causal windows and explicit episode boundaries. Its metadata
records:

- schema version;
- target-domain identity and slope geometry;
- source checkpoint identity;
- observation/action/command contracts;
- command sequence;
- accepted and rejected step counts;
- episode count and termination reasons;
- randomness-disabled assertions.

The new schema uses `target_domain_id` instead of requiring `fault_profile`.
Stage D accepts this schema and continues to accept the current actuator-fault
schema through one explicit migration boundary. It never guesses target-domain
identity from a directory name.

`collection.mp4` is rendered from the longest accepted contiguous episode so the
video shows one understandable attempt rather than concatenated resets.

### 5. Adaptation

Stage D reads `target.pt` only. It preserves the current paper-aligned LoRA
contract: rank 8, alpha 16, dropout 0.05, Planner frozen, base IDM frozen, and
LoRA inserted only in the IDM attention projections already selected by the
active adaptation contract.

Temporal train/validation splitting remains episode-aware. Windows from one
causal neighborhood cannot cross the split or its purge gap. Admission rejects
flat-ground, mixed-condition, malformed, or checkpoint-incompatible data before
constructing the optimizer.

The adapted checkpoint records both its source checkpoint lineage and the exact
target bundle identity.

### 6. Before/after evaluation

Evaluation is a separate workflow, not part of the adaptation dataset builder.
It runs:

1. the source Planner-IDM checkpoint zero-shot on the slope;
2. the adapted Planner-IDM checkpoint on the same slope;
3. a flat-ground regression for both checkpoints, enabled by default through
   `evaluation.run_flat_regression=true`.

The slope pair uses identical command, reset state, seed, control horizon, and
termination rules. Identity means restoring the complete
`capture_rollout_snapshot()` state, including backend, task, RNG, cached final
observation, and autoreset state; restoring only physics coordinates is
insufficient. The evaluator saves:

```text
artifacts/fada_evaluation/g1_slope_15_mujoco/<run-id>/
  zero_shot.mp4
  adapted.mp4
  metrics.json
  manifest.json
```

Metrics are computed in ramp coordinates:

- final, maximum, mean, and RMS lateral centerline error;
- final, maximum, mean, and RMS yaw error relative to ramp heading;
- uphill progress along the ramp surface;
- forward-velocity tracking error;
- fall, finish, and ramp-exit status;
- time and distance before failure.

Foot positions, obtained through the public G1 task-state accessor, define
whether either foot center leaves the 0.8 m corridor. Pelvis lateral error is
reported separately and is not mislabeled as a foot step-off.

For every scalar error metric, improvement is reported as
`zero_shot - adapted`, so positive values consistently mean adaptation improved
the result. Raw signed lateral and yaw trajectories are also preserved.

The flat regression is diagnostic only. It reports whether slope adaptation
causes a material loss of flat-ground survival, progress, or tracking quality;
it does not enter the LoRA objective.

## Data Flow

```text
source Planner-IDM checkpoint
        |
        v
15-degree target env -- target-only rollouts --> Stage C target.pt
        |                                        |
        |                                        v
        |                              frozen Planner + frozen IDM
        |                                        + IDM LoRA training
        |                                        |
        +-------------------------------> adapted checkpoint
                                                 |
source checkpoint + adapted checkpoint ----------+
        |
        v
same-seed slope evaluation --> videos + straight-line metrics
```

## Failure Handling

- Configuration mismatch fails before environment construction.
- A policy I/O or checkpoint lineage mismatch fails before rollout.
- Failure to enter the slope is counted by reason and eventually raises a
  bounded collection error; it is not silently accepted as target data.
- A fall or ramp exit ends only the current episode, not the whole collection.
- Exhausting the total-step budget reports accepted steps, episode counts, and
  termination histogram.
- Artifact publication is transactional: incomplete collection, video, or
  manifest output is kept out of the final bundle directory.
- Evaluation never overwrites an existing run directory.

## Compatibility and Retirement

- Preserve the existing Planner-IDM checkpoint schema and playback controller.
- Treat the current working-tree Q/V-attention `fada-adapted/v3` implementation
  as an explicit prerequisite. Slope work preserves its exact adapter and
  legacy-loader behavior rather than reconstructing it from committed `HEAD`.
- Preserve existing actuator-gain artifacts and configs as historical experiment
  inputs.
- Retire `nominal`, `faulty`, `delta`, and `excess` terminology from the slope
  route. They do not describe FADA's target-only adaptation supervision.
- Do not add slope branches to the generic script. Dispatch occurs through the
  typed target-domain owner selected by Hydra.
- Keep MuJoCo scene and rendering details in the backend/config boundary.

## Verification

Implementation must add RED-first tests for:

1. slope task composition and task-level XML loading;
2. the 15-degree geometry and 0.8 m corridor metadata;
3. fail-closed rejection of any randomization, noise, push, latency, or actuator
   fault in slope mode;
4. unchanged 98-D raw and 66-D projected observation contracts;
5. slope-entry-gated acceptance and exclusion of flat startup records;
6. causal windows never crossing episode resets;
7. multi-episode aggregation reaching the accepted-step budget;
8. bounded failure when the robot never reaches the slope;
9. schema admission and explicit migration of actuator-fault artifacts;
10. Stage D freezing Planner and base IDM while updating only LoRA parameters;
11. identical zero-shot/adapted evaluation conditions;
12. ramp-coordinate metrics, support-foot exit detection, and improvement sign;
13. transactional artifact publication and MuJoCo video rendering.
14. non-autoreset terminal capture, atomic episode reset, and full rollout-state
    restoration during paired evaluation.

Focused tests run first, followed by the affected FADA suite, configuration
composition tests, Ruff, Pyright on changed ownership boundaries, and
`git diff --check`. Real MuJoCo collection, LoRA training, and policy-quality
claims remain separate user-authorized gates.

## Acceptance Criteria

The implementation is complete when:

- one config-selected command launches Stage C in the deterministic 15-degree
  slope domain;
- Stage C produces a valid target-only bundle from one or more episodes without
  mixing startup or flat-ground records;
- Stage D consumes that bundle without modifying Planner or base IDM weights;
- one evaluation entrypoint produces comparable zero-shot and adapted videos and
  straight-line metrics under identical slope conditions;
- the prior actuator-gain artifacts remain readable at their explicit migration
  boundary;
- all scoped automated verification passes.
