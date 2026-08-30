# FADA Playback Boundary Refactor Design

## Goal

Make FADA nominal playback deterministic and independent from training domain
randomization while reducing diagnostic responsibilities in the G1 environment.

## Preserved behavior

- All training Hydra task names, schedules, tensor layouts, optimizer behavior,
  checkpoint schemas, and legacy execution routes remain unchanged.
- Training keeps its current command, reset, observation-noise, curriculum, and
  domain-randomization distributions.
- Existing public playback entrypoints and checkpoint loading remain compatible.
- Diagnostics remain opt-in and read-only.

## Ownership

- `BackendAdapter.build_play_env_cfg_override` owns applying a playback profile
  to the composed task environment.
- The FADA Planner-IDM task config owns the nominal playback profile. Playback
  must not reconstruct a list of training randomization fields in Python.
- `LocomotionDRProvider` owns reset-pose randomization. Its training default
  remains enabled; playback may disable it through `DomainRandConfig`.
- `g1/action_trace.py` owns G1 action-trace formatting. `G1WalkEnv` owns only
  collecting the values already available at the environment boundary.

## Data flow

`play_interactive.py` composes the selected task with `training.play_only=true`.
FADA playback asks `BackendAdapter` for a play override. The adapter starts from
the training task environment, applies the task-owned nominal `play_profile`,
and passes the resolved override to `create_env`. Keyboard reset may further set
the requested command, but it is not the owner of domain randomization.

## Nominal playback contract

- actuator, gain, friction, mass, COM, DoF-bias, gravity, armature, delay, torque
  disturbance, and pushes are disabled;
- observation noise and curriculum are disabled;
- command resampling and randomized walking commands are disabled;
- reset base velocity, XY displacement, and yaw randomization are disabled;
- action latency and action-execution faults remain disabled.

## Non-scope

- Removing legacy workflows or compatibility formats.
- Changing FADA model architecture, privileged inputs, training losses, or
  training curriculum.
- Starting simulation, training, deployment, Git commit, or publication.
- Mechanically splitting unrelated generic distillation workflow/data modules.

## Proof route

1. A config-level regression must fail before the fix because FADA playback
   still contains training randomization.
2. A reset-provider test must fail before the fix because nominal reset pose
   cannot currently be disabled.
3. Existing G1 action-trace output test characterizes the diagnostic behavior
   before extraction.
4. Focused config, playback, G1 reset/trace, and FADA playback tests must pass.
5. A final maintainability review must confirm that playback configuration and
   diagnostics each have one owner and no new reverse dependency.
