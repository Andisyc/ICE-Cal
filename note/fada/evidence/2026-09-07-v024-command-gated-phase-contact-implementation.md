# FADA v024 Command-Gated Phase-Contact Implementation Evidence

Status: implemented / offline module tested / bounded runtime smoke passed / full formal runtime pending

Design identity: `FADA-METHOD-v024/FADA-TRAIN-v024/command_gated_phase_contact_v1`

## Implemented boundary

- One final Hydra selector: `mujoco_fada_phase_contact`.
- Command owner canonicalizes the joint planar/yaw dead zone once; downstream null gating uses
  exact zero.
- Phase owner supports deterministic stand/start phases, command-scaled frequency, all four
  null/motion transitions, old-command Reward timing, and next-observation resampling timing.
- An observation-only refresh can commit an external command transition but cannot advance the
  phase clock, resample, recompute Reward, or reuse a prior actuator target.
- Reward owner provides bounded phase/contact mismatch, exact-null force balance, and normalized
  clipped torque relaxation using public net-contact data and post-step virtual PD torque.
- Behavior preflight and schema-3 checkpoint identity reject v024 parameter/profile/config drift.
- Actor observation remains 98-D with command at `93:96` and phase at `96:98`; the FADA
  state/action/command contract remains `66/29/3`.

## Verification

- Focused v024 semantic/config/identity tests: `28 passed`.
- Independent final review characterization: `129 passed, 3 skipped`.
- Ruff on all v024 production and test files: passed.
- `git diff --check`: passed.
- Module-test manifest validator: `ADMITTED-OFFLINE`.
- Independent maintainability verdict: `CONSTRUCTION_PASS`; no P0/P1 findings remain.
- Official v024 MuJoCo reset/step smoke: passed with 32 environments; runner preflight, mixed
  null/non-null phase initialization, 98/303-D observations, finite Reward, and post-step target
  invalidation were observed.
- Source/v024 DR, preflight, actuator-provider, and official-route regression set: `89 passed`.

## Domain-randomization delta

Source and v024 training exclude targeted actuator faults. Left-knee gain attenuation remains a
downstream failed-rollout collection condition after Oracle freeze. Generic all-joint Kp/Kd
randomization and the remaining physical DR axes stay enabled. Runtime preflight enforces the
Oracle/downstream fault boundary without duplicating fault state in the Critic tail.

## Evidence boundary

No training, checkpoint round trip, simulator playback, deployment, or policy-quality evaluation
was run in this delta. The bounded reset/step smoke does not close the full formal route. The `1 N`
contact threshold, net-force orientation, Reward learnability, and gait quality remain live evidence
questions.
