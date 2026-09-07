# FADA research registry

This directory records the ICE-Cal/FADA axis-bank calibratable-Tracker design, its Contracts,
engineering transition, and the superseded Support–Query implementation lineage.

## Current semantic boundary

Active `FADA-METHOD-v024` and `FADA-TRAIN-v024` own the Source Oracle semantics. One privileged
G1WalkFlat/MuJoCo policy receives exact null or commanded-motion regimes: command sampling owns a
0.1 dead zone, null commands expose `[pi, pi]`, and non-null transitions restore a deterministic
pi-separated phase pair before advancing a bounded translation-and-yaw-conditioned clock. A
radian-normalized contact schedule replaces phase-height tracking; null-command foot-force balance
and torque relaxation remain quality terms rather than competing regime owners.

v024 preserves the v022 typed privileged input, broad physical DR and 20+1 lineage requirements.
The historical left-knee-only actuator-strength attenuation and its curriculum are disabled in
both source and v024 training; generic all-joint Kp/Kd randomization remains enabled. Its
production selector and owner modules are implemented, and the confirmed Module Test Cards pass
offline. `f_min`, `f_max`, command-intensity spans, contact-force calibration and Reward scales are
the first engineering values recorded by the v024 selector; policy acceptance remains constrained
by `testing/v024_reward_ordering_card.json`. A bounded official reset/step smoke has passed; the
full formal audit and training have not run. v023 is
historical and unrun; `mujoco_fada_phase` remains only a
compatibility and comparison selector. v022 remains the phase-neutral source baseline and its old
validation observation does not establish an admitted v024 lineage.

Active `FADA-CONTEXT-METHOD-v009` and `FADA-CONTEXT-TRAIN-v008` separately own data-driven
task-relevant correction-basis discovery, typed latent operators, coefficient readout and scale
calibration for a frozen Planner–Tracker. Their current status is engineering proposal, not the
previous v008/v007 analytic-axis implementation. Old v008/v007 module and review receipts are
historical and do not establish v009 correctness.

## Recall order

1. `../README.md` and `../governance/README.md`
2. `../architecture/08_in_context_execution_calibration.html`
3. `../architecture/09_in_context_execution_calibration_design_inspector.html`
4. `contracts/README.md` for active semantic authority
5. `plans/2026-09-07-command-gated-phase-contact-oracle.md` for the current Source Oracle design
6. `testing/v024_module_test_cards.md` and `testing/v024_reward_ordering_card.json`
7. `contracts/active/method/FADA-CONTEXT-METHOD-v009.md` and its v008 training pair for Context

Historical failed routes and old receipts are evidence, not policy-quality proof and not authority
to revive a superseded design.

## Active configuration surface

- `mujoco_clean_baseline`: gait-free, privilege-free, and domain-randomization-free SAC baseline.
- `mujoco_fada_source`: canonical v022 phase-neutral privileged source teacher; left-knee-only
  actuator-strength DR is disabled while generic all-joint Kp/Kd and broad physical DR remain.
- `mujoco_fada_phase_contact`: canonical v024 command-gated phase/contact Source Oracle.
- `mujoco_fada_phase`: historical v023 compatibility/comparison profile.
- `mujoco_fada_target` plus the `fault` or `target_domain` Hydra group: target collection and
  evaluation conditions.
- `mujoco_context_teacher` and `mujoco_context_teacher_phase1`: the distinct full-action and
  residual Context teacher runtimes.

Superseded diagnostic and gait-reward experiment names are no longer active Hydra task selectors.
Their checkpoint behavior maps to the canonical source or phase profile recorded in
`plans/2026-09-07-repository-simplification.md`.

The approved 15-degree narrow-slope reproduction has a separate operational
runbook at `docs/runbooks/fada-slope-traversal.md`. It keeps source training,
target-only collection, Q/V-only IDM LoRA adaptation, and same-snapshot
before/after evaluation as four distinct evidence boundaries.
