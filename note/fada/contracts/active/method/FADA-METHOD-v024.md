---
contract_id: FADA-METHOD-v024
status: active / implemented / offline-module-tested
effective_date: 2026-09-07
supersedes: FADA-METHOD-v023
preserves_baseline: FADA-METHOD-v022
design_plan: ../../../plans/2026-09-07-command-gated-phase-contact-oracle.md
scope: command-gated phase/contact scheduling for one mixed standing-and-walking privileged Oracle
---

# FADA Command-Gated Phase-Contact Oracle Method Contract

## Objective

Train one privileged `G1WalkFlat` MuJoCo Oracle that can stand under a null command and walk under
a non-null command without optimizing two contradictory gait and standing objectives. The command
owns whether locomotion is requested; phase owns the expected contact schedule only while that
command requests motion.

This Contract changes the Oracle Reward and command semantics. It does not change the Planner-IDM
architecture, FADA adaptation method, Context calibration method, backend contract, or checkpoint
schema by itself.

## Command and regime ownership

1. Immediately after command sampling, set the command to exact zero when
   `norm(command_xy) < 0.1 m/s` and `abs(command_yaw) < 0.1 rad/s`.
2. Apply the configured standing-environment fraction after dead-zone normalization. The initial
   route preserves `rel_standing_envs=0.3`.
3. Every downstream standing/gait branch uses exact null-command detection. It must not recreate a
   second interval threshold.
4. The first v024 profile sets `heading_command=false`. Any later command producer or heading
   feedback must feed the same final command canonicalizer before observation, phase, and Reward;
   no consumer may duplicate dead-zone logic.
5. A null reset and null-to-null step expose `[pi, pi]` and do not advance phase.
6. A non-null reset and null-to-non-null transition atomically initialize a deterministic
   pi-separated phase pair. The pair must begin with both feet in stance and one fixed leading leg;
   `[0, pi]` is an engineering candidate, not a frozen constant.
7. A non-null-to-non-null step preserves the pi separation and advances both phases at the same
   command-conditioned frequency. A non-null-to-null transition atomically exposes `[pi, pi]`.
   This command-transition initialization is distinct from any future touchdown phase reset.
8. Phase ownership belongs directly to this per-environment command-regime state machine and must
   not depend on `gait_constraint.enabled`.

After dead-zone normalization, translation and yaw jointly define a dimensionless
`s_cmd in [0, 1]` from their configured positive saturation spans. Pure yaw, lateral, backward, and
forward commands must all activate the clock. Frequency is piecewise:

- null command: `frequency = 0`;
- non-null command: `frequency = f_min + (f_max - f_min) * s_cmd`.

The non-null branch requires `0 < f_min <= f_max`; the jump from zero to `f_min` is the admitted
null/non-null boundary, not a third near-standing regime. The spans, `f_min`, and `f_max` are
engineering parameters requiring Module Test Card coverage before implementation or training.

## Contact-schedule objective

- Replace phase-height tracking with a per-foot stance/contact schedule.
- Phase is stored in radians. Each foot uses
  `phase_fraction = (phase mod 2*pi) / (2*pi)` and
  `stance_expected = phase_fraction < duty_factor`.
- The first profile uses `duty_factor = 0.55`; admissible deterministic pi-separated startup
  requires `0.5 < duty_factor < 1.0`.
- `contact_measured` is derived from the public foot-contact signal. `1 N` is only an initial
  calibration candidate, not a frozen sensor truth.
- Penalize a foot contacting during its swing window and leaving contact during its stance window.
  `contact_cost` is the mean per-foot mismatch and `reward_contact = -w_contact * contact_cost`.
  Correct matching returns zero and every mismatch returns a finite negative Reward; null command
  receives no constant positive bonus.
- Null-command phase degenerates to a two-feet-in-contact target.
- Relative-height references such as `min(left_foot_z, right_foot_z)` and equality tracking of a
  swing-height curve are forbidden.

Contact mismatch is a bounded, compensable soft constraint so recovery remains learnable. Its
coefficient is admitted only when the v024 Reward Ordering Card holds on the named test regimes.
Binary contact matching does not prove clearance, lack of sliding, load bearing, or task motion;
those remain separate external measurements.

The contact scheduler does not own foot-clearance optimization. Any later clearance constraint must
be introduced as a separately reviewed one-sided safety/feasibility term rather than hidden inside
the phase target.

## Standing quality

The prior generic Standing Reward group is not an independent regime owner. Two null-command-only
quality terms may remain:

- left/right vertical foot-force balance;
- actuator torque relaxation.

They refine load distribution and energy use; they do not decide whether the robot should stand.

## Reward responsibility and ordering

- velocity tracking owns whether body motion executes the command;
- contact scheduling owns stance/swing mismatch only;
- null-command foot-force balance and torque relaxation own load distribution and energy only;
- fall penalties are dense soft signals, while environment termination owns the non-compensable
  fall boundary.

No fixed ratio such as contact `5.0` versus tracking `1.0` is a method prior. Contact and standing
scales are engineering parameters selected only after formula sign, boundary, contact truth-table,
and all six relations in `FADA-V024-PHASE-CONTACT-ORDERING` are fixed. Total return alone cannot
establish the ordering.

## Observation, source, and persistence

- Actor and Critic retain the v022 typed privileged-input contract and broad physical DR. Targeted
  actuator faults are excluded from Oracle training; left-knee gain attenuation is reserved for
  downstream failed-rollout collection after Oracle freeze. Generic all-joint Kp/Kd randomization
  remains enabled.
- The policy and distilled student both observe command and phase.
- The source behavior identity is `command_gated_phase_contact_v1`.
- A source lineage must use one behavior identity and one dead-zone/contact/frequency configuration.
  Same-shape checkpoints from v022 or v023 cannot enter silently.
- Planner-IDM retains `H=30`, `K=6`, the 66-D state projection, 29-D previous Action history,
  command3, and action-free future contract.

## Evidence and fallback boundary

The command dead zone and null-command dispatcher have mature external precedents recorded in the
design plan. Their combination with deterministic regime-transition phase initialization,
dual-channel command intensity, piecewise non-null frequency, and the contact scheduler does not
yet have repository runtime evidence. A fixed-duration double-support `STARTING` state is the
smallest named fallback if direct pi-separated startup fails; activating it requires a new semantic
review and must not be hidden inside threshold or coefficient tuning.

The owner implementation and offline Module Test Cards are current. A bounded official reset/step
smoke has reached MuJoCo, but this does not prove full-route persistence, training stability, gait
quality, recovery, or policy quality.
