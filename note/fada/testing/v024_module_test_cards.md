# FADA v024 Module Test Cards

Status: human-confirmed specification implemented on 2026-09-07. Offline Module Correctness is
recorded by `../evidence/2026-09-07-v024-command-gated-phase-contact-module-test.json`; formal
runtime and policy-quality evidence do not exist yet.

Design identity: `FADA-METHOD-v024/FADA-TRAIN-v024/command_gated_phase_contact_v1`.
Reward ordering authority: `FADA-V024-PHASE-CONTACT-ORDERING` in
`v024_reward_ordering_card.json`.

## MTC-A — canonical command dispatcher and gait intensity

- **Module / semantic object:** final locomotion command, exact null regime, `s_cmd`, and frequency.
- **Requested behavior:** canonicalize the final command once; apply the 30 percent forced-standing
  selection after dead-zone normalization; map translation and yaw jointly to `s_cmd`; output zero
  frequency only for a null command and `[f_min, f_max]` for a non-null command.
- **Preserved behavior:** configured velocity ranges, command axes/order, per-environment batching,
  and command visibility to Actor/Critic/student remain unchanged.
- **One semantic owner:** command construction in `unilab.envs.locomotion.g1.walk_commands`, consumed
  by `G1WalkControlBindings` without downstream threshold recreation.
- **Production public input and output:** command `(N,3)`, standing-selection mask, dead-zone values,
  positive saturation spans, `f_min`, and `f_max` -> canonical command, exact null mask, `s_cmd`, and
  per-environment frequency.
- **State or transaction boundary:** after the last active command producer and before observation,
  phase, or Reward consumption. The first v024 profile has `heading_command=false`.
- **Forbidden dependencies and behavior:** no Reward-owned dead zone; no translation-only activity;
  no backend-private access; no third near-standing regime; no `getattr`/`hasattr` capability probe.
- **Semantic fixture:** asymmetric rows for forward, backward, lateral, pure yaw, mixed xy/yaw,
  exact zero, just inside each dead-zone boundary, exact equality, just outside, and saturation.
  Units are m/s for xy and rad/s for yaw; row identity must survive batching.
- **Independent expected answers:** both magnitudes strictly inside their dead zones become exact
  zero before standing selection; equality is non-null because the rule uses `<`; pure yaw and
  lateral rows have nonzero `s_cmd`; a non-null boundary row returns `f_min`; saturated rows return
  `f_max`; sign reversal at equal magnitude preserves intensity.
- **Sensitivity counterexample:** a translation-only mutant fails pure yaw; `<=` fails equality;
  applying standing selection first fails the ordering fixture; an extra consumer threshold causes
  observation/Reward disagreement.
- **Required S/C/T profile:** S1; C1-C4; T-value, T-boundary, T-order, T-role, T-invariance,
  T-failure.
- **Design and card identity:** `FADA-v024/MTC-A/canonical-command-v1`.
- **Stop condition:** any consumer sees a command, null mask, or frequency derived from a different
  threshold or command version.

## MTC-B — command-gated phase lifecycle

- **Module / semantic object:** per-environment null/non-null phase state machine.
- **Requested behavior:** null reset/step exposes `[pi,pi]`; non-null reset and null-to-non-null
  establish a deterministic pi-separated pair with both feet initially in stance and one fixed
  leading leg; continuing motion preserves the separation; motion-to-null returns to `[pi,pi]`.
- **Preserved behavior:** radian phase storage, modulo `2*pi` wrapping, batch independence, and
  policy observation of the current emitted phase.
- **One semantic owner:** phase lifecycle in `unilab.envs.locomotion.g1.walk_control`, used by the
  G1 control binding for two ordered operations without duplicated phase math.
- **Production public input and output:** prior phase `(N,2)`, prior null mask, current null mask,
  per-row frequency, `ctrl_dt`, stand phase, and configured startup phase -> `reward_phase` for the
  command that produced the current action, then `next_observation_phase` after an optional
  resampled-command regime transition.
- **State or transaction boundary:** each physics step advances the old-command phase at most once
  before Reward. Reward uses that command and `reward_phase`. After Reward, a resample may only
  absorb to stand or initialize startup phase for the next observation/action; this second
  operation must not advance phase again. No physics-step clock advance occurs elsewhere.
  An observation-only refresh may canonicalize an externally written command and apply only its
  regime transition; it must not advance the clock or resample.
- **Forbidden dependencies and behavior:** no dependency on `gait_constraint.enabled`; no random
  startup in the first profile; no touchdown reset; no synchronized non-null pair; no double update.
- **Semantic fixture:** four transition rows (`null->null`, `null->non-null`,
  `non-null->non-null`, `non-null->null`), plus null and non-null reset rows with asymmetric batch
  order and a wraparound phase. Resampling fixtures assert both the old-command `reward_phase` and
  the new-command `next_observation_phase` for null-to-non-null and non-null-to-null boundaries.
- **Independent expected answers:** null outputs exactly `[pi,pi]`; startup has circular separation
  pi, both normalized phases in the stance window, and the configured leading leg; continuing
  motion adds `2*pi*f*ctrl_dt` to both feet modulo `2*pi`; the separation remains pi. On a resample,
  Reward still sees the old command and its once-advanced phase, while the next observation sees
  the new command with `[0,pi]` or `[pi,pi]` and no additional increment.
- **Sensitivity counterexample:** current freeze-then-common-advance behavior starts from `[pi,pi]`
  and remains synchronized, so it must fail the separation assertion; random offset startup fails
  deterministic leading-leg identity.
- **Required S/C/T profile:** S1/S3; C1-C5; T-value, T-transition, T-order, T-identity,
  T-invariance, T-persist, T-diff.
- **Design and card identity:** `FADA-v024/MTC-B/phase-state-machine-v1`.
- **Stop condition:** any transition loses pi separation, changes the leading-leg identity, advances
  null phase, or requires a simulator to establish its module-local answer.

## MTC-C — phase/contact mismatch cost

- **Module / semantic object:** phase fraction, expected stance, measured contact, contact cost, and
  signed contact Reward.
- **Requested behavior:** normalize radian phase to cycle fraction; compare it with a strict
  `duty_factor` boundary; compare public vertical foot force with a strict contact threshold; return
  the mean binary mismatch as a bounded cost and its negative weighted Reward.
- **Preserved behavior:** left/right foot identity, public backend contact sensor contract, vectorized
  batch shape, and no ownership of foot-height targets.
- **One semantic owner:** contact cost math in `unilab.envs.locomotion.g1.walk_reward`; Reward
  bindings only gather the public sensor fields and register the returned term. The generic Reward
  dispatcher remains the sole scale/weight owner.
- **Production public input and output:** phase `(N,2)` in radians, left/right vertical force,
  `duty_factor`, and contact threshold -> expected/measured contact masks and per-row cost in
  `[0,1]`; the binding negates that cost and the dispatcher applies the configured positive weight,
  yielding the final contribution in `[-weight,0]`.
- **State or transaction boundary:** one Reward evaluation from the current phase and current public
  contact observation.
- **Forbidden dependencies and behavior:** no `phase < 0.55` radian comparison; no relative foot
  height; no swing-height equality target; no positive match bonus; no asset parsing or
  backend-private sensor access in the hot path.
- **Semantic fixture:** with `duty_factor=0.55`, use normalized phases `0.10`, `0.50`, exact `0.55`,
  and `0.60`, represented in radians; use forces below, equal to, and above the candidate threshold;
  reject nonpositive thresholds/weights and duty factors outside `(0.5,1.0)`.
- **Independent expected answers:** fraction `<0.55` is stance and equality is swing; force strictly
  above threshold is contact; zero mismatches gives cost/Reward `0/0`; one of two feet mismatching
  gives `0.5/-0.5*weight`; two mismatches gives `1/-weight`; `[pi,pi]` plus two contacts gives zero.
- **Sensitivity counterexample:** comparing radians directly marks `pi` as swing and fails the null
  fixture; a positive matching score gives a null-command bonus; swapping foot identities fails an
  asymmetric one-mismatch case.
- **Required S/C/T profile:** S1; C1-C4; T-value, T-sign, T-boundary, T-range, T-role,
  T-permutation, T-failure.
- **Design and card identity:** `FADA-v024/MTC-C/contact-cost-v1`.
- **Stop condition:** a correct null stand earns a positive constant, a mismatch escapes the bounded
  negative term, or the answer depends on foot height or private backend behavior.

## MTC-D — exact-null standing quality terms

- **Module / semantic object:** vertical foot-force balance and actuator torque relaxation under
  the exact null command.
- **Requested behavior:** force balance is
  `abs(Fz_left-Fz_right)/max(Fz_left+Fz_right,epsilon)`, with both-zero load defined as cost `1`;
  torque relaxation is the per-row mean across actuators of
  `(clip(tau,-tau_max,tau_max)/tau_max)^2`. Both return exactly zero for non-null rows.
- **Preserved behavior:** public net-contact sensor shape and force axis, actuator order, per-row DR
  gains, and public actuator force limits remain unchanged. Existing count-based standing terms are
  not reused or modified.
- **One semantic owner:** pure normalized costs in `walk_reward.py`; the binding gathers public
  force and the control owner publishes v024 torque provenance.
- **Production public input and output:** canonical command/null mask, left/right `Fz`, positive
  epsilon, torque `(N,A)`, and positive `tau_max (A,)` -> two finite per-row costs in `[0,1]`.
- **State or transaction boundary:** reset rows explicitly publish zero torque with an invalid-target
  marker because no action exists. Every actual v024 step must replace it from final actuator
  target, post-step q/dq, applied per-row Kp/Kd, and force limits before Reward, or fail closed. The
  marker is consumed once; reset/observation refresh preserves Reward and cannot reuse stale torque.
- **Forbidden dependencies and behavior:** no contact-count proxy, sum-over-actuators L2, duplicated
  threshold null mask, backend-private actuator force, or missing-target-to-zero fallback.
- **Semantic fixture:** equal load, asymmetric load, both-zero load, nominal/limit/excess torque,
  asymmetric actuator limits, exact-null and non-null rows, reset-only invalid marker, and actual-step
  missing target.
- **Independent expected answers:** equal nonzero load is `0`; one-sided load and both-zero load are
  `1`; half-limit torque on every actuator is `0.25` regardless of actuator count; clipped limit or
  excess is `1`; every non-null row is exactly `0`; missing actual-step target raises.
- **Sensitivity counterexample:** contact counts cannot distinguish `90/10 N` from `50/50 N`; sum L2
  changes by 29x with actuator count; silent missing-target zero appears optimal; threshold-gated
  null disagrees at the exact dead-zone boundary.
- **Required S/C/T profile:** S1/S3; C1-C5; T-value, T-boundary, T-range, T-role, T-invariance,
  T-failure, T-diff.
- **Design and card identity:** `FADA-v024/MTC-D/null-quality-v1`.
- **Stop condition:** a non-null row receives either cost, both-zero load is rewarded, actuator
  count changes torque scale, or a post-action step can use unproven zero torque.

## MTC-E — v024 configuration, observation, and lineage identity

- **Module / semantic object:** the future canonical v024 Hydra profile and persisted behavior
  identity.
- **Requested behavior:** compose exactly one G1WalkFlat/MuJoCo v024 selector containing the accepted
command, phase, contact, standing-quality, privilege, physical-DR, and sealed-lineage fields; expose
  command and phase to policy/student; persist `command_gated_phase_contact_v1` and the effective
  command/phase/contact configuration.
- **Preserved behavior:** v022 typed privilege and normalization, broad physical DR with generic
  all-joint Kp/Kd randomization, 98-D Actor task observation, 66/29/3 Planner-IDM split, schema-3
  sealed `240..4800 + 5000` lineage, and FADA-ADAPT v003 boundaries. Targeted actuator faults must
  remain outside Oracle training and are owned by downstream failed-rollout collection.
- **One semantic owner:** the future v024 task YAML owns the effective configuration; structured
  config and checkpoint/run metadata validate and persist it without Python-side hyperparameter
  reinterpretation.
- **Production public input and output:** Hydra task selector -> composed config, observation spec,
  run contract snapshot, and source checkpoint behavior identity.
- **State or transaction boundary:** config composition before environment creation and lineage
  validation before source collection/load.
- **Command transition timing:** the Reward for one physics step uses the command visible to the
  Actor that produced its action. A periodic resample is canonicalized and committed only to the
  next observation/next action, together with its phase transition. The reset observation carries
  an explicit zero-torque initial value because no actuator target exists; every actual step must
  refresh torque from the final target and post-step joint state or fail closed.
- **Forbidden dependencies and behavior:** `training.sim_backend` cannot switch backend; no silent
  use of `mujoco_fada_phase`; no same-shape v022/v023 checkpoint admission; no missing phase/command;
  no checkpoint schema padding or migration.
- **Semantic fixture:** canonical profile, historical compatibility profile, same-shape stale
  checkpoint, missing behavior identity, altered dead zone/frequency/contact setting, and preserved
  66/29/3 split.
- **Independent expected answers:** only the canonical profile composes as v024; targeted actuator
  faults are absent while generic all-joint Kp/Kd randomization remains true; every stale or
  mismatched identity fails before rollout; the
  policy/student retain command and phase; preserved dimensions and sealed checkpoint set remain
  exact.
- **Sensitivity counterexample:** relabeling `mujoco_fada_phase` as v024 or deleting one behavior
  field from persistence must fail identity assertions even when tensor dimensions match.
- **Required S/C/T profile:** S1/S3; C1-C5; T-shape, T-order, T-role, T-identity, T-persist,
  T-failure, T-diff.
- **Design and card identity:** `FADA-v024/MTC-E/config-lineage-v1`.
- **Stop condition:** configuration composition, observation construction, or checkpoint admission
  can disagree about the active behavior identity without failing closed.

## Evidence boundary

These confirmed cards authorize writing narrow module tests only after an affected-module
engineering plan and implementation authority exist. They do not establish `MODULE-CORRECT`,
official-route connectivity, simulator behavior, learnability, gait quality, or permission to train.
Every card first requires a semantic pseudo-sample through the eventual production public boundary
and a controlled sensitivity counterexample; copied formulas and total-return assertions are
forbidden oracles.
