# Current Distillation Evidence Ledger

Date: 2026-07-15

## E1: Current Code Ownership

- Source: CodeGraph exploration of `train_distill.py`, `collector.py`, `data.py`,
  `trainer.py`, `dagger.py`, `checkpoint.py`, and interactive playback.
- Class: code-confirmed.
- Fact: current code has separate owners for collection, dataset schema,
  behavior loss, DAgger, checkpoint persistence, and playback.
- Limitation: code ownership does not prove physical policy quality.

## E2: Local Candidate Artifacts

- Source command: `ls -lh *.pt` on 2026-07-15.
- Class: artifact-confirmed.
- Facts:
  - `walk_stand_moe_aggregated.pt`: about 5.7 MB.
  - `walk_stand_moe_expert_rollout.pt`: about 3.2 MB.
  - `walk_stand_moe_stand_fixed.pt`: about 5.7 MB.
- Limitation: file size includes optimizer-state coverage and is not a policy
  quality or network-completeness metric.

## E3: Repeated Startup And Stop-Transition Failure

- Source: user report on 2026-07-15.
- Class: human-observed live evidence.
- Fact: all three local candidates may require repeated launches before initial
  standing succeeds, and frequently lose balance when commanded motion stops.
- Limitation: no single structured log currently records checkpoint identity,
  seed/reset state, command schedule, and failure metrics together.

## E4: Expert-Rollout Candidate Zero-Command Differential

- Source command:
  `uv run /private/tmp/unilab_distill_zero_command_diff.py`.
- Class: runtime-confirmed, local MuJoCo differential.
- Facts:
  - `walk_stand_moe_expert_rollout.pt` remained near `base_z=0.735` and below
    roughly `1.3 deg` tilt for 100 zero-command steps on CPU and MPS.
  - Student/standing-teacher action MSE was generally `1e-6` to `1e-5` after
    the initial transient.
  - One stale random-command observation did not reproduce the reported fall.
- Limitation: 100 steps and one reset do not test restart sensitivity or
  walk-to-stop recovery.

## E5: Current Generic Playback Sentinel Is Weak

- Source: `scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py`.
- Class: code-confirmed and runtime-confirmed.
- Fact: the sentinel currently passes finite physics shape, finite/non-zero
  actions, checkpoint load, and routing agreement.
- Limitation: it does not require stable base height, tilt, non-termination,
  repeated reset success, or stop-transition recovery.

## E6: Human-Controlled Atlas And Document Contracts

- Source commands: `npm run check` under
  `note/architecture/auxiliary/atlas_app/` and `jq empty` over the three active
  distillation maps on 2026-07-15.
- Class: document-contract-confirmed.
- Facts:
  - the Concept Figure has six visible method/execution blocks;
  - five method design points map one-to-one to active contract sections;
  - the Method-to-Code atlas has ten runtime-ordered distillation owner cards;
  - the UniLab Runtime atlas has nine runtime-ordered reading cards from CLI
    routing through playback and contains no supporting row;
  - source paths and positive line hints pass repository-local validation.
- Limitation: static validation does not by itself prove browser click delivery
  to the editor; that is a separate interaction acceptance.

## E7: Rendered Source Navigation

- Source: in-app browser interaction against the repository-local atlas server
  on `http://127.0.0.1:8766/` on 2026-07-15.
- Class: interaction-confirmed.
- Facts:
  - the actual Method-to-Code SVG rendered 19 source links;
  - the actual UniLab Runtime SVG rendered 22 source links;
  - the first generated href was
    `/open-source?path=conf%2Fdistill%2Fconfig.yaml&line=1`;
  - clicking the visible reading-card row changed the page status to
    `opened conf/distill/config.yaml:1`;
  - the server logged the same path/line and the VS Code CLI exited with code 0;
  - comparison with the working FEMR 04 Atlas showed that the stable interaction
    contract is `preventDefault` plus same-origin `fetch` POST and HTTP 204;
  - the UniLab entry and viewer now canonicalize file or preview-server access to
    `http://127.0.0.1:8766/`, so `/open-source` is owned by the Atlas server;
  - the FEMR-style contract was revalidated by clicking the rendered Runtime
    source row for `src/unilab/cli.py:172`: the page reported `opened`, the server
    logged the same location, and the VS Code CLI exited with code 0;
  - dry-run validation returned HTTP 200 for the valid location and HTTP 400
    for traversal, missing-file, and zero-line requests.
- Limitation: source navigation proves the human code-reading interaction, not
  policy behavior or physical checkpoint acceptance.

## E8: Causal Spine Concept Figure

- Source: `npm run check` under
  `note/architecture/auxiliary/atlas_app/` plus in-app browser rendering of the
  active Concept Figure on 2026-07-15.
- Class: document-contract-confirmed and interaction-confirmed.
- Facts:
  - six blocks render as one horizontal causal spine plus one lower DAgger
    block;
  - seven interactions use explicit side anchors and orthogonal segments;
  - the validator rejects connector segments entering an 18 px expanded
    non-endpoint block rectangle;
  - command routing stays above the spine and execution feedback stays below
    the spine;
  - Student-State DAgger feedback enters at right-center and leaves at
    left-center on one shared horizontal centerline; the right feedback uses a
    single orthogonal bend and both terminal segments are horizontal;
  - Student-State DAgger and MoE Student share one vertical centerline;
  - browser fit-width renders at 88 percent with all six titles and the three
    non-local labels visible;
  - visual inspection found no connector or label crossing a block.
- Limitation: this proves method readability and geometry, not implementation
  correctness or policy quality.

## E9: Playback Reset And Stop-Transition Root Causes

- Source: code-order audit, `log.txt`, and three-checkpoint parameter audit on
  2026-07-15.
- Class: integration-root-cause-confirmed for reset ordering;
  training-distribution-gap-confirmed for walk-to-stop.
- Evidence ledger:
  `2026-07-15-playback-reset-and-stop-transition-root-causes.md`.
- Facts:
  - playback resets the G1WalkFlat environment before keyboard control replaces
    the sampled command with zero;
  - the current zero-command trace begins with non-zero base linear and angular
    velocity inherited from reset;
  - current role-specific DAgger has no walk-to-zero transition collection;
  - correct hard routing selects stand expert 1 but does not place post-walk
    recovery states inside that expert's training distribution.
- Limitation: neither repair is implemented, and standing-teacher recovery on
  post-walk states remains unmeasured.

## E10: Iterative DAgger Execution Audit

- Source: `dagger.py`, `train_distill.py`,
  `tests/algos/test_g1_distillation_contract.py`, and
  `tests/scripts/test_train_scripts.py` on 2026-07-15.
- Class: code-confirmed and contract-test-confirmed.
- Facts:
  - `run_iterative_dagger_updates()` performs a new student rollout inside
    `for iteration in range(num_iterations)`;
  - every iteration appends the new dataset, aggregates datasets `1..k`, and
    runs `updates_per_iteration` optimizer updates before the next rollout;
  - `training.online_dagger=true` reaches this iterative owner through
    `run_online_dagger_update()`;
  - the separate collect, merge, and offline-update branches represent one
    outer DAgger iteration unless the human repeats the complete sequence with
    the updated checkpoint;
  - inner optimizer update count is not evidence of additional outer DAgger
    iterations.
- Limitation: this audit proves the generic per-role iterative path, not that the
  proposed single-entry multi-role workflow or walk-to-stop transition loop is
  implemented.

## E14: Bug 3 - Public DAgger Workflow Under-Iteration

- Source: `src/unilab/algos/torch/distill/dagger.py`,
  `src/unilab/algos/torch/distill/workflow.py`, `scripts/train_distill.py`,
  `conf/distill/config.yaml`, and the single-entry workflow proposal.
- Class: code-confirmed; workflow-contract-confirmed; physical training
  efficiency unmeasured.
- Symptom: the manual procedure exposes one student-policy rollout followed by
  one merge and offline update. Repeating only optimizer updates does not create
  a new student state distribution, so this is one outer DAgger iteration.
- Confirmed implementation:
  - `run_iterative_dagger_updates()` performs rollout, cumulative aggregation,
    and update inside `dagger_iterations` outer cycles;
  - `run_multirole_dagger_workflow()` preserves bootstrap plus prior rounds,
    writes checkpoint lineage, and supports same-run resume and parent fork;
  - role artifact preflight can reuse compatible stand/walk datasets while
    collecting only a missing role.
- Confirmed workflow gap:
  - `training.workflow.enabled` remains `false` by default, so the one-entry
    multi-role workflow is opt-in;
  - the low-level collect, merge, and offline-update branches remain easy to
    invoke as the apparent normal route;
  - the workflow profile supplies roles but does not itself make the public
    route fail closed against the one-shot diagnostic path.
- Decision: Bug 3 is a real workflow/entrypoint defect, not a missing DAgger
  algorithm. The method contract remains unchanged: finetuning is the update
  inside each outer DAgger cycle, not a separate final stage.
- Limitation: no claim is made here about the number of iterations needed for
  policy quality; that requires transition-aware live acceptance after Bugs 1
  and 2 are repaired.

## E11: Role Artifact Preflight And Manifest

- Source: `src/unilab/algos/torch/distill/workflow.py` and
  `tests/algos/test_distill_workflow.py` on 2026-07-15.
- Class: owner-contract-test-confirmed.
- Facts:
  - role reuse is bound to teacher bytes, dataset bytes, canonical owner config,
    schema, dimensions, projections, intent filter, and thresholds;
  - compatible stand/walk roles remain `REUSE` when an absent height role is
    added as `COLLECT`;
  - changed bytes fail as `STALE`, semantic shape changes fail as
    `INCOMPATIBLE`, and a filename without a manifest is not reusable;
  - manifest I/O is cold-path and atomic; dataset tensors are not loaded merely
    to decide reuse.
- Command: `uv run pytest tests/algos/test_distill_workflow.py -q`.
- Result: `4 passed`.
- Limitation: Bootstrap collection/training and iterative DAgger are not yet
  connected to this owner.

## E12: Single-Entry Bootstrap And Multi-Role DAgger Workflow

- Source: `workflow.py`, `train_distill.py`, `conf/distill/workflow/`, CLI,
  owner tests, script tests, and Atlas validator on 2026-07-15.
- Class: owner-contract and formal-route integration confirmed; physical policy
  quality untested.
- Facts:
  - `uv run train --algo distill` selects one default-off workflow owner;
  - the walk/stand profile resolves role owners and teacher identities without
    exposing manual collect/merge/finetune stages;
  - preflight collects only missing roles and can explicitly adopt a compatible
    legacy dataset after loading and validating its tensor/metadata contract;
  - each DAgger outer round rolls out the previous round's student, preserves
    role-labelled cumulative sources, updates, and atomically records lineage;
  - same-run resume skips completed rounds and fork does not mutate its parent.
- Commands:
  - `uv run pytest tests/algos/test_distill_workflow.py tests/algos/test_g1_distillation_contract.py tests/scripts/test_train_scripts.py tests/test_cli.py -q`
  - `uv run ruff check ...` over all touched Python owners/tests.
- Results: `302 passed`, 3 pre-existing zero-element tensor warnings; Ruff
  passes. The profile-only warning was removed and rechecked.
- Limitation: this does not implement transition-conditioned walk-to-stop
  collection, playback reset repair, physical acceptance, or promotion.

## E13: Reset And Transition Repair Preflight

- Source: fresh code audit recorded in
  `evidence/2026-07-16-reset-and-transition-repair-preflight.md`.
- Class: code-confirmed; prior reset symptom runtime-confirmed; teacher recovery
  authority unconfirmed.
- Facts:
  - a second reset after writing zero is not a valid fix because reset samples
    and overwrites commands again;
  - the command-observation capability probe may reset physics and restore only
    command/observation arrays;
  - keyboard updates env command info after reset without refreshing env and
    playback-session observations;
  - hard routing reads the new env-info command while expert forward may consume
    a cached observation containing the previous command;
  - transition DAgger must remain blocked until synchronized standing-teacher
    recovery is measured on post-walk states.
- Decision: repair plan RT-1..RT-8 separates playback lifecycle, live reset,
  teacher authority, contract versioning, transition data, workflow, and
  physical promotion gates.
- Limitation: no code was changed and no new live run was performed.

## E15: RT-1 Deterministic Reset And Command-Observation Probes

- Date: 2026-07-16.
- Sources: `tests/envs/locomotion/g1/test_gait_constraint.py` and
  `tests/visualization/test_interactive_playback.py`.
- Class: deterministic probe-confirmed; no production behavior changed.
- Commands:
  - `uv run pytest tests/envs/locomotion/g1/test_gait_constraint.py -q -s -k 'rt1_playback_reset_probe_exposes_active_walk_reset or g1_standing_reset_zeros_base_qvel_without_touching_walk_samples'`
  - `uv run pytest tests/visualization/test_interactive_playback.py -q -s -k 'rt1_playback_probe_exposes_command_observation_skew or distill_playback_hard_routes_moe_by_command_intent'`
- Results:
  - reset probe: `command=[0.2, 0.0, 0.0]`, `gait_enabled=1.0`,
    `base_qvel_norm=0.653136`; `2 passed`;
  - command-observation probe: `routing=('active',)`,
    `runtime_command=[0.2, 0.0, 0.0]`,
    `cached_obs_command=[0.0, 0.0, 0.0]`, `action=[0.0, 0.0]`; `2 passed`.
- Facts:
  - without a standing-specific reset override, the G1 walk reset plan can
    produce an active gait and non-zero base velocity;
  - hard routing can observe the new active command while the playback session
    still feeds the policy the previous inactive observation;
  - both defects are reproducible without a simulator or policy-quality run.
- Decision: RT-1 is complete as a reproduction gate. RT-2 may now repair the
  playback reset distribution and atomic command-observation synchronization.
- Limitation: these are semantic fake-path probes; they do not prove MuJoCo
  timing, contact behavior, or standing-teacher recovery authority.

## E16: RT-2 Playback Reset And Atomic Command Synchronization

- Date: 2026-07-16.
- Sources: `src/unilab/base/np_env.py`,
  `src/unilab/visualization/interactive_playback.py`,
  `scripts/play_interactive.py`, and their focused tests.
- Class: implementation and contract-test-confirmed; live playback remains
  unconfirmed.
- Changes:
  - `NpEnv.refresh_state()` recomputes the current observation without physics
    stepping or changing `step_counter`;
  - `RslRlPlaybackSession.set_external_command()` writes the command, refreshes
    env state, then reloads the wrapper observation used by routing and experts;
  - playback capability probing refreshes the session cache after its reset and
    restore sequence;
  - G1 walk distill playback forces `rel_standing_envs=1.0`, disables transition
    reset sampling when configured, and zeros the standing reset qvel limit when
    that owner field exists; training reset configuration is unchanged.
- Command: `uv run pytest tests/base/test_np_env.py tests/visualization/test_interactive_playback.py tests/scripts/test_train_scripts.py -q`.
- Result: `258 passed in 6.27s`.
- Additional checks: Ruff passed for all touched Python owners/tests;
  `git diff --check` passed.
- Decision: RT-2 implementation gate is complete. Proceed to RT-3 repeated-reset
  live sentinel before any transition DAgger or teacher-authority work.
- Limitation: no MuJoCo viewer run has yet confirmed repeated cold starts,
  physical qvel reset, or command-driven route/expert timing.

## E17: RT-3 Initial Live Sentinel Failure And Owner Identity Gap

- Date: 2026-07-16.
- Source: scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py
  with a real MuJoCo g1_walk_flat/mujoco environment.
- Class: live integration failure; root cause localized to the shared playback
  reset-contract owner.
- Command:
  uv run python scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py
  --task g1_walk_flat/mujoco --action-mode policy --device cpu
  --make-temp-policy-checkpoint --temp-student-model-type moe
  --reset-repetitions 32 --steps 0
- Result: 32/32 reset records violated the intended standing contract;
  reset commands were non-zero, gait_enabled was 1, and base qvel was
  non-zero.
- Fact: the shared session receives Hydra's canonical G1WalkFlat, while the
  reset owner only recognized CLI-style g1_walk_flat; the helper therefore
  returned the unmodified env override.
- Decision: repair task identity normalization and add regression coverage for
  canonical, snake-case, and hyphenated task names before retrying RT-3.
- Limitation: this failure does not invalidate the RT-2 state-refresh design;
  it proves that the shared owner was not reached for one task identity form.

## E18: RT-3 Repeated-Reset MuJoCo Sentinel

- Date: 2026-07-16.
- Sources: src/unilab/visualization/interactive_playback.py,
  tests/scripts/test_train_scripts.py, and
  scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py.
- Class: live integration gate passed for reset lifecycle; policy quality is
  intentionally untested.
- Command:
  uv run python scripts/deploy/check_unilab_g1_distill_playback_live_sentinel.py
  --task g1_walk_flat/mujoco --action-mode policy --device cpu
  --make-temp-policy-checkpoint --temp-student-model-type moe
  --reset-repetitions 32 --steps 1
- Results: focused owner/sentinel tests 16 passed; Ruff and syntax checks
  passed; the live sentinel exited 0 with 32/32 reset records passing.
- Runtime facts: reset_command_abs_max=0,
  reset_actor_command_abs_max=0, reset_command_mismatch_abs_max=0,
  reset_gait_enabled_max=0, reset_base_qvel_norm_max=0,
  actions_shape=(1,29), and policy_action_nonzero=0.049958.
- Decision: RT-3 is complete. RT-4 may now measure standing-teacher recovery
  authority on synchronized post-walk states; do not infer that the student
  can stand, walk, or recover from this lifecycle gate.
- Limitation: the probe used a temporary MoE checkpoint and did not exercise
  command switching, real checkpoint quality, or walk-to-stop recovery.

## Decisions

- The three `.pt` files remain candidate artifacts, not accepted policies.
- The historical migration note is no longer the default current-method entry.
- `note/distillation/README.md` is the default human/LLM control-room entry.
- The active Concept Figure uses the human-selected Causal Spine composition.
- `Single Policy` is not an independent design point; single-checkpoint
  deployment is an output property of `MoE Student`.
- The Atlas entry contains only `01 UniLab Runtime Atlas`, `02 Method-to-Code
  Atlas`, and `03 Concept Figure`; superseded supporting maps remain only in Git
  history.
- Checkpoint lineage is implemented by the single-entry workflow. Acceptance
  and promotion remain pending until scenario semantics and thresholds exist.

## Open Risks

- Student walk-to-stop recovery has no live evidence.
- Current DAgger may match teacher actions locally without satisfying a repeated
  physical transition gate.
- Generic outer DAgger workflow connectivity is implemented; a walk-to-stop
  transition scenario remains absent.
- Height-control role is not trainable without a qualified teacher.
## E19: RT-4 Teacher Contract Preflight And Differential Probe Boundary

- Date: 2026-07-16.
- Sources: the two SAC checkpoint files,
  scripts/deploy/check_unilab_g1_distill_teacher_obs_contract.py,
  src/unilab/algos/torch/distill/teacher.py, and the RT-4 diagnostic probe.
- Class: checkpoint contract-confirmed; differential probe construction
  runtime-confirmed.
- Facts:
  - walking and standing checkpoints both expose actor input dim 98 and
    action dim 29; the live env exposes obs 98 and critic 101;
  - both checkpoint payloads contain actor/q/optimizer state but no normalizer
    payload; repository-native loading returns obs_normalizer=False for both;
  - the first vectorized-cohort attempt was rejected as probe evidence because
    reset rows diverged after stepping; the final probe disables observation
    noise only inside the diagnostic and replays one exact snapshot instead.
- Decision: use one static-standing baseline plus exact WT and WS branches from
  the same post-walk snapshot. No training or student checkpoint is involved.
- Limitation: this establishes teacher input compatibility and probe validity,
  not student recovery or long-horizon physical acceptance.

## E20: RT-4 Standing-Teacher Recovery Authority Differential

- Date: 2026-07-16.
- Source: scripts/deploy/check_unilab_g1_distill_teacher_recovery_differential.py
  with the real walking and standing SAC checkpoints.
- Class: live differential gate passed for the tested scenario.
- Command:
  PYTHONWARNINGS="ignore" HYDRA_FULL_ERROR=1 uv run python
  scripts/deploy/check_unilab_g1_distill_teacher_recovery_differential.py
  --pre-switch-steps 80 --post-switch-steps 80 --walk-vx 0.4 --device cpu
- Runtime facts:
  - the pre-walk walking-teacher rollout lasted 80 steps without termination;
  - WT and WS restored the same post-walk kinematic snapshot exactly
    (both restore max abs diff 0.0);
  - the zero-command switch was synchronized in both branches and disabled
    gait;
  - WT ran 80 post-switch steps without termination, min height 0.7152,
    max tilt 3.404 degrees;
  - WS ran 80 post-switch steps without termination, min height 0.7214,
    max tilt 2.799 degrees;
  - the WT and WS switch action MSE was identical at 0.040987;
  - static standing also ran 80 steps without termination.
- Decision: RT-4 passes for this walk-vx=0.4 scenario. The standing teacher
  is authorized as a candidate recovery oracle for RT-5 contract design.
- Limitation: this does not prove the distilled student recovers, does not
  cover lateral/yaw commands, and does not close the walk-to-stop gate.

## E21: RT-4 Final Differential Completion Rerun

- Date: 2026-07-16.
- Source: scripts/deploy/check_unilab_g1_distill_teacher_recovery_differential.py
  on the final probe version.
- Class: live differential completion gate passed.
- Command:
  PYTHONWARNINGS="ignore" HYDRA_FULL_ERROR=1 uv run python
  scripts/deploy/check_unilab_g1_distill_teacher_recovery_differential.py
  --pre-switch-steps 80 --post-switch-steps 80 --walk-vx 0.4 --device cpu
- Runtime facts:
  - pre-walk: 80 steps, no termination, min height 0.7173, max tilt 2.076
    degrees;
  - WT: 80 post-switch steps, no termination, min height 0.7137, max tilt
    2.905 degrees;
  - WS: 80 post-switch steps, no termination, min height 0.7279, max tilt
    2.666 degrees;
  - SS: 80 static-standing steps, no termination, min height 0.7346, max tilt
    1.227 degrees;
  - WT/WS snapshot restore max abs diff is 0.0 and switch action MSE is
    0.088259;
  - all 14 probe checks pass, including static gait disabled and both
    zero-command observation synchronizations.
- Decision: RT-4 is complete. Advance to the human approval gate for
  DISTILL-TRAIN-v002; do not treat this as student physical acceptance.
## E22: RT-5 Transition Training Contract Proposal

- Date: 2026-07-16.
- Source: note/distillation/plans/transition_training_contract_proposal.md.
- Class: design proposal and contract-alignment evidence; not active.
- Facts:
  - the proposal maps the transition scenario to DT-M-02, DT-M-04,
    DT-M-05, DT-M-01, and DT-X-01 without adding a top-level Concept Figure
    block or a third expert;
  - it defines pre-switch walking labels, an atomic zero-command switch,
    post-switch standing-teacher labels, scenario_label, transition_age,
    command_before, command_after, and checkpoint lineage;
  - it keeps DISTILL-TRAIN-v001 active and marks v002 as pending human approval;
  - it preserves existing static stand/walk datasets and proposes collection
    only for a missing walk_to_stop scenario.
- Decision: RT-5 proposal drafting is complete; do not enter RT-6 until the
  human approval gate in the proposal passes.
- Limitation: no transition schema, collector, workflow, or training code has
  been activated.

## E23: RT-5 Contract Activation

- Date: 2026-07-16.
- Sources: `note/distillation/contracts/active/training/DISTILL-TRAIN-v002.md`,
  `note/distillation/contracts/history/training/DISTILL-TRAIN-v001.md`, and
  the contract registry.
- Class: governance/contract evidence.
- Facts:
  - the user instruction to execute RT-6 was treated as acceptance of the
    already-reviewed v002 transition semantics;
  - v002 is the only active training contract in the registry;
  - v001 was copied unchanged into contract history and removed from the
    active training path;
  - the proposal is retained as an accepted decision record and points to the
    active v002 contract.
- Decision: transition semantics are active for implementation, while
  collector, workflow, and physical acceptance remain open gates.

## E24: RT-6a Transition Dataset Schema

- Date: 2026-07-16.
- Sources: `src/unilab/algos/torch/distill/data.py`,
  `src/unilab/algos/torch/distill/trainer.py`, and
  `tests/algos/test_g1_distillation_contract.py`.
- Class: S1 implementation contract evidence.
- Facts:
  - legacy datasets without transition fields remain loadable;
  - active transition rows preserve `scenario_labels`, int64
    `transition_ages`, `command_before`, and `command_after` through batch
    slicing and save/load;
  - `walk_to_stop` post-switch rows reject non-zero `command_after`, static and
    walk rows require age `-1`, and incomplete/mixed transition fields fail
    closed;
  - multi-source merge preserves transition chronology and rejects field
    presence mismatches rather than padding silently;
  - `uv run pytest tests/algos/test_g1_distillation_contract.py -q` reports
    `70 passed, 3 warnings`;
  - Ruff and `uv run python -m py_compile` pass for the touched owners/tests.
- Decision: RT-6a schema owner passes. No transition collection or training was
  started; RT-6b collector ownership is the next implementation boundary.

## E25: RT-6b Transition Collector Owner

- Date: 2026-07-16.
- Sources: `src/unilab/algos/torch/distill/collector.py` and
  `tests/algos/test_g1_distillation_contract.py`.
- Class: S2 collector contract/probe evidence; no live G1 collection.
- Facts:
  - the new `collect_transition_distillation_dataset_from_env()` path is
    opt-in and the existing collector default path is unchanged;
  - a vectorized golden env records active command rows, one atomic transition
    to zero command, and post-switch standing-teacher labels;
  - pre-switch rows carry `role_labels=walk_flat`, `command_intents=active`,
    and age `-1`; post-switch rows carry `role_labels=stand`,
    `command_intents=inactive`, and ages beginning at `0`;
  - a terminating row is reset to the active pre-switch phase and remains
    schema-valid; finite student rollout and teacher target actions are
    checked;
  - `uv run pytest tests/algos/test_g1_distillation_contract.py -q` reports
    `72 passed, 3 warnings` after the collector owner was added;
  - Ruff and py_compile pass for the collector, data, trainer, package export,
    and contract tests.
- Decision: RT-6 schema plus collector owner pass their bounded S1/S2 gates.
  The formal workflow does not dispatch this path yet, and no live transition
  dataset or student training run was started.

## E26: RT-7 Scenario Workflow Dispatch, Manifest Lineage, And Quota

- Date: 2026-07-16.
- Sources: `src/unilab/algos/torch/distill/data.py`,
  `src/unilab/algos/torch/distill/offline.py`,
  `src/unilab/algos/torch/distill/workflow.py`,
  `scripts/train_distill.py`, `conf/distill/workflow/g1_walk_stand.yaml`,
  and `tests/algos/test_distill_workflow.py`.
- Class: S2 formal-route connectivity and regression evidence; no real G1 run.
- Facts:
  - the workflow now dispatches three explicit scenarios: `walk_flat`,
    `static_stand`, and `walk_to_stop`; the transition scenario is represented
    as a scenario, not as a third teacher role;
  - scenario specifications carry `kind`, source roles, and positive quotas;
    weighted batch balancing allocates deterministic integer counts and
    transports scenario labels through the update batch;
  - each scenario artifact records scenario identity, source roles, dataset
    hash, sample count, and input checkpoint hash; resume rejects a changed
    scenario specification instead of silently continuing under a new plan;
  - the configured profile composes `walk_flat=0.50`,
    `static_stand=0.25`, and `walk_to_stop=0.25`, while the legacy role-only
    workflow remains available when scenarios are not configured;
  - the raw Hydra profile intentionally leaves
    `training.workflow.enabled=false` for backward compatibility; the public
    `uv run train --algo distill ...` CLI adds
    `training.workflow.enabled=true`, covered by the CLI route contract test;
  - focused distillation/config/script tests report `394 passed, 8 skipped,
    4 warnings`; the eight skips
    are the existing `mlx not installed` cases in `tests/scripts/test_train_scripts.py`;
    `tests/test_cli.py` reports `33 passed`; Ruff and py_compile pass for the
    touched owners; the formal Hydra compose prints all three scenarios and
    quotas;
  - no real teacher/student checkpoint was used for a formal transition run,
    no real transition dataset was persisted, and no physical student recovery
    gate is claimed.
- Decision: RT-7 passes its bounded S2 workflow integration gate. Advance to
  RT-8 for exact real-artifact execution and repeated physical acceptance.

## E27: RT-8 Bounded Real-Artifact Workflow

- Date: 2026-07-16.
- Sources: `logs/distill_workflow/rt8_bounded_20260716_retry4/run_manifest.json`,
  its checkpoint/dataset artifacts, the two teacher checkpoints, and the
  bounded public `uv run train --algo distill ... workflow=g1_walk_stand`
  command.
- Class: S2/S3 formal-route runtime evidence; no promotion claim.
- Facts:
  - walking teacher `model_5000.pt` is 98-D/29-D, has no `obs_normalizer`, and
    has SHA256 `7a0729a45859b2db05f2a642f6e80eedbd25f8135a75ff2af9dddae58bbf8279`;
    standing teacher is 98-D/29-D, has no `obs_normalizer`, and has SHA256
    `91e18d3d1f469b2bead350cd41b33494c39c8ec8d26f2daf802e0273afa2c6da`;
  - the public workflow collected 128 walking and 128 standing rows, created
    a 256-row bootstrap dataset, ran one DAgger iteration, and persisted a
    448-row aggregate with `walk_flat`, `static_stand`, and `walk_to_stop`
    sources;
  - the transition artifact contains 64 rows: 32 active pre-switch rows and
    32 inactive post-switch rows, with `command_after` zero and transition
    schema `DISTILL-TRAIN-v002`;
  - final candidate:
    `logs/distill_workflow/rt8_bounded_20260716_retry4/checkpoints/dagger_iteration_1.pt`;
    its SHA256 is
    `012f238569f3880ed20ef4e5336c556ece45c8cfc4507f6d3676a6430ab86f66`;
  - run manifest records the bootstrap checkpoint hash, scenario artifact
    hashes, teacher identities, input checkpoint lineage, and completed
    iteration `1`.
- Decision: RT-8a real-artifact workflow passes. The candidate remains subject
  to the separate physical gate.

## E28: RT-8 Student Transition Physical Gate

- Date: 2026-07-16.
- Source: `logs/distill_workflow/rt8_bounded_20260716_retry4/student_transition_live.txt`.
- Class: S4 live sentinel; gate failed.
- Command: `uv run scripts/deploy/check_unilab_g1_distill_student_transition_live.py`
  with `--repeats 32 --active-steps 20 --stop-steps 20 --device cpu`.
- Facts:
  - the exact candidate checkpoint loaded in actor mode with 98-D input, 29-D
    action, hard command routing, and no normalizer;
  - the grid covered 32 resets, 11 forward, 11 lateral, and 10 yaw transitions;
    all 96 phase blocks produced nonzero actions;
  - minimum base height was `0.61085`, maximum tilt was `64.8482` degrees,
    but `total_done_count=19` and `stop_speed_decay_pass=false`;
  - by command, done counts were forward `9`, lateral `5`, yaw `5`, and every
    command group failed the stop-speed decay condition;
  - the probe returned exit code `1` and the candidate was not promoted.
- Decision: RT-8b physical acceptance is blocked. The failure boundary is the
  student transition training/data owner, not checkpoint identity or workflow
  manifest lineage. Do not start an unbounded run from this candidate.

## E29: RT-9a Student-Transition Failure Mechanism Audit

- Date: 2026-07-16.
- Sources: `logs/distill_workflow/rt8_bounded_20260716_retry4/rt9a_transition_audit.txt`,
  the RT-8 transition dataset audit, the RT-8 manifest, and the candidate
  checkpoint runtime configuration.
- Class: core parameter path plus bounded live sentinel; no code or training
  behavior was changed.
- Facts:
  - the saved `walk_to_stop` artifact is schema-valid and semantically
    consistent: 64 rows, 32 active pre-switch rows, 32 inactive post-switch
    rows, `role_labels` match intent, `transition_ages` are `-1,0..7`, and
    `command_after` is zero for post-switch rows;
  - the artifact metadata records `pre_switch_steps=8`, `env_steps=15`,
    `switch_count=4`, `post_switch_rows=32`, and `done_seen_samples=0`;
    no post-switch state at age 8 or later and no failure-boundary row is
    present;
  - the aggregate contains 448 rows with scenario counts
    `static_stand=192`, `walk_flat=192`, and `walk_to_stop=64`; the formal
    iteration manifest records only `updates=4` with batch size 32 and a
    25% transition quota, so the bounded iteration exposes about 32 sampled
    transition rows, about half post-switch, with replacement;
  - forced-expert offline MSE on `walk_to_stop` improves from `0.046952`
    at `bootstrap_student.pt` to `0.042905` at `dagger_iteration_1.pt`,
    while the live candidate still fails the transition gate;
  - a three-command, 20-active/20-stop bounded live probe reports forward
    stop speed `1.2889` versus active `0.5435`, lateral stop speed `0.1489`
    with one termination, and yaw stop speed `0.0` with one termination;
    the summary has `total_done_count=2` and
    `stop_speed_decay_pass=false`;
  - the candidate loads with 98-D actor input, 29-D action output, no
    normalizer, and playback hard command routing targets
    `active->expert 0`, `inactive->expert 1`.
  - offline raw-router diagnostics on the aggregate select experts 0 and 2
    while expert 1 has zero argmax rows for both bootstrap and DAgGER-1;
    this does not invalidate playback hard routing, but it shows that the
    internal router loss has not converged under the bounded update budget.
- Decision: the first failed boundary is transition-state coverage and its
  training exposure, not dataset schema, checkpoint identity, normalizer,
  or playback hard-routing configuration. RT-9b must repair this owner before
  another physical acceptance run.
- Limitation: E29 identifies the coverage/training boundary; it does not yet
  decide the final collection horizon, quota, or optimizer update count.

## E30: RT-9b Transition Coverage And Training Exposure Owner

- Date: 2026-07-16.
- Sources: `src/unilab/algos/torch/distill/collector.py`,
  `src/unilab/algos/torch/distill/offline.py`, `scripts/train_distill.py`,
  `conf/distill/config.yaml`, `conf/distill/workflow/g1_walk_stand.yaml`, and
  the focused contract tests.
- Class: S1/S2 implementation and formal-route contract evidence; no bounded
  retrain or physical acceptance rerun was started.
- Facts:
  - transition collection now accepts `min_post_switch_steps`, requires at
    least `num_envs * (pre_switch_steps + min_post_switch_steps)` rows, checks
    the actual maximum post-switch age, and persists both horizon facts in
    dataset metadata;
  - the active walk/stand profile requires 20 post-switch steps and 8 expected
    replay passes for the `walk_to_stop` scenario;
  - offline balanced training now computes the minimum expected update budget
    for selected labels and fails closed when `max_updates` is too small;
  - the RT-8 aggregate numerical probe computes `required_updates=64` for
    transition replay under `batch_size=32`, quota `0.25`, and replay passes
    `8`, so the former `updates=4` configuration is rejected;
  - focused transition/replay tests pass (`5 passed`), the affected distill
    suite passes (`275 passed, 8 skipped`), Ruff passes, and `uv run python -m
    py_compile` passes for the touched Python owners;
  - the architecture atlas check passes with the repository's current Node
    runtime path: `roughjs viewer import and UniLab atlas data contracts OK`
    and `atlas OK runtime_modules=9 method_modules=11 concept_nodes=6`.
- Decision: RT-9b implementation and connector gates pass. Existing stand/walk
  role artifacts remain reusable. RT-9c is the next step: bounded retrain with
  the strengthened transition contract, followed by the same physical gate.
- Limitation: E30 proves the guard and exposure contract, not that the new
  candidate will physically recover; that remains an S4 live question.

## E31: RT-9c Reused-Role Bounded Retrain And Physical Gate

- Date: 2026-07-16.
- Sources: `logs/distill_workflow/rt9c_bounded_20260716_run2/run_manifest.json`,
  the RT-9c dataset audit outputs, and
  `logs/distill_workflow/rt9c_bounded_20260716_run2/student_transition_live.txt`.
- Class: S2 formal-route integration plus S4 MuJoCo live sentinel.
- Facts:
  - the workflow completed one bounded DAgger iteration with
    `role_decisions={stand: REUSE, walk_flat: REUSE}`; no stand/walk role
    recollection was performed;
  - the bootstrap aggregate contains 256 rows (128 walk/active and 128
    stand/inactive); the DAgger aggregate contains 640 rows from five sources:
    the two reused role artifacts, two student-state role rollouts, and one
    `walk_to_stop` transition rollout;
  - the new transition artifact has 128 rows, 32 active and 96 inactive,
    `pre_switch_steps=8`, `min_post_switch_steps=20`,
    `max_post_switch_age=23`, `post_switch_rows=96`,
    `done_seen_samples=0`, and `transition_schema=DISTILL-TRAIN-v002`;
  - the DAgger iteration manifest records `updates=128`, satisfying the
    strengthened transition replay budget for the 25% transition quota and
    eight requested replay passes; all audited tensors are finite with
    student/teacher observation dimensions 98 and action dimension 29;
  - the exact candidate
    `logs/distill_workflow/rt9c_bounded_20260716_run2/checkpoints/dagger_iteration_1.pt`
    was loaded in actor mode with no observation normalizer and hard routing
    `active->expert 0`, `inactive->expert 1`;
  - the same 32-reset forward/lateral/yaw MuJoCo gate produced nonzero actions
    in all 96 active/stop phase blocks but failed with `total_done_count=26`,
    `min_base_height=0.3016155`, `max_tilt_deg=64.9881`, and
    `stop_speed_decay_pass=false`; forward/lateral/yaw done counts were 8/9/9;
  - the live probe returned exit code 1 and the candidate was not promoted.
- Decision: RT-9c passes role reuse, transition coverage, replay exposure,
  schema, checkpoint identity, and runtime loading, but fails the physical
  student-transition gate. The strengthened transition contract is therefore
  integrated and exercised; it did not by itself establish recovery quality.
  Do not promote this candidate or launch an unbounded run.
- Limitation: E31 does not identify whether the remaining physical failure is
  caused by transition target quality, student rollout state distribution,
  router/role consistency, or optimizer/model capacity. The next step must
  isolate that policy-quality boundary before changing the role artifacts.

## E32: RT-9d Student-Transition Policy-Quality Isolation

- Date: 2026-07-16.
- Sources: `scripts/deploy/check_unilab_g1_distill_rt9d_policy_quality.py`,
  `logs/distill_workflow/rt9c_bounded_20260716_run2/rt9d_policy_quality.txt`,
  `logs/distill_workflow/rt9c_bounded_20260716_run2/rt9d_policy_quality_bootstrap.txt`,
  `logs/distill_workflow/rt9c_bounded_20260716_run2/rt9d_policy_quality_role_artifacts.txt`,
  the RT-9c checkpoint runtime config, and
  `scripts/train_distill.py` / `src/unilab/algos/torch/distill/dagger.py`.
- Class: core parameter path plus secondary offline connectivity probe; no
  training, collector, checkpoint, or MuJoCo behavior was changed.
- Facts:
  - the candidate runtime config contains
    `command_intent_loss_coef=0.25`, targets `active->0` and `inactive->1`,
    and `student_routing_mode=soft`;
  - on both bootstrap and RT-9c aggregate data, raw router argmax is expert 0
    for 100% of rows. Candidate mean route probabilities are
    `[0.5084, 0.3215, 0.1701]` for stand and `[0.5488, 0.2938, 0.1573]`
    for walk, so this is a semantic near-tie/bias rather than an empty or
    non-finite router output;
  - using the deployment-equivalent forced expert action, candidate MSE on
    the original role artifacts is stand `0.000798` and walk `0.010743`, while
    on the student transition artifact it is stand `0.005153` and walk
    `0.024733`; the student-state transition distribution amplifies target
    error, but the role targets remain learnable;
  - on the transition artifact, raw student action MSE for post-switch stand
    rows is `0.018147` versus `0.005153` with forced expert 1; the raw router
    selects expert 0 for all 96 stand rows;
  - code tracing shows ordinary role DAgger resolves
    `_FixedExpertRolloutPolicy`, while the `walk_to_stop` workflow loads the
    full student and passes it directly as `rollout_policy`; that full student
    uses the checkpoint's soft routing mode and does not receive the playback
    command-selected expert index;
  - the existing semantics probe's `--hard-routing` is not sufficient for this
    question because `MoEStudentPolicy.forward(hard_routing=True)` performs
    raw argmax; deployment hard routing instead indexes the selected expert
    action after reading the env command.
- Decision: the first actionable failed boundary is transition rollout policy
  semantics: `walk_to_stop` student-state collection is not using the same
  command-conditioned expert authority as deployment. This explains why
  transition states amplify stand/walk target error and why the raw router
  bias can contaminate post-switch states. Target quality is not the first
  owner to change, and role artifacts remain reusable.
- Limitation: E32 does not prove that replacing the transition rollout with a
  command-conditioned expert will pass MuJoCo. That repair and its bounded
  retrain are the next separate step.

## E33: RT-9e Command-Conditioned Transition Rollout Repair

- Date: 2026-07-16.
- Sources: `src/unilab/algos/torch/distill/collector.py`,
  `src/unilab/algos/torch/distill/dagger.py`, `scripts/train_distill.py`,
  `tests/algos/test_g1_distillation_contract.py`, and the updated
  `note/architecture/architecture/02_g1_distillation_method_to_code.data.json`.
- Class: S1 implementation plus S2 connector/architecture contract; no
  retrain or physical live gate was run.
- Facts:
  - transition collection now accepts either the legacy single rollout policy
    or an explicit `rollout_policies_by_intent` map, and fails closed when
    neither or both contracts are supplied;
  - with the map, pre-switch rows use the active policy and post-switch rows
    use the inactive policy based on the collector's existing `post_switch`
    state; metadata records `rollout_policy=command_intent_experts`;
  - `resolve_command_intent_rollout_policies()` resolves the checkpoint
    runtime targets and returns the corresponding MoE expert modules; missing
    or out-of-range active/inactive targets fail closed;
  - the semantic fixture observed actions `active, active, inactive` across
    the transition switch and verified the persisted rollout mode; the full
    affected suite passes `278 passed, 8 skipped`, Ruff passes, and the atlas
    check reports `runtime_modules=9 method_modules=11 concept_nodes=6`;
  - active `DISTILL-METHOD-v001` already forbids soft-mixture collection for
    pure experts, so no contract version change was required.
- Decision: RT-9e implementation and connector gates pass. The transition
  workflow no longer has to collect student states through an unbound full
  soft MoE; it can use the deployment-aligned active/inactive expert map.
  Existing role artifacts remain reusable.
- Limitation: the new route has not yet generated a real replacement
  `walk_to_stop.pt`, and no physical claim follows from these tests. RT-10
  must run one bounded retrain and the same MuJoCo gate before promotion.
