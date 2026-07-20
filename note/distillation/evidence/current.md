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

## E34: DAgger Persistent Runtime HP-1

- Date: 2026-07-16.
- Source: `evidence/2026-07-16-dagger-persistent-runtime-hp1.md`.
- Class: S1/S2 protocol and spawned-process lifecycle evidence; no real G1
  collection, training throughput, or physical behavior claim.
- Facts: one `AsyncRunner`-owned worker serves sequential typed requests;
  request/result and weight-version identity fail closed; worker exceptions
  propagate; cleanup reaps the process. The HP-1 test passes `4 passed`, the
  IPC/runtime group passes `53 passed`, the distill/workflow/script group
  passes `280 passed, 8 skipped`, the final ordered affected gate passes
  `341 passed`, and Ruff passes.
- Decision: HP-1 is complete. HP-2 may connect this service interface to the
  workflow while preserving the outer iteration barrier.
- Limitation: `SharedWeightSync` is represented by explicit expected/observed
  version identity but real shared-weight attachment remains HP-3. No speedup
  claim is authorized before structured HP-4 timing evidence.

## E35: DAgger Workflow Barrier HP-2

- Date: 2026-07-16.
- Source: `evidence/2026-07-16-dagger-workflow-barrier-hp2.md`.
- Class: S1/S2 core-parameter and offline-connectivity evidence; no real G1
  collection or speedup claim.
- Facts: `execution_mode=legacy` preserves the callback and old manifest
  shape; `persistent_async` activates one checkpoint/version per outer
  iteration, passes it through every scenario request/result, and activates
  the next checkpoint only after the updater writes it. A real spawned HP-1
  runner crosses the workflow connector. The affected gate passes `345
  passed`, Ruff passes, and the atlas contract check passes.
- Decision: HP-2 is complete. HP-3 owns the Hydra/script connector, reusable
  real distillation runtime, and actual `SharedWeightSync` attachment.
- Limitation: no live env/teacher/student persistence, throughput, Motrix, or
  physical-quality claim follows from HP-2.

## E36: DAgger Hydra Connector HP-3a

- Date: 2026-07-16.
- Source: `evidence/2026-07-16-dagger-hydra-connector-hp3a.md`.
- Class: S0/S2 config and entrypoint connectivity; no real runtime claim.
- Facts: the Hydra owner defaults to `legacy`; OFF forwards only the existing
  scenario callback; ON requires an injected service factory, forwards only
  the service, records the mode, and closes the service. Missing production
  factory fails closed. The affected distill group passes `70 passed` and Ruff
  passes.
- Decision: HP-3a is complete. HP-3b must replace the injection-only production
  gap with the real runtime owner before ON commands are authorized.
- Limitation: no real env/model reuse, shared weights, throughput, or policy
  behavior is proven.

## E37: DAgger Shared Student Runtime HP-3b1

- Date: 2026-07-16.
- Source: `evidence/2026-07-16-dagger-shared-student-hp3b1.md`.
- Class: S1/S2 shared-memory and spawned-runtime connectivity; no G1/live
  collection claim.
- Facts: actual `SharedWeightSync` publishes versions 1 and 2 from two tiny
  checkpoints; the same spawned worker observes exact weight sums 3 and 9 in
  one resident model. Shape drift fails before publication. The impact group
  passes `53 passed`; the final split isolation gate passes `372 + 1` tests;
  Ruff and atlas validation pass.
- Decision: HP-3b1 is complete. HP-3b2 owns real teachers/envs and scenario
  dataset collection before production ON wiring.
- Limitation: no real checkpoint family, G1 resource, throughput, or physical
  behavior is proven.

## E38: DAgger Resource Lifecycle HP-3b2a

- Date: 2026-07-16.
- Source: `evidence/2026-07-16-dagger-resource-lifecycle-hp3b2a.md`.
- Class: S1 fake-resource lifecycle; no G1/MuJoCo claim.
- Facts: complete owner identity keys cache resources; role strings do not.
  Every request resets command/done/transition-age state. Reuse, init, reset,
  error, and close counters are deterministic; normal and exceptional cleanup
  close each resource once. Tests pass `2 passed` and Ruff passes.
- Decision: HP-3b2a passes. Dataset differential is the next gate.
- Limitation: real env/teacher construction and collector semantics remain
  unconfirmed.

## E39: DAgger Dataset Differential HP-3b2b

- Date: 2026-07-16.
- Source: `evidence/2026-07-16-dagger-dataset-differential-hp3b2b.md`.
- Class: S1 deterministic collector/data differential; no MuJoCo claim.
- Facts: legacy self-reset and persistent pre-reset role/transition paths match
  schema dimensions, labels, intent/scenario/transition-age schedules, teacher
  actions, and teacher identity. Each resets exactly once. The collector impact
  group passes `17 passed`.
- Decision: HP-3b2b passes; production factory wiring may proceed.

## E40: DAgger Production Persistent Runtime HP-3b2

- Artifact: `evidence/2026-07-16-dagger-production-runtime-hp3b2.md`.
- Owner route: OFF-default Hydra/script connector -> persistent runtime ->
  exact-identity G1 resource cache -> existing collectors/data owner.
- Differential: role/intent/scenario/age/teacher-action semantics pass with
  exactly one reset per legacy or persistent request.
- Live sequence: one worker PID `62266`, weight version `1`, four requests,
  student init `1`, teacher/env init `2/2`, reset `4`, errors `0`.
- Cleanup: final teacher/env close counters are `2/2`; both exact resources
  were reaped after the walk/stand/transition/walk sequence.
- Regression: IPC/runtime `59 passed`; distill/workflow/config/script
  `419 passed`; fail-closed summary validation `2 passed`; Ruff and atlas pass.
- Decision: HP-3b2 passes connectivity and lifecycle. Control returns to the
  user before bounded persistent training; no speedup or policy-quality claim.

## E41: DAgger Performance Metrics Contract HP-4a

- Artifact: `evidence/2026-07-16-dagger-performance-contract-hp4a.md`.
- Owner: `src/unilab/algos/torch/distill/performance.py`; generic trace events
  remain separate and runtime owners are not instrumented in HP-4a.
- Contract: required DAgger run/request/checkpoint/teacher/config identity,
  canonical seconds/count stages, rows/second, outcome and cleanup state.
- Fail closed: bad hashes/counts/durations, mode/version mismatch, run/request
  identity drift, missing stages, incompatible duplicates, and persisted-rate
  drift.
- Evidence: test-first import failure, then focused `16 passed`; runtime impact
  `39 passed`; entry/config `70 passed, 250 deselected`; Ruff and atlas pass.
- Decision: schema is `implemented-not-integrated`. Return control before
  runtime connectors, Gate 0B, HP-4b, or any performance claim.

## E42: Persistent Worker Metrics Connector HP-4a2a

- Artifact: `evidence/2026-07-16-dagger-worker-metrics-connector-hp4a2a.md`.
- Boundary: worker emits identity-free observations; parent config/checkpoint
  identity and JSON persistence remain HP-4a2c.
- Envelope: schema version `1`; stages are exactly `weight_sync`,
  `artifact_write`, and request-wide `total_elapsed`.
- Fake clock: durations `0.1/0.25/4.0`; row counts `0/4/4`; role env steps `2`,
  transition env steps `3`; cleanup remains `pending`.
- Compatibility: existing flat result metrics remain unchanged.
- Evidence: focused `18 passed`, impact `30 passed`, factory `2 passed`, Ruff
  passes. A mistyped no-test command is explicitly excluded from evidence.
- Decision: HP-4a2a passes; HP-4 integration is partial and HP-4a2b/2c remain
  blocked. No runtime artifact or performance claim exists.

## E43: Collector Metrics Connector HP-4a2b

- Artifact: `evidence/2026-07-16-dagger-collector-metrics-connector-hp4a2b.md`.
- Owner stages: teacher/student inference, direct env step, and tensor packing;
  cached reset/resource timing is explicitly excluded.
- Default isolation: `performance_clock=None` emits no performance metadata.
- Role golden: durations `2/2/1/3`, rows `4/4/0/4`, env steps `0/0/1/0`.
- Transition golden: durations `4/4/3/5`, rows `16/8/0/8`, env steps
  `0/0/3/0`; double teacher rows reflect two actual forwards.
- Transport: observations use copied dataset metadata and worker validates
  schema version and exact stage order before pass-through.
- Evidence: focused `23 passed`, full collector contract `80 passed`, runtime
  impact `33 passed`; Ruff passes.
- Decision: HP-4a2b passes S1/S2. Parent identity/artifact, cleanup-final,
  resource/reset and live timing remain blocked; no performance claim.
- Limitation: real resource construction and nondeterministic simulator values
  remain unconfirmed.

## E44: Parent Metrics Artifact HP-4a2c

- Artifact: `evidence/2026-07-16-dagger-parent-metrics-artifact-hp4a2c.md`.
- Owner route: resolved cfg + all role teacher hashes -> run context -> exact
  worker observations -> parent request/checkpoint/version identity -> atomic
  run-local JSON -> reload -> manifest path/hash/count.
- Resume: completed-iteration artifacts validate identity/hash/count and append
  idempotently; crash-mid-iteration and pre-contract completed runs fail closed.
- OFF isolation: legacy passes no context and writes no metrics artifact; the
  default remains `legacy`.
- Evidence: 2c1 `22 passed`; 2c2 `38 passed`; final affected `373 passed`;
  distinct-two-teacher connector `2 passed`; Ruff, compile, and atlas pass.
- Decision: HP-4a2c passes S1/S2/S3. Reset/resources, cleanup-final, Gate 0B,
  live A/B, bottleneck verdict, speedup, and policy quality remain unconfirmed.

## E45: Gate 0B A/B Identity Freeze

- Artifact: `evidence/2026-07-16-dagger-gate0b-identity-freeze.md`.
- Assets: RT-10 parent/final student, both teachers, and both reusable role
  datasets exist and their recomputed hashes agree with recorded identities.
- Compose: legacy/persistent candidate configs match on parent, roles,
  scenarios, seed `1`, CPU, four envs, 128 rows/scenario, one iteration, and
  eight updates; only execution mode and output directory differ.
- Blockers: legacy writes no structured metrics; aggregation/learner/checkpoint
  stages are not connected; final cleanup is not persisted; worktree is not an
  immutable commit/bundle.
- Decision: Gate 0B is BLOCKED. No HP-4b command ran and no performance claim
  follows. A separately authorized measurement-symmetry step is required.

## E46: HP-4a2d Measurement Symmetry Repair

- Artifact: `evidence/2026-07-16-dagger-measurement-symmetry-hp4a2d.md`.
- Legacy: formal scenario collection now emits exact mode-specific request
  stages while integer callbacks without context remain artifact-free.
- Workflow/learner: aggregation, batch staging, forward, backward, optimizer,
  and checkpoint save are measured at their actual owners and enriched with
  workflow identity.
- Cleanup: both modes append a post-close cleanup record; persistent reports
  require worker PID and resource counters, then atomically refresh manifest
  metrics path/hash/count.
- Evidence: HP-4a2d1 `238 passed`; owner golden `25 passed`; final affected
  `312 passed, 8 skipped`; Ruff passes.
- Decision: HP-4a2d passes S1/S2/S3. No live timing or training ran. Gate 0B
  remains blocked until a separately authorized rerun freezes immutable source
  identity; HP-4b and speedup claims remain unauthorized.

## E47: Gate 0B Rerun Pass

- Artifact: `evidence/2026-07-16-dagger-gate0b-rerun-pass.md`.
- Raw identity manifest SHA-256:
  `2f53362f04ff41d63049004a629410f79de4116ba7983afe240dd4c64e3df1d0`.
- Deterministic source bundle SHA-256:
  `b75f100e212d7edfe09d3c5920918265eafb12eb5cc3c41b3d4d664104d0e779`;
  740 files and identical hash across two generations.
- Seven canonical asset hashes match E45. Read-only compose differs only by
  execution mode and unique run directory; shared config hash is `d6e047...`.
- Eight balanced, unique-output runs and their exact environment/argv/overrides
  are frozen but not executed. Fresh affected suite: `312 passed, 8 skipped`.
- Decision: Gate 0B rerun passes S0/S3. HP-4b remains a separate human decision;
  no training, simulator, server mutation, or performance claim occurred.

## E48: HP-4b Frozen Preflight Blocked

- Artifact: `evidence/2026-07-16-hp4b-frozen-preflight-blocked.md`.
- Command: `uv run python /private/tmp/hp4b_frozen_preflight.py` from E47's
  extracted cwd.
- Runtime fact: package build exits 1 because frozen `README.md` is absent;
  `pyproject.toml:9` requires `readme = "README.md"`.
- Root cause: E47's deterministic source scope omitted one package-build input.
- Isolation: all eight output directories remain absent; no env, collection,
  update, A/B timing, server mutation, or HP-4c action occurred.
- Decision: current Gate 0B executable identity and HP-4b are BLOCKED. Do not
  copy the mutable README into the frozen cwd; regenerate and reverify a new
  bundle under separate authorization.

## E49: Gate 0B Bundle Repair Pass

- Artifact: `evidence/2026-07-16-gate0b-bundle-repair-pass.md`.
- Repair: replace E47's fragile allowlist with all 1241 Git-visible files;
  explicitly require README, LICENSE, pyproject, uv.lock, and `src/unilab`.
- r2 bundle SHA-256: `f7d87a155462955efb300fff6f369fad38886faae6c6d11dc4cf1abca77ac632`;
  identical across two generations.
- r2 identity manifest SHA-256:
  `256da8cf279b7283144565005731e4e94c0d8ab1ac56c27d019dfd5cf00732ab`.
- Frozen-cwd evidence: uv build/import succeeds, 1241 source and seven external
  asset hashes match, G1 XML loads with nq/nv/nu 36/35/29, compose symmetry
  passes, and the r2 HP-4b output root remains absent.
- Decision: Gate 0B executable freeze is restored to PASS. Zero A/B runs
  started; HP-4b execution remains a separate human gate.

## E50: HP-4b Order 1 Schema Block

- Artifact: `evidence/2026-07-16-hp4b-order1-schema-blocked.md`.
- Frozen route: order 1 legacy r1 from E49's r2 cwd; exit 1 during cumulative
  aggregation.
- Runtime fact: role scenario datasets lack `scenario_labels`, while
  `walk_to_stop` has them; parent and new cumulative sources both have mixed
  transition-field presence.
- Owner boundary: `workflow.py` carries source scenario identity, but `data.py`
  fail-closed presence validation runs before source annotation can normalize
  the cumulative inputs.
- Partial evidence: three collections and 21 request records complete;
  aggregation, learner, checkpoint, cleanup, and orders 2-8 are absent.
- Decision: HP-4b is BLOCKED at order 1. No speed or HP-4c claim. An explicitly
  authorized schema-owner repair and new freeze are required before rerun.

## E51: HP-4b Fork Scenario Identity Repair

- Artifact: `evidence/2026-07-16-hp4b-fork-scenario-identity-repair.md`.
- Root cause: `fork_workflow_run()` discarded `scenario` and
  `preserve_row_role_labels` while reconstructing completed-parent cumulative
  sources; the existing data-owner annotation route therefore had no explicit
  scenario identity with which to upgrade legacy role rows in memory.
- Repair: preserve both `WorkflowDatasetSource` fields at the workflow fork
  owner. Strict merge validation, collectors, scripts, scenario semantics, and
  existing `.pt` artifacts remain unchanged.
- Evidence: red `KeyError: 'scenario'`; focused owner chain `3 passed`; affected
  suite `288 passed, 8 skipped`; Ruff passes. The fork fixture also proves every
  parent source hash is unchanged.
- Decision: bounded schema repair passes. E49's source freeze is stale after
  this code change, so Gate 0B and HP-4b remain BLOCKED pending separately
  authorized refreeze and rerun. No server or frozen-r2 mutation occurred.

## E52: Gate 0B Refreeze r3 Pass

- Artifact: `evidence/2026-07-16-gate0b-refreeze-r3-pass.md`.
- Determinism: two generations produce the same 1244-file bundle SHA-256
  `f66ab818...7191` and source-manifest SHA-256 `69ce41e3...f87b`.
- Identity: r3 manifest SHA-256 `1f9e447c...87d7`, state
  `FROZEN_NOT_EXECUTED`, `execution_authorized=false`; seven external artifact
  hashes and the balanced eight-run workload remain fixed.
- Frozen cwd: `/private/tmp/unilab-hp4b-f66ab818`; build succeeds, import is
  asserted to load frozen source, all source/asset hashes match, G1 XML reports
  36/35/29, compose differs only by allowed fields, and output root is absent.
- Evidence: frozen-cwd affected suite `312 passed, 8 skipped`.
- Decision: Gate 0B r3 passes. Zero HP-4b runs started; execution remains a
  separate human gate and must use only the r3 identity/cwd/order.

## E53: HP-4b r3 Order 1 Execution-Environment Block

- Artifact: `evidence/2026-07-17-hp4b-r3-order1-exec-env-blocked.md`.
- Attempt: r3 order 1 legacy from `/private/tmp/unilab-hp4b-f66ab818`.
- Runtime fact: nested frozen `uv run` selects the default
  `/Users/chengyuxuan/.cache/uv` because outer `--cache-dir` is not inherited;
  cache initialization exits 2 before Python/Hydra.
- Root boundary: r3 freezes source/config/assets/workload but omits the formal
  nested dependency/cache environment identity. E52's direct preflight did not
  exercise this exact subprocess boundary.
- Isolation: formal A/B output root absent; zero manifest/dataset/aggregate/
  checkpoint/metrics/cleanup artifacts; orders 2-8 absent; no simulator or
  training ran.
- Decision: Gate 0B executable identity and HP-4b are BLOCKED. A separately
  authorized execution-env repair/refreeze is required; no retry occurred.

## E54: Gate 0B Execution Environment r6 Pass

- Artifact: `evidence/2026-07-17-gate0b-execution-env-r6-pass.md`.
- Accepted identity: r6 SHA-256 `cbf054a8...b778`; r3 source bundle/cwd remain
  unchanged. r4/r5 are explicitly rejected candidates.
- Environment: absolute uv launcher/hash/version; explicit venv/provider,
  cache, no-sync, frozen PYTHONPATH, and progress variables; 171-package
  provider snapshot SHA-256 `dfa668fd...6bd`.
- Exact nested facts: provider snapshot live match; nested import resolves the
  provider venv and frozen `unilab`; nested `scripts/train_distill.py --help`
  exits 0; preflight artifact SHA-256 `b135a94c...92f4`.
- Isolation: r6 formal output root absent and `training_started=false`.
- Decision: Gate 0B r6 passes. HP-4b remains separately authorized; no A/B run
  or server mutation occurred.

## E55: HP-4b r6 Order 1 Teacher Contract Block

- Artifact: `evidence/2026-07-17-hp4b-r6-order1-teacher-contract-blocked.md`.
- Runtime route: r6 nested environment succeeds; three 128-row scenarios and a
  1024-row transition-aware aggregate are created.
- Failure: learner preflight rejects a 98-D teacher checkpoint against composed
  `teacher.obs_dim=99` before trainer/optimizer construction.
- Owner: workflow YAML overrides student to 98 but omits teacher; generic
  `config.yaml` supplies 99 while both task role specs/checkpoints are 98.
- Partial evidence: 21 request records; no learner stages, checkpoint,
  cleanup-final, acceptance, or orders 2-8.
- Decision: HP-4b remains BLOCKED. A separately authorized config-owner repair
  and new freeze are required; no override or retry occurred.

## E56: Workflow Teacher Config Repair And r7 Refreeze

- Artifact: `evidence/2026-07-17-workflow-teacher-config-r7-pass.md`.
- TDD: workflow compose regression fails 99 vs 98, then passes after the Hydra
  workflow owner declares `teacher.obs_dim=98`; generic default remains 99.
- Real contract: walk and stand task specs/checkpoint actor inputs are all 98-D
  and both production guards pass.
- Tests: affected suite `313 passed, 8 skipped`; Ruff passes.
- r7: deterministic 1248-file bundle SHA-256 `3ae830b2...4819`; identity
  SHA-256 `9b180b46...9718`; frozen cwd `/private/tmp/unilab-hp4b-3ae830b2`.
- Frozen evidence: build, source/assets, XML 36/35/29, compose symmetry, real
  teacher probe, exact nested env/import/help, and frozen 313+8 suite pass.
- Decision: Gate 0B r7 passes with output absent and execution false. HP-4b
  remains a separate human decision; r6 partial output is not resumed.

## E57: HP-4b r7 Order 1 Acceptance Oracle Block

- Artifact: `evidence/2026-07-17-hp4b-r7-order1-acceptance-oracle-blocked.md`.
- Formal run: order 1 legacy exits 0; complete manifest, 1024-row aggregate,
  16 actual updates, checkpoint, 28 metrics records, and cleanup persist.
- Oracle failure: external harness incorrectly requires raw legacy role files
  to contain scenario labels; E51 assigns scenario identity through source
  mappings and validates it on the aggregate.
- Runtime proof: raw walk/stand retain role schema, transition is native, and
  aggregate scenario counts are 384/384/256.
- Isolation: acceptance absent; orders 2-8 absent; no rerun or speedup claim.
- Decision: HP-4b and final Gate 0B acceptance identity are BLOCKED pending a
  separately authorized, frozen acceptance-oracle repair.

## E58: Acceptance Oracle v2 And Existing Order 1 Pass

- Artifact: `evidence/2026-07-17-hp4b-acceptance-oracle-v2-order1-pass.md`.
- Frozen oracle SHA-256 `9e62b678...b631`; contract SHA-256
  `d46aefd4...a6a2`; both bind r7 identity `9b180b46...9718`.
- Semantics: raw roles validate role/hash/count without scenario fields;
  transition validates native fields; aggregate validates complete 384/384/256
  scenario identity; run validates checkpoint/28 records/cleanup/lifecycle.
- Existing order 1 acceptance SHA-256 `5096ac20...adcf`, `accepted=true`.
- Attestation: seven tracked training artifact hashes unchanged,
  `training_rerun=false`, order 2 absent.
- Decision: oracle repair and order 1 pass. HP-4b remains partial with orders
  2-8 absent; resume requires separate authorization and must not rerun order 1.

## E59: HP-4b r7 Order 2 Persistent Output Directory Block

- Artifact: `evidence/2026-07-17-hp4b-r7-order2-persistent-output-dir-blocked.md`.
- Preflight: r7 identity/oracle hashes and all seven immutable order-1 artifact
  hashes match; order 1 remains accepted and is not rerun.
- Runtime: with shared memory available, persistent order 2 reaches the spawned
  G1 worker and first scenario collection, then `torch.save` fails because
  `datasets/dagger_iteration_1` was never materialized.
- Owner: workflow constructs `iteration_dir` but does not create it; legacy
  script callbacks create their own parent while the real persistent worker
  writes the request path directly. Test `_write()` helpers mask this gap.
- Isolation: partial manifest is `BOOTSTRAP_COMPLETE`; no dataset, aggregate,
  checkpoint, metrics, cleanup-final, oracle output, or acceptance. No matching
  child process remains, and orders 3-8 are absent.
- Decision: HP-4b is BLOCKED at order 2. Repair the workflow materialization
  boundary, add a persistent pre-dispatch regression, refreeze source/output
  identity, and return for separate execution authorization.

## E60: Workflow Materialization Repair And r8 Refreeze

- Artifact: `evidence/2026-07-17-workflow-materialization-repair-r8-refreeze-pass.md`.
- RED/GREEN: spawned no-mkdir collector fails on the absent request parent,
  then passes after `workflow.py` materializes `iteration_dir` before dispatch.
- Scope: one workflow-owner line and one regression boundary; scripts, worker,
  data, method, teacher, checkpoint, workload, and oracle semantics unchanged.
- Tests: focused `1 passed`; workflow `18 passed`; persistent/IPC `40 passed`;
  full affected suite `493 passed`; Ruff and diff check pass.
- r8: two equal bundle generations SHA-256 `ea1d4f7a...b25e`, 1252 files,
  identity SHA-256 `0dc04b35...240e`, frozen cwd
  `/private/tmp/unilab-hp4b-ea1d4f7a`.
- Preflight: source/assets/build/provider/import/help/XML/compose/two real teacher
  contracts/frozen 493-test suite pass; oracle v2 is bound before execution.
- Isolation: `training_started=false`; r8 output root absent. A future formal
  A/B requires separate authorization and starts at r8 order 1, not r7 order 2.

## E61: HP-4b r8 Eight-Run A/B Complete, Timing Partial

- Artifact: `evidence/2026-07-17-hp4b-r8-eight-run-ab-complete-partial-timing.md`.
- Execution: exact frozen orders 1-8 complete without retry; eight train commands
  exit zero and eight oracle v2 outputs report `accepted=true`.
- Semantics/lifecycle: all runs have 28 records, 16 updates, 1024-row aggregate,
  equal 384/384/256 scenario signatures, complete checkpoints/cleanup; persistent
  counters are 3 requests, 3 resets, 2 resource init/close, 2 cache hits, 0 errors.
- Raw e2e: legacy `[0.617251, 0.489405, 0.458731, 0.593628]`; persistent
  `[0.589469, 0.543215, 0.667084, 0.562266]` seconds.
- Medians: request total 0.396478 legacy vs 0.255483 persistent; complete e2e
  0.541516 legacy vs 0.575867 persistent; persistent cleanup median 0.176112.
- Uncertainty: paired legacy/persistent ratios `[1.047, 0.901, 0.688, 1.056]`
  cross 1; range 0.368 and sample stdev 0.172. No stable speedup claim.
- Decision: HP-4b execution/semantic/lifecycle/timing-artifact gates pass, but
  performance conclusion is PARTIAL. Return control before separately gated
  HP-4c/HP-5.

## E62: HP-4c Finds No Recurring Owner

- Artifact: `evidence/2026-07-17-hp4c-no-recurring-owner-two-iteration-discriminator.md`.
- Cleanup: median 0.176112 s, CV 5.57%, median e2e share 30.03%; stable but a
  composite `scenario_collector.close()` interval that occurs once per invocation.
- Request residual: median 0.154255 s, CV 7.74%, median e2e share 26.07%; walk
  and stand consume 0.119814/0.030430 s on two cache misses, while the warm
  transition request consumes only 0.002251 s with both identities hit.
- Source trace: `PersistentResourceCache.acquire/run_request` owns cold
  teacher/env materialization and reset; formal cleanup wraps runtime/process/
  IPC/resource/weight-sync close after all outer iterations.
- Verdict: `NO_HP5_OWNER`. Stable material costs are once-per-worker/invocation,
  while confirmed recurring residuals are negligible.
- Next discriminator: one newly frozen legacy/persistent pair at two outer
  iterations with an iteration-aware oracle; no timer/code change unless that
  pair fails to distinguish cold from warm behavior.

## E63: r9 Order 1 Complete, Oracle v3 Blocked

- Artifact: `evidence/2026-07-17-hp4c-r9-order1-oracle-v3-blocked.md`.
- Freeze: r9 identity `894e1d30...83f7`, oracle v3 `44175d63...d821`,
  unchanged r8 source bundle `ea1d4f7a...b25e`; exact no-training preflight
  passes before execution.
- Runtime: legacy order 1 exits zero with two complete iterations, 55 timing
  records, aggregate rows `1024 -> 1408`, checkpoint lineage intact, and actual
  updates `16 -> 24`.
- Blocker: oracle v3 wrongly requires an explicit null legacy
  `input_weight_version` key and wrongly freezes actual updates as `16 -> 16`.
  Source proves the key is persistent-only; replay-budget auto-expansion proves
  the second iteration legitimately executes 24 updates.
- Diagnostic: a temporary corrected oracle accepts all existing order-1
  artifacts. Formal v3 acceptance remains absent; training was not rerun.
- Isolation: persistent order 2 and its output/log/acceptance remain absent.
  A separately authorized oracle v4/amendment must accept existing order 1 and
  return control before order 2.

## E64: Oracle v4 Accepts Immutable Existing Order 1

- Artifact: `evidence/2026-07-17-hp4c-oracle-v4-order1-pass.md`.
- Freeze: oracle v4 `9acbbef2...83a0`; amendment
  `3d58eb5d...4c7c`; both bind immutable r9 identity `894e1d30...83f7`
  while preserving v3 and its contract unchanged.
- Corrected contract: legacy version fields are optional/absent; timing versions
  are null. Configured eight updates are a floor; transition replay expansion
  yields valid actual counts `16 -> 24`.
- Acceptance: existing order 1 reports `accepted=true`, two iterations, 55
  metrics, aggregates `1024 -> 1408`, and complete semantic/checkpoint checks.
- Immutability: recursive before/after inventories cover all 16 training files
  and have identical SHA-256 `048923de...41c2`; `training_rerun=false`.
- Isolation: persistent order-2 output/log/acceptance remain absent. Resume
  requires separate authorization and must not rerun legacy order 1.

## E65: r9 Two-Iteration Amortization Confirmed

- Artifact: `evidence/2026-07-17-hp4c-r9-two-iteration-amortization-pass.md`.
- Resume: exact preflight `22a14939...24cb`; persistent order 2 and oracle v4
  pass; execution complete `886ea139...a129`; legacy was not rerun.
- Persistent iteration 1 -> 2: weight version `1 -> 2`, updates `16 -> 24`,
  request total `0.315057 -> 0.149815 s` (-52.45%), request residual
  `0.216987 -> 0.008742 s` (-95.97%).
- Lifecycle: iteration 2 adds four cache hits and zero env/teacher init; final
  counters are 6 requests/resets/hits, 2 resource init/close, 0 errors.
  One cleanup is `0.168563 s`, or `0.084282 s` amortized per iteration.
- Limitation: persistent process `3.585368 s` is 20.64% slower than legacy
  `2.971893 s` in this single pair; learner work also grows to 24 updates.
- Verdict: `AMORTIZATION_CONFIRMED`, but `hp5_owner=null` and no stable speedup
  claim. Persistent remains OFF-default; repeated performance benchmarking is
  a separate future decision.

## E66: r10 Repeated Two-Iteration Benchmark Frozen

- Artifact: `evidence/2026-07-17-hp4c-r10-multirep-freeze-pass.md`.
- Identity: r10 `8f14c14c...1185`, oracle v4 `9acbbef2...83a0`, contract
  `9329719d...aed6`, preflight `4d97819a...1fb4`.
- Workload: exact r9 two-iteration semantics, four repetitions per route, order
  `L1,P1,P2,L2,L3,P3,P4,L4`; all eight per-order compose hashes are frozen and
  normalize to shared hash `25c2bb69...0055`.
- Decision gate: process elapsed is primary; stable direction requires median
  paired ratio below 1, persistent median below legacy, and at least 3/4 paired
  ratios below 1. Otherwise `NO_STABLE_SPEEDUP`.
- Preflight: exact source/provider/assets/teachers/compose/oracle/import/help
  contracts pass. Output root and execution logs remain absent;
  `training_started=false`.
- Isolation: execution requires separate authorization; HP-5 and default-on
  promotion remain closed.

## E67: r10 Eight-Run Benchmark Complete, No Stable Speedup

- Artifact: `evidence/2026-07-17-hp4c-r10-eight-run-no-speedup.md`.
- Execution: exact order `L1,P1,P2,L2,L3,P3,P4,L4`; 8/8 commands exit zero
  and 8/8 immediate oracle-v4 checks accept. Execution-complete hash is
  `617b5a52...840f`; no retry or reorder occurred.
- Primary: legacy/persistent medians `2.286095/2.891427 s`; paired P/L ratios
  `0.963548, 1.272740, 1.266714, 1.262870`; median ratio `1.264792`; only 1/4
  is below 1. Verdict: `NO_STABLE_SPEEDUP`.
- Secondary: persistent iteration-2 request total/residual medians fall
  34.69%/93.70%; all four runs add four cache hits and zero env/teacher init;
  median cleanup is 0.153230 s once per process.
- Decision: secondary amortization does not override primary process time.
  `hp5_owner=null`; HP-5/default-on remain unauthorized and persistent remains
  OFF-default. No policy-quality claim is made.

## E68: HP-6a Readiness Blocked by Stale Runtime Status

- Artifact:
  `evidence/2026-07-17-hp6a-readiness-stale-runtime-status-blocked.md`.
- Review fact: `async_runtime.py:10` and `performance.py:7` still claim live
  timing and A/B are absent, contradicting E61/E65/E67.
- Classification: Important source-level audit-status drift; no runtime defect
  is inferred from this finding.
- Stop: source repair was outside HP-6a scope. No test, Ruff, atlas, training,
  or `make test-all` command ran after the first finding.
- Next: separately authorize the two-owner stale-status repair and HP-6a
  restart. HP-6b, contract activation, default-on, commit, and PR remain closed.

## E69: HP-6a1 Runtime Audit-Status Repair Pass

- Artifact: `evidence/2026-07-17-hp6a1-runtime-status-repair-pass.md`.
- Scope: two runtime-owner module docstrings plus equivalent current
  Method-to-Code performance gaps; no behavior/config/contract/route change.
- Facts: current surfaces record E61/E65/E67 runtime timing and A/B,
  `NO_STABLE_SPEEDUP`, OFF-default, no HP-5 owner, and pending HP-6/physical
  acceptance. Structured stale assertions return no hits.
- Verification: both modules compile; targeted Ruff passes; atlas contracts
  pass with 9 runtime modules, 11 method modules, and 6 concept nodes.
- Decision: E68 status drift is repaired. E70 may restart HP-6a; HP-6b and all
  activation/default/PR actions remain closed.

## E70: HP-6a Restart Tests Green, Runtime Atlas Still Stale

- Artifact: `evidence/2026-07-17-hp6a-restart-runtime-atlas-blocked.md`.
- Executable evidence: owner probe 10/10; algorithm 137 passed; script/config
  326 passed; IPC 74 passed and 24 skipped; targeted Ruff passed. The sandbox
  shared-memory failures were invalid environment evidence and passed on the
  exact permission-corrected rerun.
- Cross-file blocker: Runtime Atlas U-RT-06/U-RT-08 still use the missed phrase
  `尚缺` to claim timing/A/B are absent, contradicting E61/E65/E67.
- Correction: E69 remains valid for source docstrings and Method-to-Code, but
  is PARTIAL for whole-Architecture coverage. E70 is BLOCKED.
- Next: bounded U-RT-06/U-RT-08 status repair plus broadened semantic assertion
  and atlas/cross-file checks. Executable source is unchanged, so the 537-test
  E70 evidence can be reused. HP-6b and all activation/default/PR actions stay
  closed.

## E71: HP-6a2 Runtime Atlas Status Repair Pass

- Artifact: `evidence/2026-07-17-hp6a2-runtime-atlas-status-pass.md`.
- Durable RED: the extended atlas checker exits 1 on old U-RT-06 `A/B 尚缺`.
- GREEN: after U-RT-06/U-RT-08 repair, atlas checks pass with 9 runtime
  modules, 11 method modules, and 6 concept nodes.
- Cross-file: zero stale timing/A/B hits in current Runtime/Method-to-Code gaps;
  both cards contain E67, `NO_STABLE_SPEEDUP`, and HP-6; active registry remains
  method v001/training v002 and v003 remains proposal; diff check passes.
- Decision: E69+E71 resolve E68/E70 status drift. Combined with E70's owner
  probe, 537 passed/24 skipped, and Ruff, HP-6a readiness is PASS. Persistent
  remains OFF-default. Return before HP-6b or activation/default/commit/PR.

## E72: HP-6b make test-all Blocked at Ruff

- Artifact: `evidence/2026-07-17-hp6b-make-test-all-lint-blocked.md`.
- Command: exact repository `make test-all` from the isolated worktree.
- Format stage: 57 files reformatted; Ruff safe-fix corrected 15 of 17 issues.
- Blocker: two F841 unused assignments at
  `check_robojudo_unilab_section8_runtime_torque.py:381-382`; Make exits 2.
- Unreached: mypy, pyright, non-slow pytest coverage, and all later production
  conclusions. E70 focused evidence remains valid but cannot substitute.
- Worktree: automatic formatter/fix mutations remain; no manual repair or
  revert was authorized.
- Next: separately authorize the two-line owner repair, mechanical diff review,
  and exact full rerun. Contract/default/commit/push/PR remain closed.

## E73: HP-6b1 Section-8 Lint Repair Pass

- Artifact: `evidence/2026-07-17-hp6b1-section8-lint-repair-pass.md`.
- Change: removed only two unused `main()` assignments from the section-8
  RoboJudo/UniLab diagnostic.
- Evidence: targeted compile and Ruff pass; AST proves helper-local state
  assignments remain in both rollout functions and are absent only from main.
- Decision: E72 F841 blocker is repaired. E74 may review formatter/auto-fix
  mutations and rerun exact `make test-all`; later decisions remain closed.

## E74: HP-6b2 Full Rerun Blocked at Mypy

- Artifact: `evidence/2026-07-17-hp6b2-mypy-blocked.md`.
- Diff review: 429 files AST-equivalent to r8 frozen source; only E73's two
  removed assignments differ; no missing counterpart or parse failure.
- Rerun: 477 files unchanged by format; Ruff passes; mypy reports 20 errors in
  8 files and Make exits 2. Pyright and coverage pytest do not start.
- Owner split: 7 errors in changed/new collector/async/workflow/G1 persistent
  owners; 13 errors in models/playback/data/G1 joystick files AST-identical to
  HEAD. Both sets block the repository gate.
- Next: separately repair branch-owned and baseline type boundaries with
  targeted evidence before another exact full rerun. All later actions remain
  closed.

## E75-E86: Type Closure, Repository Gate, And Runtime Contract Decision

- E75/E76 repair branch-owned and baseline type boundaries with targeted
  mypy/Ruff and affected-suite evidence.
- E77 exposes collector Pyright narrowing gaps; E78 repairs them and reports
  zero targeted Pyright errors.
- E79 reaches non-slow coverage and exposes fourteen G1/Stewart/docs/CLI
  failures. E80-E84 repair the G1 compatibility owner, optional-provider test
  selection, generated support matrix, and temporary-checkout uv isolation.
- E85 closes the exact fourteen-node regression set: 12 passed, 2 expected
  optional-provider skips.
- E86 exact `make test-all` PASS: Ruff/mypy/Pyright green; 1556 passed,
  51 skipped, 256 deselected; 70% coverage.
- Human decision: persistent DAgger integration is complete, promotion is
  deferred, and the default remains `legacy`. `DISTILL-TRAIN-v003` is active;
  E67 `NO_STABLE_SPEEDUP` forbids default-on promotion and leaves no HP-5 owner.
- Open boundaries: RT-10 physical acceptance, optional Motrix runtime,
  slow/S4, height teacher, promoted checkpoint, and explicit diagnostic-only
  labeling of the manual route.

## E87-E90: Mainline Merge And Persistent Live Learner Bottleneck

- E87: local main merge commit `06d31ad6` combines `0abed823` High Speed
  DAgger and `f882431b` HP persistent runtime. The exact merged snapshot passes
  `make test-all`: 1578 passed, 30 skipped, 256 deselected; Ruff, mypy, and
  Pyright pass.
- E88: server artifact
  `/ssd1/cyx/UniLab/logs/distill_workflow/g1_walk_stand_persistent_test01/distillation_metrics.json`
  records `persistent_async`, one collector PID `1127593` across all observed
  scenarios and iterations, workflow PID `1127462`, and weight versions 1/2/3
  for outer iterations 1/2/3.
- E89: iteration 2 spends approximately 26.64 s collecting three scenarios,
  while workflow stages include 515.90 s batch staging, 144.59 s forward,
  165.44 s backward, 12.81 s optimizer, and 2.10 s aggregation. Learner work
  accounts for about 97% of the observed iteration and staging is the largest
  owner.
- E90: `offline.py` rebuilds label index pools from the full Python label tuple
  inside every balanced update, then `_indexed_batch()` transfers indices to
  the dataset device and repeatedly synchronizes them back to CPU for string
  labels. This is code-confirmed; the individual runtime shares remain
  unconfirmed.
- Decision boundary: persistent reuse is runtime-confirmed, but there is no new
  speedup or promotion claim. The proposed advanced performance phase is
  `HP-7`, documented in `plans/dagger_learner_staging_optimization.md`.
  Main-conversation control resumes at three options: A/recommended executes
  HP-7a no-training staging discrimination; B designs cached pools plus a batch
  schedule without implementation; C implements immediately. At the E90
  boundary none was authorized; E91 below records the later authorization and
  local implementation gate. HP-7a's fastest falsifier is that cached pools
  leave staging near 515.90 s, redirecting ownership toward GPU index-select,
  synchronization, or pinned-CPU/GPU-native staging.

## E91: HP-7a Learner-Staging Probe Implementation

- Date: 2026-07-17.
- Scope: local no-training probe implementation; no server CUDA benchmark.
- Owner path: workflow manifest aggregate dataset -> `data.py` loader ->
  `offline.py` pool/sampling helpers -> deploy probe timing/differential.
- Result: six focused probe/sampler tests pass; targeted Ruff format/check and
  mypy pass; CLI help exits zero.
- Confirmed: current and cached benchmark paths produce identical sampled
  indices, quota counts, string labels, and tensor batches on semantic CPU
  fixtures. The probe never constructs a trainer.
- Open: server CUDA substage timing, dataset-scale ratio, and peak allocation.
  At the E91 boundary HP-7a remained partial and HP-7b was unauthorized; E92
  below closes the server discriminator and records the later human decision.

## E92: HP-7a Server Learner-Staging Discriminator PASS

- Date: 2026-07-17.
- Artifact: `/ssd1/cyx/UniLab/hp7a_iteration2_staging.json` over the existing
  iteration-2 aggregate dataset; no learner update was executed.
- Runtime result: current `31.8345 s`, cached candidate `1.3357 s`, ratio
  `23.8338x`, CUDA peak allocation `622215168` bytes.
- Dominant owner: label-pool construction consumes `29.8662 s`, approximately
  `93.8%` of current staging. GPU index-select and H2D are secondary.
- Semantic differential: sampled indices, label counts, string labels, and
  tensor batches are all exactly equal; `pass=true`.
- Verdict: HP-7a PASS. The microbenchmark-supported HP-7b direction is one
  immutable label-index pool construction per loaded cumulative dataset with
  explicit invalidation when dataset identity changes.
- Human decision: authorize HP-7b design only. HP-7c implementation,
  batch-schedule expansion, training, end-to-end speedup claims, promotion, and
  default-on remain unauthorized.

## E93: HP-7b Immutable Label-Pool Cache Design Freeze

- Date: 2026-07-17.
- Scope: design and governance synchronization only; no production code, test,
  config, training, or Architecture modification.
- Semantic object: one immutable CPU `torch.int64` label-to-row-index mapping
  for the active balance key and ordered labels.
- Unique owner: the offline sampler in `offline.py`; `data.py` retains dataset
  schema/loading ownership and no global/process/workflow/IPC cache is allowed.
- Identity/lifetime: constructed from the exact loaded dataset labels inside
  one offline invocation, reused only by that invocation, released on return,
  and rebuilt for every new dataset/iteration/resume/fork or balance identity.
- RNG invariant: pool construction consumes no RNG; the existing per-update
  `torch.randint` and `torch.randperm` order and generator state must remain
  exactly equal. No batch schedule is generated.
- Memory bound: persistent pool payload is `8 * sum(n_k) <= 8N` bytes plus
  `O(K)` headers, CPU-only, with no label copy or per-update schedule retention.
- Acceptance: S1 owner equivalence/memory tests, S2 formal offline integration,
  then frozen HP-7a plus one bounded persistent S3/S4 differential. Local
  staging speedup alone does not authorize an end-to-end or promotion claim.
- Decision: HP-7b PASS. Return control before separately authorizing HP-7c;
  batch scheduling, pinned memory, GPU-native labels, replay/quota/semantic/
  default/promotion changes remain outside the frozen design.

## E94: HP-7c Owner Implementation And Formal Integration PASS

- Date: 2026-07-17.
- Scope: HP-7c1 owner implementation and HP-7c2 local/formal integration; no
  server CUDA probe, bounded training, default, or promotion action.
- Owner implementation: `offline.py::BalancedLabelIndexPools` stores one
  invocation-local CPU/int64 pool tuple bound to the loaded labels, balance key,
  and ordered selected labels. `run_offline_distillation_updates()` constructs
  it once after replay-budget validation and reuses it for every update.
- S1 facts: three updates invoke the builder once; five fixed-seed iterations
  match the rebuild path in sampled indices, counts, and final generator state;
  malformed source membership fails closed; payload is `<=8N`.
- S2 facts: complete distillation/probe and workflow/script affected suites
  report 301 passed. Targeted Ruff, mypy, and Pyright pass.
- Architecture: Method-to-Code and Runtime Atlas now expose the offline cache
  owner and mark bounded live evidence pending. Concept Figure is unchanged
  because method and training semantics did not change. Atlas checker and
  viewer/data contracts pass with the bundled modern Node runtime.
- Exclusions confirmed: no batch schedule, pinned memory, GPU-native labels,
  replay budget, quota, execution default, or promotion change.
- Decision boundary: HP-7c remains partial. Frozen HP-7a production-path CUDA
  rerun and one bounded persistent workflow require separate server authority.

## E95: HP-7c3 Production-Path Server Sentinel PASS

- Date: 2026-07-17.
- Artifact: `/ssd1/cyx/UniLab/hp7c3_production_path.json`.
- Scope: server CUDA production offline path with a no-op trainer; no
  forward/backward/optimizer/checkpoint or bounded workflow was executed.
- Runtime facts: `production_cache_build_count=1` across
  `production_update_count=512`; staging is `2.166843445971608 s` total and
  `0.004232116105413297 s/update`.
- Semantic facts: production sampled-index digest equals the rebuild reference,
  final RNG state is equal, `training_executed=false`, and `pass=true`.
- Interpretation: cache wiring and sampling equivalence are runtime-confirmed
  on the production function. Comparing `2.1668 s` with E92's `31.8345 s`
  suggests about `14.69x`, but this is an inference across different warmup and
  timing boundaries, not a frozen A/B claim.
- Decision boundary: HP-7c remains partial until one bounded persistent workflow
  verifies real learner updates, checkpoint/manifest lineage, memory, staging,
  and end-to-end time. Default-on and promotion remain unauthorized.

## E96: HP-7c3 Gate 0 SSH Authentication Block

- Date: 2026-07-17.
- Authorized scope: no-training server identity/oracle materialization only.
- Discriminator: read-only `BatchMode` SSH to configured alias `SUST_4090`,
  followed only on success by repository path and Git HEAD reads.
- Observed result: the network connection reached the configured host but SSH
  returned `Permission denied (publickey,password)` before the remote command.
- Confirmed absence: no remote repository/artifact bytes were read, no freeze
  JSON/oracle/log/output directory was created, and no Python, environment,
  collection, learner, or training process started.
- Classification: external authentication boundary, not source/config/artifact
  identity failure.
- Stop: do not attempt passwords or alternate hosts automatically. Resume Gate
  0 only through a user-authenticated session or explicitly provided
  non-interactive identity. Gate 1 remains unauthorized.

## E97: HP-7c3 Gate 0 Amendment And Owner-CLI Compose PASS

- Date: 2026-07-17.
- Artifact: `evidence/2026-07-17-hp7c3-gate0-amendment-compose-pass.md`.
- Source identity: observed server HEAD
  `4fd2f67c08bb5372221ee1347561145b27238a75`; frozen runtime-owner mismatch
  list empty.
- Artifact correction: manifest-owned walk/stand datasets resolve under
  `/ssd1/cyx/UniLab/model/teacher/`; teacher and dataset hashes are recorded in
  the evidence artifact.
- Replay decision: the human accepts exact required/effective updates `12320`;
  configured floor remains `512` and production auto-expansion remains active.
- Metrics classification: parent manifest/observed metrics hashes differ, but
  `fork_workflow_run()` does not consume parent metrics. The drift is recorded
  audit-only/non-blocking and is not rewritten as a match.
- Compose: formal `uv run --no-sync train --algo distill` owner CLI exits `0`;
  resolved YAML is 6795 bytes, SHA-256
  `741676aca03cbed11f9ad6e37105216b3acb545b35ebc86690202b2c0798798d`,
  with empty stderr and the frozen workflow fields/98-D teacher-student
  contract.
- Decision: compose boundary PASS. Gate 0 remains PARTIAL pending server
  freeze/oracle materialization and fresh output-absence preflight. Gate 1 is
  not authorized.

## E98: HP-7c3 Gate 0 r5 Identity PASS And Oracle v6 Local Contract

- Artifact: `evidence/2026-07-17-hp7c3-gate0-r5-pass-oracle-v6-local.md`.
- Server r5 identity preflight accepts with no failures, training false, and
  output paths absent; freeze SHA-256 is `eaab2f8a...b822`.
- Review limits r5 to identity evidence: its post-run oracle omitted command,
  dependency/GPU, supervisor, scenario/weight/worker, full metrics/cleanup, and
  telemetry gates.
- Oracle v6 adds those fail-closed clauses and a generated-but-not-executed
  supervisor. Local contract evidence is 3 tests plus Ruff/compile/diff PASS.
- Decision: Gate 0 remains PARTIAL until v6 server materialization/preflight.
  Gate 1 remains closed.

## E99: HP-7c3 Bounded Persistent Workflow PASS

- Artifact: `evidence/2026-07-17-hp7c3-bounded-persistent-pass.md`.
- Frozen r6 oracle accepts one real persistent iteration with 12,320 updates,
  853,504 aggregate rows, checkpoint, 28 successful metric records, and cleanup.
- Wall time is 368.38 s. Staging is 34.3355 s (`0.00278697 s/update`, 9.32% of
  wall); backward is 167.6141 s, forward 131.3816 s, and optimizer 12.9387 s.
- The bounded per-update staging is about 22.3x below E92's current-path
  observation, but this is a cross-workload inference, not a frozen A/B or
  end-to-end speedup claim.
- NVIDIA telemetry contains unrelated PIDs; its 18,264 MiB peak is not
  attributable to the workflow. CPU max RSS is 1,901,596 KiB.
- Decision: HP-7 implementation and bounded integration close PASS. The active
learner bottleneck is now forward/backward. No rerun, promotion, default-on,
or policy-quality conclusion is authorized.

## E100: HP-7 Side-Session Closeout

- Artifact: `evidence/2026-07-17-hp7-side-session-closeout.md`.
- HP-1 through HP-7 engineering scopes are closed; HP-7 ends PASS by E99.
- This does not close RT-10 physical acceptance, student policy quality, or
  persistent promotion/default-on.
- No formal training run is frozen. The main session must choose whether its
  lineage begins at the original parent iteration-3 checkpoint, explicitly
  promotes the r6 sentinel checkpoint, or evaluates r6 first.
- No second run or training command is authorized by this closeout.

## E101: FT-0 Formal Identity Owner Implementation

- Artifact:
  `evidence/2026-07-17-ft0-formal-identity-owner-implementation.md`.
- `formal_identity.py` owns the clean parent-iteration-3 lineage, reviewed
  workload, owner-CLI argv/environment, fresh outputs, source/artifact freeze,
  and generated supervisor/oracle contracts.
- r6/HP-7 sentinel lineage, dirty/missing inputs, invalid workload/mode/device,
  and existing outputs fail closed. Preflight records
  `training_executed=false` and never invokes training.
- Local evidence: 15 focused/regression tests pass; targeted Ruff and mypy pass.
- Decision: owner implementation PASS; FT-0 remains PARTIAL pending deploy
  connector, compose/dependency/GPU integration, and server materialization.
  FT-1 remains closed.

## E102: FT-0 Deploy Connector Integration

- Artifact: `evidence/2026-07-17-ft0-deploy-connector-integration.md`.
- The thin connector observes Git including untracked runtime files,
  owner-CLI Hydra compose, dependency/import identity, GPU identity, reviewed
  hard artifacts, command/output identity, and generated artifacts.
- A file-level temporary-repository preflight accepts with
  `training_executed=false`; the formal run directory remains absent and the
  frozen training argv is never invoked.
- Local connector/owner/HP-7 regression: 18 passed; Ruff and mypy pass.
- Decision: local integration PASS. FT-0 remains PARTIAL pending a reviewed
  formal workload/output spec and authenticated server no-training
  materialization. FT-1 remains closed.

## E103: FT-0 Formal Two-Round Spec Freeze

- Artifact: `evidence/2026-07-17-ft0-formal-two-round-spec-freeze.md`.
- Initial identity: r1, later superseded by r2 in E105.
- Human-selected workload: original parent iteration 3, two added outer
  iterations, aggregate rows `853504/855040`, required/effective schedule
  `[12320, 12352]`, and total `24672` updates.
- The formal owner now stores an iteration-aware schedule; the previous scalar
  representation would have produced the wrong two-round total `24640`; the
  postflight oracle checks each manifest iteration against the frozen schedule.
- Local spec/argv validation reports `training_executed=false`, excludes r6,
  and binds a fresh formal output root.
- Decision: spec freeze PASS. Server materialization must recompute the schedule
  from the real aggregate; FT-1 remains closed.

## E104: FT-0 Aggregate Workload Owner Integration

- Artifact:
  `evidence/2026-07-17-ft0-aggregate-workload-owner-integration.md`.
- The one-line materializer loads the real parent aggregate, reads resolved
  scenario quota/replay fields, and calls the offline replay owner for every
  added outer iteration.
- Observed rows, required/effective schedule, and total are frozen and compared
  with the spec; any mismatch blocks preflight.
- Serialized-dataset fixture plus focused regression: 23 passed; Ruff/mypy pass.
- Decision: local integration PASS. Server materialization remains unexecuted;
  FT-1 remains closed.

## E105: FT-0 r1 Compose Owner Repair And r2 Refreeze

- Artifact:
  `evidence/2026-07-17-ft0-r1-compose-owner-repair-r2-refreeze.md`.
- Server r1 fails Hydra compose with return code 2 before workload observation;
  freeze/preflight report training false. r1 is permanently closed.
- Root cause: CLI-generated overrides preceded passthrough Hydra flags;
  teacher/dataset environment interpolation was also absent.
- Repair: `build_command()` retains route ownership, compose flags precede
  script overrides, and compose/supervisor share the reviewed artifact env.
- Real local compose and 24 focused regressions pass; Ruff/mypy pass.
- Current identity: `plans/formal_dagger_2round_r2.spec.json`; FT-1 closed.

## E106: FT-1 Acceptance Oracle v2 Implementation

- Artifact:
  `evidence/2026-07-17-ft1-acceptance-oracle-v2-implementation.md`.
- The v2 oracle closes checkpoint/aggregate hashes, parent-to-iteration
  lineage, weight/scenario/worker identity, per-iteration metric stages,
  cleanup, metrics hash/count, telemetry, dependency/import, and GPU gates.
- It reads existing r2 artifacts and never executes, repairs, resumes, or
  deletes training output.
- Three semantic fixtures and the focused suite report 41 passed; Ruff/mypy pass.
- Decision: implementation PASS; r2 server acceptance remains PARTIAL pending
  one read-only v2 evaluation. Training must not rerun.

## E107: Formal Fresh Eight-Iteration Gate 0 Design

- Artifact: `evidence/2026-07-17-formal-fresh-8iter-gate0-design.md`.
- Spec: `plans/formal_dagger_fresh_8iter_r1.spec.json`.
- Fresh lineage uses strict-REUSE role data, 20,000 bootstrap updates, and eight
  DAgger iterations with schedule `4096..32768`, total `147456`.
- Gate 0 recomputes the schedule from real role datasets; oracle v2 validates
  fresh bootstrap-to-iteration checkpoint lineage.
- Real compose, role-dataset discriminator, and focused suite report 46 passed;
  Ruff/mypy pass.
- Decision: local Gate 0 integration PASS; server materialization and final
  fresh training remain unexecuted.

## E108: Formal Runtime Status Correction For r2 And Fresh r1

- Date: 2026-07-20.
- L0 source: user-provided server command outputs and the pulled
  `g1_walk_stand_formal_fresh_8iter_20260717_r1.log` in this task conversation;
  the raw server log is not stored in this checkout.
- r2 fact: its Gate 0 preflight returned `accepted=true` and
  `training_executed=false`; its later two-iteration postflight-v2 result also
  returned `accepted=true`. The oracle read existing artifacts and did not
  rerun training.
- Fresh-r1 fact: its Gate 0 preflight returned `accepted=true`; its supervisor
  created `bootstrap_student.pt` and `dagger_iteration_1.pt` through
  `dagger_iteration_4.pt`. The next aggregate/update boundary hit CUDA OOM.
- Decision: r1 is a preserved interrupted identity, not a resume/retry target.
  It cannot be called an eight-iteration completed candidate. Any fresh r2
  must use a new frozen output identity after a separately approved resource
  and workload decision.

## E109: Formal Auto Output Identity Local PASS

- Date: 2026-07-20.
- Artifact: `evidence/2026-07-20-formal-auto-output-identity-pass.md`.
- `FormalDaggerAutoOutputIdentity` is the only new owner for a semantic
  `run_name` to one Gate-0-resolved, lexically time-sorted output stem.
- The deploy connector freezes the resolved paths and metadata; it rejects
  mixing `run_name` with manual `run_dir`/`artifact_dir`, and continues to
  accept explicit legacy manual paths.
- Local evidence: 56 focused tests, Ruff, mypy, and Atlas validation pass.
  No server command, Gate 0 materialization, supervisor, or training ran in
  this step.
- Decision: local control-surface integration PASS. Choosing a fresh-r2
  resource/workload spec and authorizing an authenticated no-training Gate 0
  remain separate human decisions.

## E110: Formal Fresh r2 Local Spec PASS

- Date: 2026-07-20.
- Artifact: `evidence/2026-07-20-formal-fresh-r2-local-spec-pass.md`.
- Approved resource containment: `collect_num_envs=32`; bootstrap/DAgger batch
  stays 512, samples per role stays 65536, eight outer iterations and the
  147456-update schedule stay unchanged.
- The r2 JSON supplies only
  `run_name=g1_walk_stand_formal_fresh_8iter_oom_r2`; it has no manual output
  path. Fixed-clock connector validation derives both output roots and freezes
  the same schedule.
- The owner-derived Hydra compose returns zero with the r2 fields. A direct
  CLI passthrough was intentionally rejected; the validated route is the
  existing `build_command()` owner path.
- Limitation: r1's reported OOM occurs before batch sampling, during complete
  aggregate CUDA load/validation. The collector change is containment only;
  it is not an OOM-fix claim. No server command, Gate 0, supervisor, or
  training ran.
- Decision: local r2 spec PASS. Authenticated server Gate 0 remains a separate
  human-authorized step.

## E111: Formal Fresh r2 Authenticated Gate 0 PASS

- Date: 2026-07-20.
- L0 source: user-provided server materializer result in this task conversation;
  this agent's SSH key was not accepted, so the generated server files were not
  independently reread locally.
- Result: `accepted=true`, `preflight_returncode=0`, and
  `training_executed=false`.
- Frozen auto identity:
  `20260720-140520_g1_walk_stand_formal_fresh_8iter_oom_r2`.
- Frozen output identities:
  `logs/distill_workflow/20260720-140520_g1_walk_stand_formal_fresh_8iter_oom_r2`
  and matching `logs/distill_role_artifacts/...`.
- Generated controls: freeze
  `/ssd1/cyx/UniLab/20260720-140520_g1_walk_stand_formal_fresh_8iter_oom_r2.freeze.json`
  and corresponding preflight JSON both exist by the accepted materializer
  result.
- Decision: the no-training identity/oracle gate is PASS. FT-1 supervisor
  execution, checkpoint production, GPU-memory behavior, and policy quality
  remain unexecuted/unconfirmed and require separate authorization.

## E112: Ordinary `train.sh` Fresh/Resume Launcher Local PASS

- Artifact: `evidence/2026-07-20-train-sh-fresh-resume-launcher-pass.md`.
- `start.sh` remains interactive playback.  New `train.sh` requires an explicit
  `fresh` or `resume` decision; fresh creates a paired time-sorted identity and
  resume requires one manifest-backed run rather than selecting a latest run.
- Local dry-run suite: `5 passed`; shell syntax passes.  A real owner-CLI
  `--cfg job --resolve` probe exits zero with placeholder paths, confirms the
  CLI-owned `training.workflow.enabled=true`, and retains `legacy` when no
  execution-mode option is given.
- No server, collector, learner update, checkpoint, candidate output creation,
  or formal materialization occurred.
- Decision: ordinary launcher control surface PASS.  It does not replace formal
  Gate 0/FT-1 controls or authorize a training command.
