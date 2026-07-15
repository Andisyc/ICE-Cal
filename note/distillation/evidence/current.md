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

- Repeated-reset acceptance is not implemented after the reset-order diagnosis.
- Standing-teacher recovery authority on post-walk states has no structured
  differential evidence.
- Current DAgger may match teacher actions locally without satisfying a repeated
  physical transition gate.
- Generic outer DAgger workflow connectivity is implemented; a walk-to-stop
  transition scenario remains absent.
- Height-control role is not trainable without a qualified teacher.
