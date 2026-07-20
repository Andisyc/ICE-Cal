# `train.sh` Fresh/Resume Launcher

Status: PASS (TL-1 local launcher contract only)  
Date: 2026-07-20

## Decision

`start.sh` remains the interactive playback launcher.  A new root-level
`train.sh` is the small, ordinary DAgger training convenience entrypoint.  It
is not a Formal Gate 0 materializer, does not produce a freeze/supervisor/oracle
identity, and does not alter `DISTILL-TRAIN-v003` semantics.

## Interface Contract

| Mode | Required input | Output identity | Guard |
| --- | --- | --- | --- |
| `fresh` | `--workflow-mode fresh` | one `YYYYMMDD-HHMMSS_<run_name>` stem under paired workflow/role-artifact roots | both paths must be absent; no directory is made by dry-run |
| `resume` | `--workflow-mode resume --resume-run <run_dir>` | exactly the supplied run identity; no timestamp is generated | `<run_dir>/run_manifest.json` must exist; an unpaired artifact root must be explicit |

The launcher always invokes the normal owner CLI, `uv run train --algo distill`.
That CLI alone generates `training.workflow.enabled=true`; the launcher never
passes it through as a duplicate reserved override.  The launcher owns only
the selected `workflow`, mode, run path, and artifact path, and rejects
passthrough overrides for `workflow`, `training.workflow.enabled`,
`training.workflow.mode`, `training.workflow.run_dir`, and
`training.workflow.artifact_dir`.

## Scope

- Root-level `train.sh`, dry-run-only focused tests, and small governance
  records.
- Optional explicit `--execution-mode`; omission preserves the repository
  `legacy` default.
- Console output only; no launcher-managed log redirection.

## Non-Scope

- Formal Gate 0/FT-1, frozen supervisors, retries, automatic resume, server
  execution, policy evaluation, workload/resource changes, default-mode
  changes, or changing DAgger/replay/collection semantics.

## Owner Boundary

- Launcher routing and timestamp convenience: `train.sh`.
- Real CLI/Hydra route: `src/unilab/cli.py::train_main`.
- Workflow lifecycle, manifests, checkpoints, and role artifacts:
  `scripts/train_distill.py::run_single_entry_workflow` and its owners.
- Formal frozen identities remain exclusively owned by
  `src/unilab/algos/torch/distill/formal_identity.py`.

## Acceptance / Stop

1. RED tests first prove the new file is absent.
2. Dry-run tests must show a paired fresh identity, exact explicit resume
   identity, no default execution-mode override, and fail-closed invalid input.
3. Shell syntax and the existing `start.sh` tests must pass.
4. Stop after local evidence.  No server command or training is authorized by
   this plan.

## Local Evidence

- RED: the initial focused suite failed because `train.sh` did not exist.
- GREEN: `bash -n train.sh` and
  `uv run pytest tests/scripts/test_train_sh.py -q` pass (`5 passed`).  The
  tests cover paired fresh identity, explicit resume identity, standard paired
  resume artifact discovery, missing-manifest rejection, and duplicate route
  rejection.
- Runtime route probe: a placeholder-path, CPU-only
  `uv run train ... --cfg job --resolve` exits `0`, resolves
  `training.workflow.enabled: true` through the owner CLI, and keeps
  `execution_mode: legacy` when the launcher omits that explicit option.
- See `evidence/2026-07-20-train-sh-fresh-resume-launcher-pass.md`.  No
  candidate output directory, server process, collector, learner update,
  checkpoint, or log was created by this validation.
