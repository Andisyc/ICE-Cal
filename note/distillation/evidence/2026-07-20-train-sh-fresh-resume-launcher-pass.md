# `train.sh` Fresh/Resume Launcher Local PASS

Date: 2026-07-20  
Scope: local launcher control surface only; no server or training execution.

## Result

`train.sh` is a separate ordinary DAgger training launcher.  `start.sh` remains
interactive playback.  The new launcher makes the training decision explicit:

| Request | Resolved identity | Fail-closed boundary |
| --- | --- | --- |
| `--workflow-mode fresh` | paired `logs/distill_workflow/YYYYMMDD-HHMMSS_<run_name>` and `logs/distill_role_artifacts/...` roots | either existing path stops before `uv run train` |
| `--workflow-mode resume --resume-run <run_dir>` | exactly the supplied manifest-backed run, with only its standard paired artifact root inferred | missing manifest, nonstandard root without explicit `--artifact-dir`, or missing artifact root stops before `uv run train` |

No automatic choice of a latest run exists.  `--execution-mode` is optional:
omission leaves the existing Hydra default `legacy`; `persistent_async` is an
explicit user choice.  The launcher does not manage a log file, so normal
stdout/stderr remains visible in the terminal.

## Owner Boundary

- `train.sh` owns only ordinary launch convenience: explicit fresh/resume
  selection, paired path derivation, and route-override rejection.
- `src/unilab/cli.py::train_main` owns `training.workflow.enabled=true`; the
  launcher does not send that reserved override.
- `scripts/train_distill.py::run_single_entry_workflow` retains workflow,
  manifest, role-artifact, checkpoint, and resume semantics.
- `formal_identity.py` remains the sole owner of frozen formal Gate 0 identity;
  this launcher neither creates nor substitutes a formal freeze/supervisor.

## Local Verification

1. RED: the new test suite initially failed with `train.sh: No such file or
   directory` (four expected failures).
2. GREEN: `bash -n train.sh` plus
   `uv run pytest tests/scripts/test_train_sh.py -q` returns `5 passed`.
3. Owner-CLI compose: with placeholder teacher/dataset paths and CPU device,
   `uv run train --algo distill --task g1_walk_flat --sim mujoco ... --cfg job
   --resolve` returns `0`.  The resolved config shows
   `training.workflow.enabled: true`, the requested fresh paths, and default
   `training.workflow.execution_mode: legacy` when no mode is supplied.

This evidence is S1/S2 launcher-and-compose evidence only.  It proves no GPU
allocation, collection, update, checkpoint, candidate output creation, server
action, or formal materialization.

## Stop Boundary

This PASS does not authorize a server run, resume, retrain, formal Gate 0/FT-1,
or policy evaluation.  A formal frozen identity must still use its matching
materializer and supervisor.  The user chooses any future ordinary `train.sh`
command and its workload/resource overrides explicitly.
