# Formal Fresh Eight-Iteration r2 Implementation Plan

Status: R2-S1 through R2-S3 local PASS; R2-S4 authenticated server Gate 0 is pending separate authorization.

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans`
> or an equivalent bounded inline workflow. This plan does not authorize a
> server Gate 0, supervisor, or training run.

**Goal:** Freeze and locally validate a new fresh eight-iteration DAgger r2
specification with automatic `run_name` output identity and a single
evidence-bounded resource intervention.

**Architecture:** The existing formal-identity owner resolves `run_name` once
at Gate 0; no new runtime owner is introduced. The r2 spec changes only
collector concurrency from 64 to 32. It retains the same role data, seed,
device, DAgger sampling/replay workload, learner batch, and default-OFF
promotion status as fresh r1.

**Tech Stack:** JSON formal spec, `formal_identity.py`, Gate 0 connector,
Hydra compose, pytest.

---

## Approved Decision

The human approved this resource profile:

| Parameter | fresh r1 | fresh r2 | Authority rationale |
| --- | ---: | ---: | --- |
| `training.workflow.collect_num_envs` | 64 | 32 | Contain collector concurrency only. |
| `bootstrap_batch_size` | 512 | 512 (owner-config inherited) | No evidence that bootstrap batch owns the failure. |
| `dagger_batch_size` | 512 | 512 | Preserve replay schedule and avoid an unproven intervention. |
| `dagger_samples_per_role` | 65536 | 65536 | Preserve collection workload and DAgger data semantics. |
| `dagger_iterations` | 8 | 8 | Preserve final formal workload. |
| `bootstrap_updates` | 20000 | 20000 | Preserve bootstrap workload. |

The formal `run_name` is
`g1_walk_stand_formal_fresh_8iter_oom_r2`. Gate 0 must generate its timestamp
prefix; this specification must not contain `run_dir` or `artifact_dir`.

## Evidence-Bounded OOM Interpretation

The r1 OOM was reported at the aggregate load/validation boundary. The current
owner path is:

```text
run_offline_dataset_update(device=cuda:0)
  -> load_distillation_dataset(..., device=cuda:0)
  -> payload tensor .to(cuda:0)
  -> _validate_obs_tensor(... torch.isfinite(tensor).all())
  -> balanced batch sampling / learner update
```

Thus `dagger_batch_size` is downstream of the reported boundary. A local
offline numerical differential confirms that batch 256 would double the
minimum replay schedule from `147456` to `294912` total updates. It is rejected
as an unproven OOM remedy. Reducing `collect_num_envs` is containment, not a
claimed repair for aggregate validation.

## Design Point Register

| design ID | canonical human name | active contract | Concept Figure block | r2 impact |
| --- | --- | --- | --- | --- |
| DISTILL-DP-05 | Student-State DAgger | `DISTILL-METHOD-v001#student-state-dagger`; `DISTILL-TRAIN-v003` | DT-M-05 | Resource-only collector concurrency change; no method, target, replay, or checkpoint-semantic change. |

The Concept Figure and Atlas require no edit: no owner, interface, dataflow, or
method semantic changes in this step.

## Step Map

### R2-S1 / 4: Freeze the reviewed local spec

- Objective: create one r2 JSON spec with `run_name`, no manual outputs, and
  exactly the approved values.
- Files: `note/distillation/plans/formal_dagger_fresh_8iter_r2.spec.json`.
- Test class: secondary contract path.
- Expected evidence: a focused regression fails before the file exists, then
  verifies the name, generated-path rule, resource values, and unchanged
  schedule.
- Stop: any manual output path, unexpected workload value, or invalid
  `run_name` blocks the local spec.

### R2-S2 / 4: Validate the formal owner and Hydra route locally

- Objective: prove the real owner route composes the r2 values without
  materializing or executing the formal workflow.
- Files: `tests/scripts/test_materialize_formal_dagger_gate0.py`.
- Test class: secondary contract path plus owner-to-Hydra compose integration.
- Expected evidence: fixed-clock identity test plus `build_command()`-derived
  `--cfg job --resolve` returns zero and emits the frozen resource values.
- Stop: route-selector parsing failure, nonzero compose, or config drift blocks
  r2 before server work.

### R2-S3 / 4: Record local evidence and current control state

- Objective: update plan/checklist/canvas/evidence without changing active
  method contracts or Atlas.
- Files: this plan, `checklists/current.md`, `task_canvas.md`,
  `evidence/current.md`, dated evidence ledger.
- Expected evidence: fresh test output, lint/type/docs checks, and explicit
  limits that GPU capacity remains live-only.
- Stop: a document claims r2 solves the OOM or authorizes training.

### R2-S4 / 4: Authenticated server Gate 0

- Objective: separately materialize and preflight the reviewed r2 identity.
- Non-scope: this local step must not perform it.
- Future entry condition: human explicitly authorizes Gate 0; GPU is exclusive
  to the run at launch; all frozen source/artifact/config checks pass.
- Future stop condition: any OOM at aggregate load/validation, foreign GPU
  occupancy, mismatch, or preflight failure preserves r2 and returns control.
  No retry/resume or batch adjustment is allowed under r2.

## Local Execution Result

R2-S1 through R2-S3 pass locally. The real spec contains the approved
`run_name`, omits both manual output paths, and the frozen owner identity
derives the expected fixed-clock paths. The owner-to-Hydra compose regression
uses `build_command()` rather than a direct `uv run train` passthrough; it
returns zero with the selected fresh, 32-env, 512-batch configuration.

This is a local config/identity result only. It cannot prove aggregate GPU
capacity, collector peak memory, checkpoint production, or policy quality.

## Acceptance Matrix

| Item | Owner | Evidence | Current status |
| --- | --- | --- | --- |
| `run_name` identity/no manual outputs | formal identity + connector | S1 fixed-clock regression | PASS, E110 |
| Resource values and unchanged replay schedule | formal spec + offline replay owner | S1 numerical differential | PASS, E110 |
| Real owner-to-Hydra composition | CLI/config connector | S2 compose regression | PASS, E110 |
| No server action or training | governance | S0 command record | PASS, E110 |
| Authenticated r2 Gate 0 | deploy connector | S3 server preflight | BLOCKED by separate authority |

## Non-Scope

- No production code change, data-owner refactor, CPU-resident dataset change,
  GPU memory fix, batch-size reduction, output cleanup, resume, retry,
  promotion, default-mode change, server Gate 0, supervisor, or training.
