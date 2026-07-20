# Formal Fresh r2 Local Spec PASS

Date: 2026-07-20.
Status: local identity/config/replay validation PASS; no server Gate 0 or
training executed.

## Scope

This evidence freezes only a new fresh-r2 specification:

```text
run_name
  -> Gate 0 time-sorted output identity
  -> existing owner CLI / Hydra compose
  -> future server Gate 0, if separately authorized
```

It changes collector concurrency from 64 to 32. It does not change DAgger
semantics, teacher/data identity, batch size, samples per role, replay quota,
outer-iteration count, seed, device, execution mode, default mode, retry,
resume, logging policy, promotion, or physical acceptance.

## Resource Decision And Boundary

| Parameter | r2 value | Evidence-bound interpretation |
| --- | ---: | --- |
| `collect_num_envs` | 32 | Collector-concurrency containment only. |
| bootstrap batch | 512 | Inherited from frozen owner config; unchanged. |
| `dagger_batch_size` | 512 | Unchanged because it is downstream of the reported OOM boundary. |
| samples / iterations | `65536 / 8` | Unchanged formal workload. |
| effective updates | `4096..32768`, total `147456` | Same replay contract as r1. |

Static owner tracing shows:

```text
run_offline_dataset_update(device=cuda:0)
  -> load_distillation_dataset(..., device=cuda:0)
  -> payload tensors .to(cuda:0)
  -> _validate_obs_tensor(torch.isfinite(...))
  -> batch sampling
```

The reported r1 failure belongs before batch sampling. The same offline replay
owner gives batch-512 total `147456` but batch-256 total `294912`; batch
reduction was therefore rejected as an unproven remedy with a known 2x workload
cost.

## Local Evidence

- RED: before the spec existed,
  `test_repository_fresh_eight_iteration_r2_spec_is_resource_scoped_and_composes`
  failed only with `FileNotFoundError` for the intended r2 JSON path.
- GREEN: the same test passed after adding
  `plans/formal_dagger_fresh_8iter_r2.spec.json`.
- The regression verifies: no manual `run_dir`/`artifact_dir`; fixed-clock
  paths from the approved `run_name`; `collect_num_envs=32`; 512 DAgger batch;
  unchanged eight-iteration schedule; and real owner-route Hydra composition.
- A pure local replay differential prints:

  ```text
  batch_512 total = 147456
  batch_256 total = 294912
  ```

- The owner-derived compose uses `build_command()` and returns zero with
  `mode: fresh`, `collect_num_envs: 32`, `bootstrap_batch_size: 512`,
  `dagger_batch_size: 512`, and the reviewed workload fields.

## Stop Conditions

The future server-side Gate 0 must stop before training if source/artifact,
compose, output-absence, device identity, or launch-exclusivity checks fail.
If an r2 execution later OOMs at aggregate load/validation, preserve r2 and
stop immediately: do not resume, retry, or lower batch. The next question then
belongs to the data-owner device-residency path, not this resource spec.

## Limit

This local work proves spec/route compatibility. It cannot prove server GPU
availability, aggregate-device peak, collector/learner memory, checkpoint
creation, training convergence, or physical policy quality.
