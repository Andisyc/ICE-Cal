# E91 — HP-7a Learner-Staging Probe Implementation

Date: 2026-07-17
Scope: local implementation and semantic verification only; no server benchmark
or learner update was executed.

## Evidence

- `scripts/deploy/benchmark_distill_learner_staging.py` resolves an aggregate
  dataset from `run_manifest.json -> dagger_iterations[].aggregate_dataset_path`
  or accepts an explicit dataset path.
- The probe measures label-pool construction, balanced sampling, index H2D,
  tensor `index_select`, and Python-label recovery separately. It compares the
  current per-update pool path with a benchmark-local cached-pool/CPU-label
  recovery candidate.
- `offline.py` now exposes owner-local pure helpers for pool construction and
  sampling from pools. The existing `_balanced_batch_indices()` wrapper retains
  its public behavior and remains the formal training path.
- The probe never constructs a trainer and reports `training_executed=false`.

## Verification

- Focused probe and existing sampler tests: `6 passed`.
- Targeted Ruff format/check: pass.
- Targeted mypy: no issues in `offline.py` and the probe.
- Exact CLI `--help`: exit zero.

## Facts

- CPU toy fixtures prove identical sampled indices, quota counts, recovered
  string labels, and selected tensors between the two benchmark paths.
- No CUDA timing or speedup conclusion exists yet. CUDA synchronization,
  dataset-scale latency, and peak allocation remain server-only evidence.

## Next

Run the frozen HP-7a command against the existing iteration-2 aggregate dataset
on an idle GPU or after the active training process releases its GPU. Persist
the JSON output, then return control before HP-7b or any production-path cache.
