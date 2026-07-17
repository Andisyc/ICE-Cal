# E93 — HP-7b Immutable Label-Pool Cache Design Freeze

Date: 2026-07-17
Status: design PASS; HP-7c implementation unauthorized.

## Scope

Freeze the smallest E92-supported production design without modifying source,
tests, configuration, runtime defaults, or promotion state.

## Frozen Decision

- Semantic object: immutable CPU `torch.int64` pools mapping the active ordered
  balance labels to row indices.
- Unique owner: the offline sampler in `offline.py`; dataset schema/loading
  remains in `data.py` and no global, workflow, IPC, or script cache is allowed.
- Identity: the exact loaded dataset instance, balance key, and ordered selected
  labels inside one offline invocation.
- Lifetime: create once before that invocation's update loop, reuse inside the
  loop, release on return, and rebuild for every dataset/iteration/resume/fork
  or balance-identity change. No cross-invocation lookup or stale fallback.
- RNG: pool construction consumes no RNG; every update retains the existing
  generator, ordered per-label `torch.randint`, and final `torch.randperm` call.
  Sampled indices and final generator state must be exactly equal.
- Memory: one active balance key only; persistent payload
  `8 * sum(n_k) <= 8N` bytes plus `O(K)` headers, CPU-only. Labels are referenced,
  not copied, and no batch schedule is retained.

## Required HP-7c Evidence

1. S1 owner tests: single construction, immutability/device/dtype, exact
   indices/counts/RNG state, fail-closed malformed pools, and `8N` bound.
2. S2 integration: unchanged offline batch/update count, balance diagnostics,
   checkpoint/manifest lineage, and legacy/OFF behavior.
3. S3/S4 bounded evidence: frozen HP-7a CUDA rerun and one bounded persistent
   workflow with staging/end-to-end timing, scenario/label counts, lineage, and
   peak CPU/CUDA memory.

## Exclusions And Stop

No batch-schedule generation, pinned memory, GPU-native labels, replay-budget or
quota changes, training-semantic changes, default execution-mode changes, or
promotion claims. HP-7b stops here. HP-7c requires separate human authorization
against this exact design.
