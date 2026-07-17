# HP-7 Advanced DAgger Learner-Staging Optimization

Status: HP-7c owner implementation and integration PASS; bounded live gate pending.

Phase ID: `HP-7`. This phase begins only because the new server live run
identifies a stable learner-side owner that E67's earlier short A/B workload
did not expose. It does not reopen persistent default-on promotion.

## Problem

The server persistent run proves collector reuse, but iteration-2
`learner_batch_staging` consumes 515.90 s and is the largest measured owner.
The active training/replay semantics remain governed by `DISTILL-TRAIN-v003`.

## Resolved Human Gate

The earlier A/B/C gate is resolved: HP-7a executed and passed E92, then the
human authorized HP-7b design only. E92 confirms repeated full-dataset pool
construction as the dominant owner. This phase therefore freezes only the
immutable pool cache; transport, label-representation, and schedule-generation
alternatives are excluded rather than carried as candidate design branches.

## Step Map

### HP-7a / 3: Staging discriminator

- Objective: separate label-pool construction, sampling, index transfer,
  tensor selection, and Python label recovery costs.
- Scope: read-only/no-training microbenchmark on an existing cumulative
  dataset.
- Non-scope: trainer updates, replay-budget changes, config/default changes.
- Owner modules: `offline.py`, `data.py`; benchmark artifact only.
- Expected evidence: per-substage latency, device identity, dataset size,
  batch size, update count, and current/cached ratio.
- Stop condition: return control with a falsifiable owner verdict.

Implementation status: E91 passes the local probe implementation gate. The
probe reuses owner-local pool/sampling helpers and proves current/cached index,
quota, string-label, and tensor-batch equality on semantic CPU fixtures. HP-7a
is now `PASS` by E92: server CUDA timing reports `31.835 s` current versus
`1.336 s` cached (`23.83x`), with label-pool construction consuming `93.8%` of
the current path and every semantic differential passing.

### HP-7b / 3: Owner-local design

- Objective: select the smallest optimization supported by Step 1.
- Candidate: construct immutable label-index pools once per loaded cumulative
  dataset and reuse them for every balanced update in that offline invocation.
- Non-scope: batch-schedule generation, pinned-memory transport, GPU-native
  labels, moving dense learner math to CPU, or weakening transition replay.
- Expected evidence: deterministic semantic differential and memory bound.
- Stop condition: user accepts or rejects the design before code changes.

Authorization status: design-only authorized by the human decision recorded in
E92. The design must prefer one immutable pool construction per loaded
cumulative dataset and explicit invalidation when that dataset changes. Do not
add batch-schedule generation, pinned-memory transport, GPU-native labels, or
production code unless separate evidence and authorization require them.

#### Frozen HP-7b Design

**Semantic object and unique owner.** The cache object is
`BalancedLabelIndexPools`: an immutable, CPU-resident mapping from each selected
balance label to one contiguous `torch.int64` row-index tensor. Its only owner
is the offline learner sampler in `src/unilab/algos/torch/distill/offline.py`.
`data.py` continues to own dataset schema/loading and does not own, persist, or
globally register this derived sampler view. IPC, workflow, trainer, and scripts
must not acquire cache authority.

**Dataset identity binding.** One cache object is constructed from the exact
`labels` tuple resolved from one loaded `DistillationTensorDataset` inside one
`run_offline_distillation` invocation. Its identity is the tuple
`(loaded dataset instance, balance_key, ordered selected_labels)`. It is not
looked up by path, weak metadata, row count, or process-global key. Quotas do
not enter pool identity because they affect sampling counts, not row membership;
they remain an input to every sampling call.

**Lifetime and invalidation.** The cache is a local immutable value created
after dataset/balance validation and before the update loop. Every update in
that invocation reuses it. Function exit releases it. A newly loaded cumulative
dataset, a new outer DAgger iteration, resume/fork invocation, changed
`balance_key`, or changed ordered `selected_labels` constructs a new cache.
There is no cross-invocation reuse and therefore no stale-cache fallback. The
dataset and cache must not be mutated in place; an attempted mismatch fails
closed before sampling.

**RNG and sampling equivalence.** Pool construction consumes no RNG. The update
loop must continue to call `_sample_balanced_batch_indices_from_pools()` exactly
once per update with the same CPU `torch.Generator`, ordered labels, quotas,
`torch.randint` calls per label, and final `torch.randperm` call. No index
schedule is prefetched or generated. Acceptance requires exact sampled-index
equality and identical generator state after multiple updates for fixed seeds,
including equal/default quotas and missing-label failure cases.

**Memory bound.** Only the active balance key is cached. Let `N` be dataset rows,
`K` selected labels, and `n_k` the rows in label `k`. Persistent index storage is
`8 * sum_k(n_k)` bytes and must be at most `8N` bytes, plus bounded `O(K)` Python
mapping/tensor-header overhead. The labels tuple is referenced, not copied.
Pool construction may retain no per-update list or batch schedule. Tests must
report computed pool bytes and assert the `8N` payload bound. E92's
`622215168` CUDA peak is a benchmark observation, not cache storage; the cache
remains CPU-only and may not increase persistent CUDA allocation.

**Owner files/modules.** HP-7c may modify only
`src/unilab/algos/torch/distill/offline.py` for the cache object/construction and
the existing sampler loop. `data.py`, workflow, trainer, IPC, Hydra, and
entrypoints are consumers or non-owners and should remain unchanged. Regression
coverage belongs in `tests/algos/test_g1_distillation_contract.py`; the existing
HP-7a probe and its tests remain differential evidence, not a production
connector.

**Acceptance ladder.** Implementation evidence is S1 deterministic owner tests:
single construction per invocation, immutable CPU/int64 pools, exact indices,
counts, RNG state, missing-label failures, and `8N` memory bound. Integration
evidence is S2 formal offline-route tests proving the same batch/update count,
balance diagnostics, checkpoint/manifest lineage, and unchanged OFF path.
Bounded live evidence is S3/S4: rerun the frozen HP-7a CUDA discriminator and
one bounded persistent workflow, recording staging substage time, update count,
scenario/label counts, weight/checkpoint lineage, peak CPU/CUDA memory, and
end-to-end workflow time. Local staging improvement alone cannot establish an
end-to-end speedup or promotion claim.

**HP-7c stop condition.** HP-7b ends after this design, checklist, task canvas,
and evidence ledger agree and documentation checks pass. Do not edit production
code, tests, config, or Architecture in HP-7b. HP-7c requires a separate human
authorization accepting this exact owner/identity/lifetime/RNG/memory contract.
Any request for batch schedules, pinned memory, GPU-native labels, replay-budget
or quota changes, training-semantic changes, default-mode changes, or promotion
returns to a new design gate rather than entering HP-7c.

### HP-7c / 3: Implementation and formal validation

- Objective: implement only the accepted owner-local optimization.
- Expected evidence: sampler equivalence, replay-contract tests, targeted
  learner metrics, repository gate, and a bounded persistent differential.
- Stop condition: no speedup/promotion claim without fresh end-to-end evidence.

HP-7c1 owner result: PASS by E94. `BalancedLabelIndexPools` is an
invocation-local frozen CPU/int64 object. The formal update loop constructs it
once after replay-budget validation and samples from it once per update. Five
fixed-seed updates match the rebuild path in indices, counts, and final RNG
state; malformed membership fails closed and payload obeys `8N`.

HP-7c2 integration result: PASS by E94. Distillation/probe plus workflow/script
suites report 301 passed; targeted Ruff/mypy/Pyright and Atlas contracts pass. No
batch schedule, pinned memory, GPU-native label, replay/quota/default/promotion
change entered the diff.

HP-7c3 live status: PENDING. It requires the frozen HP-7a server discriminator
against the production path and one bounded persistent workflow. Return control
before either server command; local and integration evidence do not establish
CUDA or end-to-end speedup.

## Current Decision

HP-7a passes E92, HP-7b is frozen by E93, and HP-7c1/HP-7c2 pass E94. Return
control before HP-7c3 server evidence. Batch scheduling, pinned memory,
GPU-native labels, training-semantic changes, promotion, and default-on remain
unauthorized.
