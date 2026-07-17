# HP-7 Advanced DAgger Learner-Staging Optimization

Status: proposed; no implementation authorized.

Phase ID: `HP-7`. This phase begins only because the new server live run
identifies a stable learner-side owner that E67's earlier short A/B workload
did not expose. It does not reopen persistent default-on promotion.

## Problem

The server persistent run proves collector reuse, but iteration-2
`learner_batch_staging` consumes 515.90 s and is the largest measured owner.
The active training/replay semantics remain governed by `DISTILL-TRAIN-v003`.

## Human Control Options

- A / recommended: execute `HP-7a`, a read-only/no-training microbenchmark
  that separately measures label-pool construction, balanced sampling,
  CPU-to-GPU index transfer, GPU `index_select`, and Python-label recovery.
- B: design the cached label-pool plus batch-schedule optimization from current
  code/runtime evidence, but do not implement it.
- C: implement and verify the optimization immediately. This has the largest
  change surface and is not recommended before the staging owner is split.

Fastest falsifier for the leading hypothesis: if cached label pools leave
staging near the observed 515.90 s, repeated Python full-dataset scanning is
not the dominant owner. The next candidates would be GPU `index_select`,
device synchronization, or a pinned-CPU/GPU-native staging design.

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
remains `PARTIAL` until the frozen command produces CUDA timing on the existing
server aggregate dataset. No HP-7b production optimization is authorized.

### HP-7b / 3: Owner-local design

- Objective: select the smallest optimization supported by Step 1.
- Candidate: cache label index pools and generate deterministic batch indices
  once or in bounded chunks.
- Non-scope: moving dense learner math to CPU or weakening transition replay.
- Expected evidence: deterministic semantic differential and memory bound.
- Stop condition: user accepts or rejects the design before code changes.

### HP-7c / 3: Implementation and formal validation

- Objective: implement only the accepted owner-local optimization.
- Expected evidence: sampler equivalence, replay-contract tests, targeted
  learner metrics, repository gate, and a bounded persistent differential.
- Stop condition: no speedup/promotion claim without fresh end-to-end evidence.

## Current Decision

Option A is authorized. E91 completes the local implementation gate; the next
bounded action is the human-run server HP-7a command. Return with its JSON
artifact before selecting HP-7b. Options B/C, production caching, training,
promotion, and default-on remain unauthorized.
