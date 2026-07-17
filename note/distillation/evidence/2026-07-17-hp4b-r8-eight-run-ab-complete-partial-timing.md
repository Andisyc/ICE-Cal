# HP-4b r8 Eight-Run A/B Complete With Partial Timing Verdict

Date: 2026-07-17

Status: formal execution, semantic acceptance, and lifecycle acceptance `PASS`;
stable end-to-end speedup conclusion `PARTIAL`.

## Frozen Execution

- Identity SHA-256:
  `0dc04b35d7a3ad04b3821372f5f11d30b6eb5d8cabbf780c4798067428e9240e`.
- Bundle SHA-256:
  `ea1d4f7a6acc3a35f9669bbc55c3df681e48100bdbe72e0880def609f5d5b25e`.
- Oracle v2 SHA-256:
  `9e62b678eb02d792c587b2a46ecc7fae1e000b9376d5bfbc229683170fedb631`.
- Execution-complete artifact SHA-256:
  `6c4c34a8ecb10bbccd8939fc56cf8324c6550ed8cb3b81b56b9a4a3eee2c1237`.
- Analysis artifact SHA-256:
  `ffb3a3d52d1b5435af822d0be0f883321163286b2abadb6b2b110d73f24e5246`.
- Attestation SHA-256:
  `8211e0b6859dc37a01344c39c304647f787cd27f580547e699b9809999f5e5c0`.

The exact frozen order completed without retry:

| Order | Route | Rep | Measured end-to-end s | Acceptance SHA-256 |
|---:|---|---:|---:|---|
| 1 | legacy | 1 | 0.617251 | `09717a78...c74a` |
| 2 | persistent_async | 1 | 0.589469 | `f9090535...2149` |
| 3 | persistent_async | 2 | 0.543215 | `90dc31f1...e56a` |
| 4 | legacy | 2 | 0.489405 | `055c1bc5...5974` |
| 5 | legacy | 3 | 0.458731 | `b5096fe9...6fe3` |
| 6 | persistent_async | 3 | 0.667084 | `6e54e077...d54f` |
| 7 | persistent_async | 4 | 0.562266 | `34b0a90f...d69f` |
| 8 | legacy | 4 | 0.593628 | `98d1f5f1...dd83` |

Every training command exited zero and every frozen oracle invocation returned
`accepted=true` before the next order started.

## Identity, Semantics, And Lifecycle

All eight runs have:

- one completed outer DAgger iteration, 16 updates, and a current checkpoint;
- 28 metrics records: 21 request, 6 workflow/learner, and 1 cleanup;
- three 128-row scenario artifacts and one 1024-row aggregate;
- aggregate scenario counts `static_stand=384`, `walk_flat=384`, and
  `walk_to_stop=256`;
- equal role/scenario/transition semantic signatures, including transition age
  `-1..23` and 96 post-switch rows;
- the same parent checkpoint, teacher hashes, seed 1, CPU device, four envs,
  scenario order, quotas, and workload.

Each persistent run uses one input weight version, one resident student, two
exact teacher/env resources, three requests, three resets, two cache hits, zero
request errors, and matching init/close counters. Each legacy run reports the
frozen per-request resource scope. No train, executor, or collector process
remains after completion.

## Timing Definition

For each run:

```text
measured end-to-end
= sum(three request total_elapsed records)
+ aggregation/learner/optimizer/checkpoint stages
+ cleanup
```

Rows/second uses the 384 newly collected rows, not the 1024 cumulative rows.

| Metric, median of four runs | Legacy | Persistent | Persistent relative to legacy |
|---|---:|---:|---:|
| Request total | 0.396478 s | 0.255483 s | -35.56% |
| Workflow/learner total | 0.145038 s | 0.147793 s | +1.90% |
| Cleanup | 0.000000 s | 0.176112 s | resident cleanup only |
| Measured end-to-end | 0.541516 s | 0.575867 s | +6.34% |
| New rows/s | 715.75 | 667.19 | -6.78% |

Raw end-to-end values:

- legacy: `[0.617251, 0.489405, 0.458731, 0.593628]`, range
  `0.158520`, sample stdev `0.077477`;
- persistent: `[0.589469, 0.543215, 0.667084, 0.562266]`, range
  `0.123869`, sample stdev `0.054465`.

The paired legacy/persistent ratios are
`[1.047, 0.901, 0.688, 1.056]`; median `0.974`, range `0.368`, sample stdev
`0.172`. The direction crosses 1, so the result does not support a stable
end-to-end speedup claim. Repetitions 2-4 also have medians `0.489405` legacy
and `0.562266` persistent.

## Decision

HP-4b formal connectivity, semantic equivalence, lifecycle, persistence, and
raw timing evidence pass. The performance verdict remains partial because the
paired direction is unstable and the route-median end-to-end result is not a
speedup. The request collection boundary is faster under persistent execution,
but cleanup and the persistent request residual consume that benefit; this is
an observation, not yet an owner-level bottleneck verdict.

HP-4c and HP-5 do not start automatically. A separate human authorization is
required for HP-4c to determine whether one stable stage owner exists or whether
another discriminator is needed. Policy quality and physical acceptance remain
outside this performance experiment.
