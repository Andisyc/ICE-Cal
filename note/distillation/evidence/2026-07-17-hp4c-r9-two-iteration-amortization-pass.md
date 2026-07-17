# HP-4c r9 Two-Iteration Amortization Pass

Date: 2026-07-17

Status: `PASS` for r9 persistent order-2 execution, oracle v4 acceptance, and
the two-iteration amortization discriminator. Verdict:
`AMORTIZATION_CONFIRMED`; no HP-5 owner is authorized.

## Resume identity and execution

- Resume preflight:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4c_discriminator_20260717_r9/order_02_resume_preflight.json`.
- Resume preflight SHA-256:
  `22a149397bbf6ecbab7d12aa8d3736b59a3b50b1054b5c101027adb4a76324cb`.
- It revalidated immutable order 1, r9 identity, oracle v4/amendment, 1252
  frozen source files, 171 provider entries, persistent Hydra compose, both
  98-D teachers, MuJoCo 36/35/29, and order-2 absence before training.
- Persistent order-2 acceptance SHA-256:
  `51c1154388a0d24351558d40f0f1ddd62501f58b9de79ae6a38d04176862451c`.
- Execution-complete SHA-256:
  `886ea139d8c6bf622c0d84b599a7c292911b79a00df6e29c8a138874647ca129`.
- Both runs report `accepted=true`; each has two iterations and 55 timing
  records. `legacy_order_1_rerun=false`.

## Persistent iteration facts

| Fact | Iteration 1 | Iteration 2 |
| --- | ---: | ---: |
| input weight version | 1 | 2 |
| actual learner updates | 16 | 24 |
| request `total_elapsed` sum | 0.315057 s | 0.149815 s |
| request component sum | 0.098070 s | 0.141073 s |
| request residual | 0.216987 s | 0.008742 s |
| workflow stage sum | 0.170988 s | 0.188289 s |
| env inits added | 2 | 0 |
| teacher inits added | 2 | 0 |
| cache hits added | 2 | 4 |

Iteration 1 to 2 request total falls `52.45%`; request residual falls `95.97%`.
The worker PID is stable across all six requests. Iteration 2 adds no resource
construction and all four resource acquisitions are cache hits: one for
`walk_flat`, one for `static_stand`, and two for `walk_to_stop`.

Final persistent counters are six requests, six resets, six cache hits, two
env/teacher init and close events, one student init, and zero errors. Cleanup
occurs once after both outer iterations, costs `0.168563 s`, and amortizes to
`0.084282 s` per iteration. It is `4.70%` of the persistent process duration.

## Interpretation boundary

The amortization mechanism is runtime-confirmed: the material request residual
is a first-iteration cold cost, while the second iteration reuses both cached
resource identities. This directly answers the E62 discriminator.

It does not establish a production throughput win:

- persistent process elapsed: `3.585368 s`;
- legacy process elapsed: `2.971893 s`;
- this single pair has persistent `20.64%` slower end to end;
- iteration 2 also raises learner work from 16 to 24 updates, increasing the
  persistent workflow-stage sum by `10.12%`;
- process startup and other unaccounted invocation costs remain outside the
  iteration-stage sums.

The machine-readable verdict is:
`/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4c_discriminator_20260717_r9/hp4c_two_iteration_amortization_verdict.json`,
SHA-256
`2db133c778187de394686195c6c892c782261296f27903f7345149babd1cd287`.

## Decision

`AMORTIZATION_CONFIRMED` with `hp5_owner=null` and `hp5_authorized=false`.
Cold resource construction and final cleanup already amortize across outer
iterations; the warm recurring residual does not expose a stable owner-layer
optimization. Do not add timers or optimize source under HP-5 from this pair.

UniLab source/config and Architecture did not change. Persistent execution
remains OFF-default, and no stable speedup claim is accepted. A future stable
throughput claim would require a separately frozen repeated two-iteration
benchmark; it is not implied by this discriminator.
