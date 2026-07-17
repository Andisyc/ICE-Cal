# HP-4c r10 Eight-Run Benchmark: No Stable Speedup

Date: 2026-07-17

Evidence ID: E67

Status: PASS for execution/oracle/analysis; primary verdict
`NO_STABLE_SPEEDUP`.

## Authorized Boundary

Execute frozen order `L1,P1,P2,L2,L3,P3,P4,L4`, invoke frozen oracle v4 after
every successful run, and stop on the first command or oracle failure. Only
after 8/8 acceptance apply the pre-registered `subprocess_elapsed_seconds`
rule. Source, config, workload, oracle, HP-5, and default-on changes were out
of scope.

## Immutable Entry Gate

- identity: `8f14c14c3e8993c2d943a4871347d0d0bf33675194595946f7e0f1fd9a3f1185`
- oracle v4: `9acbbef280203ad1b3fce686a01dbea9aeb77462cc6a765d50405f75d09683a0`
- oracle contract: `9329719d4dcbac6f0e83170faad8a2f2037a2b1777c885c28ca6248d34a6aed6`
- frozen preflight: `4d97819a7905d68ef985b325ed91aaed7ee736719d478dcbfc1b61f254801fb4`
- source bundle: `ea1d4f7a6acc3a35f9669bbc55c3df681e48100bdbe72e0880def609f5d5b25e`
- frozen cwd: `/private/tmp/unilab-hp4b-ea1d4f7a`
- fresh entry facts: formal output root, execution logs, and all eight run
  directories absent; frozen cwd present.
- execution start: `31fc8d1d9eacf87ca208f834d0ab1952ac479abed047afa59804da4a76951be3`

## Live Execution and Acceptance

All commands exited zero and every immediate oracle-v4 invocation returned
`accepted=true`. No run was retried or reordered.

| Order | Route | Rep | Process seconds | Acceptance SHA-256 |
|---:|---|---:|---:|---|
| 1 | legacy | 1 | 3.010497 | `c1fe6618...1cf5` |
| 2 | persistent_async | 1 | 2.900757 | `d6be6f80...4727` |
| 3 | persistent_async | 2 | 2.884136 | `fcdab103...127a6` |
| 4 | legacy | 2 | 2.266085 | `06e636ad...5197` |
| 5 | legacy | 3 | 2.282326 | `a1f43cbe...cbd1` |
| 6 | persistent_async | 3 | 2.891054 | `82c3e42c...457` |
| 7 | persistent_async | 4 | 2.891801 | `b4155913...6a83` |
| 8 | legacy | 4 | 2.289864 | `a54be888...8a42` |

Execution-complete SHA-256:
`617b5a52a23277bb1d31f3a5f08a5209621ad3063ec9cd7efa28e2b00b7a840f`.

Each accepted run contains two ordered outer iterations, update counts
`16 -> 24`, exact checkpoint lineage, frozen scenario/schema facts, and
complete metrics/manifest identities. Persistent runs close with 6 requests,
6 resets, 6 cache hits, 2 env/teacher init and close, and 0 request errors.

## Pre-Registered Primary Verdict

| Fact | Result | Required |
|---|---:|---:|
| legacy median seconds | 2.286095 | - |
| persistent median seconds | 2.891427 | `< legacy median` |
| paired ratios P/L | 0.963548, 1.272740, 1.266714, 1.262870 | - |
| median paired ratio | 1.264792 | `< 1` |
| pairs below 1 | 1 / 4 | `>= 3 / 4` |

All three stable-direction requirements fail. The verdict is therefore
`NO_STABLE_SPEEDUP`. Persistent process timing is highly consistent (CV
0.00204), while legacy repetition 1 is a cold outlier relative to the next
three repetitions; the balanced paired rule prevents that one pair from
creating a false speedup claim.

## Secondary Mechanism Evidence

Across four persistent repetitions, median request total falls from 0.226276 s
in iteration 1 to 0.147781 s in iteration 2 (-34.69%). Median request residual
falls from 0.135108 s to 0.008516 s (-93.70%). Every persistent repetition adds
four cache hits and zero env/teacher initialization in iteration 2. Median
cleanup is 0.153230 s once per process, or 0.076615 s per iteration.

This confirms the cache/cleanup mechanism from E65 but cannot override the
end-to-end primary metric. No recurring owner is eligible for HP-5.

## Artifacts and Decision

- analysis:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4c_multirep_20260717_r10/benchmark_analysis.json`
- analysis SHA-256:
  `70fa762db56d5bf3830fbf66ec07c3fc36fc0b38ffda13c781f25c496dd49ac0`
- execution logs:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4c_multirep_20260717_r10/execution_logs`

Decision: execution and acceptance pass, but stable speedup does not.
`hp5_owner=null`, `hp5_authorized=false`, and
`default_on_authorized=false`. Persistent remains OFF-default. This benchmark
makes no policy-quality or physical-acceptance claim.
