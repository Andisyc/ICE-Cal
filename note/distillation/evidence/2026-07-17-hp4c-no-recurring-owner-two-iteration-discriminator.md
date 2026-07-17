# HP-4c No Recurring Owner, Two-Iteration Discriminator Required

Date: 2026-07-17

Status: HP-4c analysis `PASS`; verdict `NO_HP5_OWNER`. HP-5 is not authorized.

## Scope

This is a read-only decomposition of E61. No source, timer, configuration,
checkpoint, oracle, or run artifact changed, and no new training ran.

Machine-readable verdict:
`/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260717_r8/hp4c_bottleneck_verdict.json`,
SHA-256 `713be7a157ebfbfc9cb0dbb8ae7514347e5fbdeb88cb5e8214cc5d422c400ae5`.

## Cleanup Candidate

Persistent cleanup is stable and material in the one-iteration experiment:

- raw seconds: `[0.181174, 0.160875, 0.181051, 0.171173]`;
- median `0.176112 s`, range `0.020298 s`, sample stdev `0.009673 s`,
  CV `5.57%`;
- e2e share median `30.03%`.

However, `scripts/train_distill.py` times the entire
`scenario_collector.close()` after `run_multirole_dagger_workflow()` has
completed all target outer iterations. The interval combines process/IPC
shutdown, resource close, SharedWeightSync cleanup, and lifecycle-report
transfer. It occurs once per formal invocation, not once per row, request, or
outer iteration. It is therefore not an HP-5 hot-path owner.

## Persistent Request Residual

The aggregate request residual is also stable:

- raw seconds: `[0.166483, 0.141742, 0.162055, 0.146455]`;
- median `0.154255 s`, range `0.024742 s`, sample stdev `0.011941 s`,
  CV `7.74%`;
- e2e share median `26.07%`.

The manifest's `collector_metrics.collect_seconds` localizes this interval
inside `_collect_role/_collect_transition`, not in the outer worker wrapper.
The per-scenario decomposition and runtime counters are:

| Scenario | Median collect residual | Counter state after request | Lifecycle class |
|---|---:|---|---|
| walk_flat | 0.119814 s | hit=0, teacher/env init=1/1, reset=1 | first identity cache miss |
| static_stand | 0.030430 s | hit=0, cumulative init=2/2, reset=1 | second identity cache miss |
| walk_to_stop | 0.002251 s | hit=2, cumulative init=2/2, reset=1 | both identities cache hit |

`PersistentResourceCache.acquire()` constructs the teacher/env bundle on the
first two misses. `run_request()` resets before collection and reuses the exact
bundle afterward. The confirmed warm-cache residual is only about 2.25 ms,
approximately 0.39% of the persistent e2e median. The outer `collect()` wrapper
residual is only about 12-24 microseconds per request.

Thus the apparent 0.154 s residual is primarily two once-per-worker cold
resource constructions, not a recurring per-request bottleneck.

## Owner Evidence

- `g1_persistent_worker.py:343-360`: `total_elapsed` wraps weight sync,
  collection, and artifact write; `collect_seconds` wraps the role/transition
  collection path. Evidence: code-confirmed.
- `persistent_resources.py:132-174`: cache miss creates teacher/env; every
  request resets; exact identities later hit cache. Evidence: code-confirmed
  and runtime-confirmed by all four counter sequences.
- `persistent_resources.py:188-196`: cached resources close once. Evidence:
  code-confirmed and runtime-confirmed by close counters.
- `scripts/train_distill.py:2113-2151`: one complete runtime close is timed
  after all target iterations. Evidence: code-confirmed.

## Verdict

No stable recurring owner exists in the current evidence:

- cleanup is stable but once per invocation and composite;
- teacher/env construction is stable but once per identity per worker;
- warm cache-hit and outer wrapper residuals are too small;
- therefore no HP-5 optimization is authorized.

The observation that cleanup median divided by the observed one-iteration
request saving is about `1.249` suggests two outer iterations may cross the
amortization boundary, but this is a heuristic, not a measured break-even.

## Smallest Additional Discriminator

Run one newly frozen pair:

- one legacy run and one persistent run;
- `dagger_iterations=2`;
- all other source, assets, teacher/checkpoint, seed, device, env count,
  scenario, quota, samples, and update fields unchanged;
- compare outer iteration 1 versus 2 request totals and cache counters;
- report the single cleanup cost both raw and amortized.

This requires a new workload identity and an iteration-aware acceptance oracle;
the r8 one-iteration oracle must not be reused unchanged. If iteration 2 does
not separate cold and warm behavior, the fallback is owner-local timers for
teacher init, env init, reset, resource close, worker join, and SharedWeightSync
cleanup. The discriminator and fallback both require separate authorization.
