# HP-7 Side-Session Closeout

Date: 2026-07-17

## Current State

- E99 closes HP-7 implementation, production wiring, and one frozen bounded
  persistent live workflow as PASS.
- The accepted r6 run completes one DAgger iteration, 12,320 learner updates,
  checkpoint/manifest/metrics, and cleanup.
- Batch staging is no longer the dominant owner. It is 34.3355 s, 9.32% of
  wall time; learner forward/backward now own most measured time.
- HP-4c still records no stable end-to-end persistent speedup. Repository
  default remains `legacy`; promotion/default-on remain unauthorized.
- RT-10 physical walk-to-stop acceptance and final student policy quality are
  still open research/runtime boundaries.

## Formal-Training Boundary

The HP engineering program is complete, but a new formal DAgger run is not yet
frozen. The main session must choose the checkpoint lineage before planning or
executing it:

1. Start from the original parent iteration-3 checkpoint for a clean formal
   lineage; or
2. Explicitly promote the r6 performance-sentinel checkpoint into the formal
   lineage; or
3. Evaluate the r6 checkpoint before choosing either lineage.

No option is selected by this side session. Formal training requires a new
run/output identity, workload/iteration count, seed/device, acceptance oracle,
and physical evaluation stop. Do not reuse or rerun the r6 supervisor.

