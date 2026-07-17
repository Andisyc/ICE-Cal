# HP-4b Order 1 Cumulative Schema Block

Date: 2026-07-16
Status: `BLOCKED`
Class: S2/S4 formal-route integration and runtime dataflow probe.

## Frozen Execution

HP-4b r2 was authorized after E49. The exact first manifest entry ran from
`/private/tmp/unilab-hp4b-f7d87a15`:

```text
route=legacy
repetition=1
output=.../hp4b_ab_20260716_gate0b_r2/legacy_r1
```

The formal command exited 1 in cumulative aggregation:

```text
ValueError: multitask sources must either all include scenario_labels or none
```

Raw stdout:
`/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260716_r2/execution_logs/order_01_legacy.log`.

## Parameter Inventory

The run stopped with manifest stage `BOOTSTRAP_COMPLETE` and zero completed
DAgger iterations. All three scenario collections had already completed:

- 21 request metrics: three scenarios times seven legacy request stages;
- `walk_flat`: 128 rows, no `scenario_labels`;
- `static_stand`: 128 rows, no `scenario_labels`;
- `walk_to_stop`: 128 rows, `scenario_labels={walk_to_stop}`.

The cumulative parent sources have the same mixed presence:

- reusable walk and stand role artifacts: no scenario labels;
- RT-10 walk/static role artifacts: no scenario labels;
- RT-10 walk-to-stop artifact: scenario labels present.

`workflow.py` carries `WorkflowDatasetSource.scenario` for active scenario
sources, but `data.py` checks identical transition-field presence before later
source-level scenario annotation can normalize the row semantics. The existing
fail-closed data contract therefore rejects the mixed cumulative set.

## Acceptance State

- Request collection boundary: reached for all three scenarios.
- Aggregation boundary: failed before aggregate artifact creation.
- Workflow/learner/checkpoint/cleanup metrics: absent.
- New student checkpoint: absent.
- Order 2 through 8: not started.
- Server process: untouched.

This is a formal workflow/data-owner integration defect. It is not evidence for
legacy or persistent speed, policy quality, or a bottleneck verdict.

## Stop Decision

HP-4b stops at order 1 according to the frozen fail-fast contract. The partial
run directory and raw log remain as evidence and must not be reused as a fresh
repetition. A repair requires a separately authorized owner-level decision for
how active role scenarios satisfy the complete transition/scenario schema
without silently mutating legacy artifacts. After repair, code identity,
workload outputs, and Gate 0B must be refrozen before any A/B rerun.
