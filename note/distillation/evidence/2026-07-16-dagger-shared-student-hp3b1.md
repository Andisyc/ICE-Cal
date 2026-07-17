# DAgger Shared Student Runtime HP-3b1 Evidence

Date: 2026-07-16

Scope: checkpoint publication and one resident worker-side student only. No G1
env, SAC teacher, real dataset, or speedup claim is included.

## Core Parameter Trace

```text
checkpoint state_dict
-> exact key/shape validation
-> SharedWeightSync.write_weights
-> version 1/2
-> spawned worker SharedWeightSync.read_weights_into
-> same resident torch module
-> DaggerCollectResult observed version + weight_sum
```

## Red/Green Evidence

The red test failed because `persistent_runtime.py` did not exist. A sandboxed
green attempt then reached the actual POSIX shared-memory boundary and failed
with `PermissionError`, which was classified as a sandbox false failure.

The same test outside the sandbox passed: `2 passed in 0.78s`.

Observed golden facts:

- first checkpoint `[1, 2]` published version 1 and worker weight sum 3;
- second checkpoint `[4, 5]` published version 2 and worker weight sum 9;
- both results came from the same spawned worker PID, different from parent;
- a `2 -> 3` input-shape drift failed before publication and version remained 1.

Impact group covering persistent runtime, async runtime, workflow,
SharedWeightSync, and AsyncRunner: `53 passed in 5.89s`.

Final combined config/workflow/runtime/IPC gate: `372 passed, 1 deselected in
14.88s`; the deselected pre-existing module-polluting script test was then run
alone and passed `1 passed in 0.67s`. Thus every selected test executed and
passed, while preserving the known isolation boundary.

Ruff: `All checks passed!`.

## Decision

HP-3b1 passes. UniLab's actual `SharedWeightSync`, not an emulated counter, now
updates one resident collector-side student across DAgger iterations. HP-3b2
must add the real role teacher/env resource owner and dataset collection before
the production factory is wired into `train_distill.py`.

## Unconfirmed

- real checkpoint metadata/model family beyond state key/shape compatibility;
- G1 teacher/env persistence and reset semantics;
- role/transition dataset semantics in the persistent worker;
- GPU transfer cost and real throughput;
- production Hydra ON command.
