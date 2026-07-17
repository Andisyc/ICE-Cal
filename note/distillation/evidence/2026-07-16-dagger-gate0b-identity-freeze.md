# DAgger Gate 0B A/B Identity Freeze

Date: 2026-07-16
Status: BLOCKED
Class: S0/S3 T-persist/T-oracle preflight. No training, simulator, or timing run.

## Design Boundary

Gate 0B is an engineering evidence gate under `DISTILL-DP-01 Teacher
Policies`, `DISTILL-DP-03 Role Data`, and `DISTILL-DP-05 Student-State
DAgger`. It does not change method semantics or active contracts.

Scope: freeze post-HP-4 connector code, assets, workload, run order, commands,
and output identities before HP-4b.

Non-scope: code repair, metrics connector implementation, MuJoCo, training,
performance conclusions, HP-4b, HP-4c, HP-5, Motrix, or server mutation.

## Candidate Asset Identity

The latest locally visible RT-10 manifest is structurally complete at
`DAGGER_ITERATION_1_COMPLETE`. This Gate does not inspect or claim RT-10
physical acceptance.

| Object | Canonical path | SHA-256 |
| --- | --- | --- |
| RT-10 parent manifest | `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/rt10_bounded_20260716_run1/run_manifest.json` | `4e2909d1a5252ac732a9228c202ec70aaa199b260260f5e568674052b41d3d83` |
| RT-10 final student | `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/rt10_bounded_20260716_run1/checkpoints/dagger_iteration_1.pt` | `96aaecf3f635163eafd46b99f83a3a1064a59600c7056a4f9d638e9aaad8ae35` |
| RT-10 bootstrap student | `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/rt10_bounded_20260716_run1/checkpoints/bootstrap_student.pt` | `96244e52efea5c4fa17c3aa55b9f90fb92c9d4df418d4fb1a2a6276c25bbc0ca` |
| walk teacher | `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/fast_sac/G1WalkFlat/2026-07-09_02-48-58_mujoco/model_5000.pt` | `7a0729a45859b2db05f2a642f6e80eedbd25f8135a75ff2af9dddae58bbf8279` |
| stand teacher | `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/fast_sac/G1StandStill/2026-07-09_22-55-05_mujoco/model_5000.pt` | `91e18d3d1f469b2bead350cd41b33494c39c8ec8d26f2daf802e0273afa2c6da` |
| walk role dataset | `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_role_artifacts/rt8_bounded_20260716_retry4/walk_flat.pt` | `e50b73020be585086c940b9c99cce583765c6caaf7b0fc7ffd2e0541b892f63e` |
| stand role dataset | `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_role_artifacts/rt8_bounded_20260716_retry4/stand.pt` | `c912e394b3396d42e04611db71927b3dbc73fcd7fffb7840944b3f03b7aae5fe` |

All hashes were recomputed from files during this Gate and agree with the
manifest where the manifest records them.

## Candidate Workload Symmetry

A read-only Hydra compose succeeded for both modes with these shared values:

```text
workflow=g1_walk_stand
mode=fork
parent=rt10_bounded_20260716_run1
algo.seed=1
training.device=cpu
collect_num_envs=4
dagger_samples_per_role=128
dagger_iterations=1
dagger_updates_per_iteration=8
scenario order=walk_flat, static_stand, walk_to_stop
scenario quotas=0.50, 0.25, 0.25
teacher paths/hashes=identical
role dataset paths/hashes=identical
```

The only composed differences were the intended `execution_mode` and separate
`run_dir`. The first sandboxed compose failed only because the sandbox denied
the shared uv cache; the approved read-only rerun succeeded. No env or training
owner was constructed.

## Code Identity Preflight

- Branch: `codex/dagger-mainline-runtime`.
- Base commit: `601a2e4013368423540554a351062b012b4c83ce`.
- Worktree is not clean: the compact status reports 43 changed/untracked paths.
- Representative post-connector hashes:
  - `conf/distill/config.yaml`: `64de26d85ffa058e09cf0344b7545bf8153704cf5f76216402a7758e1c9234da`;
  - `conf/distill/workflow/g1_walk_stand.yaml`: `1c9e506c867dc2512b628cf8ed3080e653f3de5f8658b187b0c45570d6110593`;
  - `scripts/train_distill.py`: `2ba21224112d442f5de7662cec700082c38375a5f2a5d9c7b876b8c6b8029b88`;
  - `performance.py`: `4edeb72f955d391d33525ec80b90195f7280aeff56d0bf770f32b1678d7fecce`;
  - `workflow.py`: `95d392055674e992c3fee62ad710d82d16b544b9715ef9d20ab2b0e0df117c02`;
  - `g1_persistent_worker.py`: `17a08211844fca679869a5ce020eca00cf4b6d32614f3e7c6a0b984b27bac64a`.
- No immutable source bundle was emitted because the measurement contract is
  blocked and any repair would immediately invalidate that bundle.

## Blocking Runtime Facts

1. Legacy has no structured metrics artifact.
   - `train_distill.py` creates `DistillationPerformanceRunContext` only for
     `persistent_async`.
   - `workflow.py` explicitly forbids a legacy `performance_context`.
   - The legacy regression asserts that no `distillation_metrics.json` exists.
2. Workflow/learner stages are schema-only, not connected.
   - `cumulative_aggregation`, `learner_batch_staging`, `learner_forward`,
     `learner_backward`, `optimizer_step`, and `checkpoint_save` occur only in
     the stage registry/tests; no formal owner records them.
3. Final cleanup evidence is not persisted by the formal workflow.
   - The persistent runtime has an in-memory `close_report`, but
     `train_distill.py` closes the service without attaching it to the workflow
     manifest or metrics artifact.
   - Worker request `total_elapsed` remains `cleanup_state=pending`.
4. Therefore the current commands cannot satisfy HP-4b's required
   route-comparable raw stages and per-repetition cleanup counters. External
   wall-clock timing would be a weaker, asymmetric substitute and is rejected.

## Candidate Commands

The compose inputs above are valid, but the commands are deliberately not
frozen as executable HP-4b commands while the measurement contract is
asymmetric. Once repaired, both commands should fork the same RT-10 parent and
differ only in `training.workflow.execution_mode` and a unique output directory.
The balanced run order should alternate route precedence across one cold and
three repetition pairs; it must be frozen only after the missing records exist.

## Decision And Stop Condition

Gate 0B is `BLOCKED`, not `PASS`. Assets and candidate workload are verified,
but code immutability, symmetric legacy/persistent stage records, workflow and
learner timing, and final cleanup persistence are incomplete. HP-4b must not
run.

The smallest next proposal is a separately authorized HP-4a2d measurement-
symmetry step, split by owner: legacy request observations, workflow/learner
observations, then cleanup-final persistence. After those steps pass, Gate 0B
must be rerun from the beginning and emit a new immutable source bundle.
