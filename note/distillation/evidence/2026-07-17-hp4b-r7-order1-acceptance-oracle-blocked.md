# HP-4b r7 Order 1 Acceptance Oracle Block

Date: 2026-07-17

Status: `BLOCKED` after a successful formal legacy run but before acceptance.

## Runtime result

- r7 identity SHA-256: `9b180b464433e0f29e59060c9245e9fbcd1879d988eeab802cee67be22f59718`.
- Order 1 legacy command exits 0.
- Manifest reaches `DAGGER_ITERATION_1_COMPLETE`.
- Aggregate has 1024 rows: walk 384, stand 384, transition 256.
- Learner records 16 actual updates after transition replay auto-expansion.
- Checkpoint SHA-256:
  `7f4bf68268f453f5b74c9290b84efa243f215c757a9f85b82cef89b1d2e45c7b`.
- Metrics contain 21 request, 6 workflow/learner, and 1 cleanup record.
- Cleanup is complete with legacy per-request resource scope.

Raw hashes:

- Manifest: `0a769b4feb64a720be48c35a6ef0a483de90795aed6fcf6583a7357e3bf7419a`.
- Metrics: `69ff6e6db3a96536696d1ada282f4412a9fa107a8944b0c8610cc6fee8241fc3`.
- Execution log: `d0c98efdaed50c30b848a7f7443e1b096f5e8e6f10924873489c3a9dc88165ea`.

## Acceptance failure

After the successful command, `/private/tmp/hp4b_run_one_r7.py` requires each
raw scenario artifact to contain 128 identical scenario labels. This fails on
raw walk and stand role artifacts.

E51's accepted contract deliberately keeps legacy role files unchanged:

```text
raw role artifact
-> WorkflowDatasetSource scenario identity
-> explicit in-memory annotation
-> transition-aware cumulative aggregate
```

The transition artifact carries native scenario/age fields. The aggregate
proves the annotation route works and contains complete scenario identity for
all 1024 rows. Therefore the failed assertion checks the wrong boundary. It is
an acceptance-oracle defect, not a collector, aggregate, learner, checkpoint,
metrics, or cleanup failure.

## Isolation and decision

- `order_01_acceptance.json` is absent.
- Orders 2-8 did not start; order 1 is not rerun.
- No source/config/workload/oracle mutation occurred after failure.
- No comparison or speedup claim is accepted.

HP-4b r7 is `BLOCKED`. Gate 0B is incomplete because the external oracle was
not frozen against E51's raw-role versus aggregate-schema ownership.

The next bounded action requires separate authorization: repair and freeze the
oracle so it validates raw role files by role/hash/count, the transition file
by native transition fields, and the aggregate by full scenario identity.
Apply it to the existing completed order-1 artifacts without rerunning training,
then return before order 2.
