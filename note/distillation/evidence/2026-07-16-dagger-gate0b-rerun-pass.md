# DAgger Gate 0B Rerun: Immutable A/B Identity

Date: 2026-07-16
Status: `PASS`
Class: S0/S3 T-persist/T-oracle. No training, simulator, or timing run.

## Design Boundary

This is an engineering identity gate under `DISTILL-DP-01 Teacher Policies`,
`DISTILL-DP-03 Role Data`, and `DISTILL-DP-05 Student-State DAgger`. It does
not change method semantics or active contracts. The Concept Figure nodes keep
stable design and contract IDs; their JSON `contract_section` fields are null,
while the active method contract contains the canonical Design Point Register.
That cross-file metadata gap is recorded but does not alter this frozen A/B
identity.

Scope: recompute assets, prove legacy/persistent compose symmetry, freeze an
immutable source bundle, and freeze workload, run order, commands, and outputs.

Non-scope: executing HP-4b, MuJoCo, training, server mutation, speedup claims,
HP-4c, HP-5, Motrix, or policy-quality acceptance.

## Frozen Raw Artifacts

- Identity manifest:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260716/gate0b_identity_manifest.json`
- Identity manifest SHA-256:
  `2f53362f04ff41d63049004a629410f79de4116ba7983afe240dd4c64e3df1d0`
- Source bundle:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260716/unilab_dagger_source_snapshot.tar.gz`
- Source bundle SHA-256:
  `b75f100e212d7edfe09d3c5920918265eafb12eb5cc3c41b3d4d664104d0e779`
- Embedded source manifest SHA-256:
  `7522d5de816ac4f9beaa1f5396a2857d958ca5b5d50035342d47bacaee48c4b1`
- Source scope: 740 files under `conf/`, `scripts/`, `src/`, `tests/`, plus
  root build/lock files; logs, assets, notes, caches, `.git`, and `.venv` are
  excluded and separately governed.
- Git identity: branch `codex/dagger-mainline-runtime`, base commit
  `601a2e4013368423540554a351062b012b4c83ce`.

The deterministic bundle was generated twice with the same SHA-256. Every
included file has an embedded path, size, and SHA-256 record.

## Asset Identity

All seven canonical asset hashes were recomputed and match E45:

- parent manifest: `4e2909d1a5252ac732a9228c202ec70aaa199b260260f5e568674052b41d3d83`
- parent student: `96aaecf3f635163eafd46b99f83a3a1064a59600c7056a4f9d638e9aaad8ae35`
- bootstrap student: `96244e52efea5c4fa17c3aa55b9f90fb92c9d4df418d4fb1a2a6276c25bbc0ca`
- walk teacher: `7a0729a45859b2db05f2a642f6e80eedbd25f8135a75ff2af9dddae58bbf8279`
- stand teacher: `91e18d3d1f469b2bead350cd41b33494c39c8ec8d26f2daf802e0273afa2c6da`
- walk dataset: `e50b73020be585086c940b9c99cce583765c6caaf7b0fc7ffd2e0541b892f63e`
- stand dataset: `c912e394b3396d42e04611db71927b3dbc73fcd7fffb7840944b3f03b7aae5fe`

The parent manifest remains `DAGGER_ITERATION_1_COMPLETE`. This gate does not
claim its physical acceptance.

## Compose And Workload Identity

The read-only Hydra differential assertion proves that the resolved legacy and
persistent configs differ only in `training.workflow.execution_mode` and
`training.workflow.run_dir`.

- normalized shared config: `d6e047f43e03de0d13af32823f09a7b538dba21672540e455e51f61f869b2000`
- legacy resolved config: `6622c3ec8cb0c909fc14192fa3e19abf668fb9c8731b4d563f5c92b86f49a5a5`
- persistent resolved config: `3d8ea0b9b978bb6ffa3eaa11ff3ad7949c288d50a019dc2a109d0d19f3e868d8`

Shared workload: seed 1, CPU, four envs, 128 rows per scenario, one outer
iteration, eight updates, scenario order `walk_flat`, `static_stand`,
`walk_to_stop`, and quotas 0.50/0.25/0.25.

The identity manifest freezes eight runs with balanced order:

```text
legacy r1 -> persistent r1
persistent r2 -> legacy r2
legacy r3 -> persistent r3
persistent r4 -> legacy r4
```

Every run has a unique absent output directory. The frozen source executes only
from the bundle-extracted cwd `/private/tmp/unilab-hp4b-b75f100e`; that path is
also required to be absent before preparation.

## Verification

- Fresh affected suite: `312 passed, 8 skipped, 5 warnings in 6.76s`.
- Deterministic bundle regeneration: identical SHA-256 on two generations.
- Gate verifier: seven assets, 740 source files, eight runs, eight unique
  outputs, no cwd/output collision, and exact legacy request, persistent
  request, workflow, and cleanup stage contracts.
- The first verifier invocation exposed only a temporary probe import mistake:
  `PERSISTENT_REQUEST_STAGE_NAMES` is owner-local to `performance.py`. The probe
  was corrected to import from that owner; no frozen source file changed.

## Decision

Gate 0B rerun is `PASS`. Code/config/assets/workload/run order/command/output
identity is frozen strongly enough to authorize a separately approved HP-4b
execution against these exact raw artifacts. No HP-4b command ran in this Gate,
and the identity manifest deliberately records `execution_authorized=false`.
