# Gate 0B Bundle Repair And Frozen Preflight

Date: 2026-07-16
Status: `PASS`
Class: S0/S2/S3 T-persist/T-oracle plus frozen-cwd live readiness.

## Repair Boundary

E48 proved that E47's deterministic bundle omitted `README.md`, a required
`pyproject.toml` build input. This repair does not patch the extracted tree.
It replaces the fragile source allowlist with the complete output of
`git ls-files -co --exclude-standard` and creates versioned r2 artifacts.

No source behavior, workload, teacher, dataset, checkpoint, server process, or
HP-4b command changed.

## Frozen Identity

- Identity manifest:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260716_r2/gate0b_identity_manifest.json`
- Identity manifest SHA-256:
  `256da8cf279b7283144565005731e4e94c0d8ab1ac56c27d019dfd5cf00732ab`
- Source bundle:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_gate0b_20260716_r2/unilab_dagger_source_snapshot.tar.gz`
- Source bundle SHA-256:
  `f7d87a155462955efb300fff6f369fad38886faae6c6d11dc4cf1abca77ac632`
- Embedded source manifest SHA-256:
  `609984f9c744a9e24fa5c3910fd1a80af40a2da49f6612169e4c4544c7f54cae`
- File count: 1241.
- Bundle size: 46,372,467 bytes.
- Frozen cwd: `/private/tmp/unilab-hp4b-f7d87a15`.
- Reserved output root:
  `/Users/chengyuxuan/ArtiIntComVis/UniLab/logs/distill_workflow/hp4b_ab_20260716_gate0b_r2`.

Two consecutive generations produced the same bundle and embedded-manifest
hashes. The new manifest explicitly asserts `pyproject.toml`, `README.md`,
`LICENSE`, `uv.lock`, and `src/unilab` before extraction.

## Frozen-Cwd Runtime Facts

The failed E47 cwd was removed as authorized. The r2 archive was extracted into
the required absent new cwd. From that cwd:

```text
uv run python /private/tmp/hp4b_frozen_preflight.py
```

exited 0 and observed:

```text
uv_build: PASS
locked packages installed: 171
source file hashes: 1241/1241 match
external asset hashes: 7/7 match
G1 scene XML: loaded
MuJoCo nq/nv/nu: 36/35/29
shared compose hash: d6e047f43e03de0d13af32823f09a7b538dba21672540e455e51f61f869b2000
legacy/persistent allowed diff only: true
HP-4b output root absent: true
```

The r2 route-specific config hashes are:

- legacy: `cd4d5362ba671609eacd840a5f3ae28132bd1f808dce75521d3deac732b0dac3`
- persistent: `3d25096964750057c38a4aa457e94f1e03cbcd7b315ba1be88b24849434612f7`

## Decision

Gate 0B bundle repair and executable frozen preflight pass. E48's root cause is
fixed at the bundle owner rather than hidden by a mutable-file copy. The r2
identity manifest still records `execution_authorized=false`; zero HP-4b A/B
runs started. HP-4b execution requires a new explicit authorization.
