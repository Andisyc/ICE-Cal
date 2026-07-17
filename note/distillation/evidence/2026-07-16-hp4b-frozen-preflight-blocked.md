# HP-4b Frozen Live Preflight Blocked

Date: 2026-07-16
Status: `BLOCKED`
Class: S2 live-sentinel readiness / integration defect.

## Runtime Fact

E47's bundle was extracted to `/private/tmp/unilab-hp4b-b75f100e`. The first
frozen-cwd command was:

```text
uv run python /private/tmp/hp4b_frozen_preflight.py
```

It exited 1 before UniLab import:

```text
Failed to build unilab
failed to open file /private/tmp/unilab-hp4b-b75f100e/README.md
No such file or directory
```

`pyproject.toml:9` declares `readme = "README.md"`. E47's bundle allowlist
included `pyproject.toml` but omitted that required build input.

## Classification And Isolation

This is a Gate 0B frozen-bundle integration defect, not an algorithm,
checkpoint, simulator, or server failure. Copying README from the mutable
worktree was forbidden because it would change executable identity. All eight
A/B output directories remained absent; no env, collection, optimizer update,
metrics artifact, or HP-4c action occurred.

## Decision

HP-4b stopped at HP-4b1. The next bounded action is a separately authorized
Gate 0B bundle repair: audit all package inputs, generate a new bundle and
identity manifest, rebuild an absent frozen cwd, and rerun frozen preflight.
