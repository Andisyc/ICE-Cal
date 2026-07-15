# Proposal: Candidate To Promoted Checkpoint Lifecycle

Status: proposal, not active contract

## Why This Exists

Current training and playback rely on manually named `.pt` files. A checkpoint
can be structurally valid while its physical behavior, parent identity, or
tested launch path remains unknown. File size and a smoke sentinel have been
misread as quality evidence.

## Proposed Flow

```text
training output
  -> candidate checkpoint + manifest
  -> deterministic contract audit
  -> repeated live role/transition sentinels
  -> acceptance report
  -> promoted checkpoint alias
  -> start.sh loads only promoted identity by default
```

## Proposed Owners

- `checkpoint.py`: checkpoint payload only; no policy-quality decision.
- New manifest owner: immutable candidate identity, parent, teachers, config,
  source data, hash, and model/optimizer byte accounting.
- New acceptance owner: repeated physical scenarios and thresholds.
- `start.sh` / playback: resolve an explicit candidate or latest promoted
  artifact and print the immutable identity.

## Proposed Scenarios

1. Repeated zero-command resets.
2. Stand -> `vx=0.2`, `0.5`, `0.8`.
3. Walk -> zero-command stop recovery.
4. Yaw and lateral command transitions.
5. CPU/MPS inference equivalence where applicable.

## Not Yet Decided

- Number of repeated seeds/resets.
- Minimum stable horizon.
- Exact base-height, tilt, termination, and teacher-action thresholds.
- Whether promotion is a copied file, symlink, or manifest pointer.
- Whether the acceptance suite is local-only or part of CI/server workflow.

No implementation should begin until these acceptance semantics are confirmed.

