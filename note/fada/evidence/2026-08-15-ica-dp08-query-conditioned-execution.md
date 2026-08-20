# ICA-DP-08 query-conditioned execution evidence

Status: `CURRENT / CLOSED-OFFLINE`  
Design: `ICE-Cal / ICA-DP-08 / FADA-CONTEXT-METHOD-v006 + FADA-CONTEXT-TRAIN-v005`  
Checkout: `codex/in-context-execution-calibration@5949136e43d3`  
Production/test content: `sha256:2ec4a818a4e1d085ba83d0c3e81928d1bbcf756a2006082cc884f1e9fc3c8c6b`  
Formal test file: `sha256:913694cb5919a8baa18ce8486ac9c96e391672eaac7e179650aef3ea2ef1370d`

This ledger records producer-owned proof for the active query-conditioned Context route. It does not
reinterpret the Design Inspector or replace raw manifests.

## E1 — Module semantics and persistence

Claim: the affected owner boundaries implement the confirmed complete-Support, current-history
`delta_z_t`, frozen-owner, K=6/first-Action, pair-balanced loss, and fail-closed schema semantics.

- Producer/status: `module-alignment-test / MODULE-CORRECT`
- Manifest: `note/testing/module_test_manifest.json`
- Manifest SHA-256: `bfcd63d287267840a785efb58d0689e7bb2682933c17ef34d851d67f7a070c9e`
- Evidence SHA-256: `note/testing/module_test_evidence.json` =
  `6ddfa6c659e3af2da5efd92ec1b6197bdf71a39f9c6b844360290ceb49839ff8`
- Observed: 16 owner rows, 86 semantic cases, 0 missing cases, and 165/165 affected module tests.
- Limitation: S1/module and persistence evidence is not official-route, simulator, device, training, or
  policy-quality evidence.

## E2 — Maintainability final gate

Claim: the authorized 21-file implementation snapshot has no unresolved P0–P3 final-gate finding.

- Producer/status: `code-review-expert / FINAL_GATE_PASS`
- Receipt: `note/fada/reviews/2026-08-15-ica-dp08-final-gate.json`
- Receipt SHA-256: `5425600138d046486b12e38116a99717a121fad3279c8a897315c30636951076`
- Review object: `note/fada/reviews/2026-08-15-ica-dp08-final-review-object.txt`
- Review-object SHA-256: `13371f6ca816f22fd069ded9dd612b363716cc7f6be0a45f5b287c62322a49d5`
- Observed: the finite-output admission and active-loader legacy-surface findings are resolved.
- Limitation: maintainability review neither proves the official runtime route nor upgrades policy
  quality.

## E3 — Official offline route and persistence

Claim: unchanged production composition owners carry the active v006 object through artifact
admission, evaluator, and Context playback to the final first-Action consumer; persisted named values
survive strict load to the first consumer.

- Producer/status: `formal-runtime-audit / LONG_TRAINING_READY` (technical status only)
- Formal evidence: `note/testing/formal_official_route_evidence.json`
- Formal evidence SHA-256: `5b2713c1be99296ec5fec269a2633feee0fb06f8c6bcf2807d8819a03b955c0f`
- Formal manifest: `note/testing/formal_audit_manifest.json`
- Formal manifest SHA-256: `f672cb0cbe8213ced8a39b1dd31c0a23f6c6aac4e730ac26d02c566aaa6cc934`
- Observed: 4/4 bounded formal cases and 169/169 expanded affected tests; R1 on
  `EDGE-01..06`; R2 persistence on `EDGE-05`; two control cycles, two Context forwards, immutable
  complete Support, changed current history/residual, and first-Action-only consumption.
- Fail-closed observations: schema-3 Context artifact, non-policy playback mode, and mismatched current
  Command reject before the prohibited consumer/mutation boundary.
- Limitation: the deterministic external seams prove necessary offline route capability only. They do
  not prove MuJoCo/device behavior, learning convergence, robustness, deployment, or policy quality.

## Authority boundary

No Context training, live simulation, GUI, hardware/device execution, policy-quality evaluation,
external operation, destructive operation, or Git action was requested or executed. The formal
validator's technical `LONG_TRAINING_READY` result is not `LONG-RUN-AUTHORIZED`; a future run requires
new human authority and revalidation of current runtime, checkpoint, configuration, and simulator
identities.

The fixed condition-level residual Contracts and evidence remain historical and forbidden as proof
for this active route.
