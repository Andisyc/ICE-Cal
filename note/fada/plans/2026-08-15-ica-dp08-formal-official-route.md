# ICA-DP-08 Global Simplified Formal Test Plan

Status: `EXECUTED-PASS` (2026-08-15)  
Design: `ICE-Cal / ICA-DP-08 / FADA-CONTEXT-METHOD-v006 + FADA-CONTEXT-TRAIN-v005`  
Checkout: `codex/in-context-execution-calibration@5949136e43d3 + content-sha256:2ec4a818a4e1d085ba83d0c3e81928d1bbcf756a2006082cc884f1e9fc3c8c6b`  
Module admission: `MODULE-CORRECT`, manifest `note/testing/module_test_manifest.json`, sha256 `bfcd63d287267840a785efb58d0689e7bb2682933c17ef34d851d67f7a070c9e`.

## 1. Claim boundary

This audit may prove only:

- `R1 OFFICIAL-OFFLINE`: the real artifact-admission and Context-playback composition roots carry the active v006 object to the final Action consumer;
- `R2 PERSISTENCE`: a schema-4 Context checkpoint and schema-2 Support/Query dataset retain their bound identities through strict load and the first playback consumer;
- the minimum necessary capability that a fixed complete Support can be reused while changed current Query histories produce a newly recomputed `delta_z_t` and a changed executed first Action.

It may not claim simulator compatibility, policy quality, convergence, robustness, deployment admission, or authorization to start training.

## 2. One coherent official offline transaction

Add one test module:

- `tests/scripts/test_fada_context_official_route.py`

The test must execute this transaction in one temporary directory:

1. Construct a deterministic H=30, K=6 frozen Planner/IDM policy, persist a valid source-checkpoint payload with its exact production schema/architecture/state keys, and load it through the real `load_fada_policy_checkpoint`. Construct a validated complete-Support/per-timestep-Query batch using production dataclasses.
2. Persist the batch with `save_support_query_dataset` and persist a schema-4 Context checkpoint with `save_context_support_query_checkpoint`. Install a deterministic nonzero current-history-to-`delta_z_t` path in the real Context Encoder; do not replace the Context, Planner, IDM, artifact, controller, or Session owner.
3. Enter `scripts/preflight_fada_context_support_query.py::main` through its real CLI parser in explicit `--artifact-admission` mode using the real repository config parser, real source/dataset/Context checkpoint loaders, real split identities, and real `prepare_context_support_query_artifact`. Do not monkeypatch an artifact or policy loader.
4. Compose the real Hydra preset `+context_playback=left_knee_070`, including `interactive.action_mode=policy`, and point its effective values at the same temporary source/dataset/Context artifacts.
5. Enter `scripts/evaluate_fada_context_support_query.py::main` through its real CLI parser with one seed, one environment and two steps. Replace only `create_fixed_fault_paired_environments` with deterministic external healthy/fault env adapters; retain the real artifact preparation, Support selection, three evaluation branches, aggregation and atomic report write.
6. Enter `scripts/play_fada_context_viser.py::main` with the composed Hydra config and retain the real imported `scripts/play_interactive.py::play_interactive`, including `_build_playback_config`, FADA branch selection, production session-factory invocation and `FADAPlaybackSession.advance`. Inject deterministic external env/wrapper dependencies through the existing `_default_fada_playback_deps` dependency mapping while retaining the real source loader/resolver. Replace only MuJoCo model/viewer/render/pacing functions with a two-iteration external viewer adapter; do not replace `play_interactive` itself.
7. The fake env/wrapper changes the current observation/action histories between the two cycles but keeps complete Support and Command identity fixed. Record the actual Actions passed to the final wrapper/env consumer.

The test must not call Context or `act_with_context` directly and must not synthesize a second artifact interpretation path. Replaced boundaries are limited to paired evaluation environments and playback env/wrapper/MuJoCo viewer I/O. The production CLI/Hydra parsers, `play_interactive`, FADA branch, Session factory, Context controller, artifact owners and final Action consumer remain real.

## 3. Design probe

Design claim: deployment recomputes one query-conditioned latent residual per control cycle from complete Support and current Query histories.  
Indispensable variable: current State/Action history at the second rollout cycle.  
Necessary capability: with the same complete Support and Command, two distinct current histories produce distinct `delta_z_t`, and the Decoder's six-step chunk exposes only index zero to the Action consumer.  
Minimum horizon: two playback control cycles.  
Success witness: two Context calls, identical bound Support identity, different current-history receipts, different residuals, and two final consumed first Actions that each equal their own `action_chunk[:, 0]` and differ from one another by the fixture's predetermined nonzero margin; no tail Action is consumed.  
Falsifier: Context is precomputed once, Support is sliced, current histories do not reach Context, the two consumed first Actions remain equal despite the controlled residual change, a tail Action reaches the consumer, or any Planner/IDM parameter changes.

The deterministic fixture is favorable but preserves the indispensable variable and all production owners. A PASS is a necessary-capability witness only.

## 4. Required edge receipts

The test/evidence receipt must record:

- `EDGE-01`: persisted dataset + schema-4 Context checkpoint -> artifact-admission owner, with exact method/schema/H/K/S/source/dataset/train/validation/split identities;
- `EDGE-02`: prepared complete Support + Support Command -> `SupportBoundContextPolicy` selected by the real playback preset;
- `EDGE-03`: current histories + Command -> `FADAPlaybackController` -> `FrozenIDMSupportQueryPolicy.act_with_context`, exactly once per active cycle;
- `EDGE-04`: six-step `action_chunk` -> `FADAPlaybackSession` -> final wrapper/env consumer, first Action only; each consumed Action equals its own chunk index zero and the two consumed Actions differ by the fixture's predetermined nonzero margin.
- `EDGE-05` (`R2`): before-save complete-Support dataset fields and Context tensors -> serialized dataset/checkpoint -> strict prepared owner with the same tensor values, named-field ownership, row/rollout/Command identities and deterministic value digests -> first post-load Action consumer. File digests alone are insufficient.
- `EDGE-06`: evaluator CLI artifact preparation -> held-out complete Support selection -> real three-branch closed-loop evaluator -> atomic report artifact.

Each receipt must bind design, checkout content digest, effective Hydra config fingerprint, cold source-file digest, dataset/checkpoint digests, row/rollout/Command identity, call count, and final effect.

## 5. Fail-closed differentials

Within the same test module, add the minimum direct official-entry differentials:

- historical Context schema 3 is rejected by artifact admission before playback construction;
- Context playback preset with non-`policy` action mode is rejected before Session mutation or Action consumption;
- current Command differing from the bound Support Command is rejected before the first Context-conditioned Action.

These are route differentials. Existing module tests remain the semantic oracle owners.

## 6. Verification and receipts

Run only bounded offline commands:

```bash
uv run pytest tests/scripts/test_fada_context_official_route.py -vv --tb=short
uv run pytest tests/algos/test_fada_planner_idm.py tests/algos/test_fada_context_support_query.py tests/algos/test_fada_context_support_query_evaluation.py tests/algos/test_fada_playback.py tests/scripts/test_preflight_fada_context_support_query.py tests/scripts/test_fada_context_official_route.py -q
uv run ruff check tests/scripts/test_fada_context_official_route.py
uv run ruff format --check tests/scripts/test_fada_context_official_route.py
```

Then create:

- `note/testing/formal_official_route_evidence.json`
- `note/testing/formal_audit_manifest.json`

Validate the manifest with `formal-runtime-audit/v3` against the exact design, checkout content digest, production entrypoints, effective config fingerprint, and persisted checkpoint identity. A technical `LONG_TRAINING_READY` result does not grant human authority to run training; `note/governance.json` must retain `authority.long_run.status=not-requested`.

## 7. Stop conditions

Stop offline and return to the owner if any of these occurs:

- the official entrypoint cannot accept the external fake without replacing a semantic owner;
- effective Hydra composition selects a different branch or non-policy Action mode;
- any artifact identity is silently defaulted or coerced;
- the final consumer is not reached exactly twice;
- the Context mechanism activates zero/one time across the required two-cycle horizon;
- Planner/IDM or bound Support mutates;
- a formal result would require simulator, GUI, training, Git, network, or a new scientific decision.

## 8. Execution receipt

- The reviewed route specification is frozen by READY plan hash
  `sha256:5b5ba06f19788c939d6dd128c2d7c6b47e656511583b139645014a4ce432b089`
  and review hash `sha256:ebd0d5f01b3cf3c5e546bd85e85ed030f8cdc302ed960101b0306843f4e34e80`.
- The bounded formal test passed 4/4 cases; the expanded affected suite passed 169/169.
- `formal-runtime-audit/v3` records R1 `EDGE-01..06`, R2 `EDGE-05`, and exact technical
  `LONG_TRAINING_READY` in `note/testing/formal_audit_manifest.json`
  (`sha256:f672cb0cbe8213ced8a39b1dd31c0a23f6c6aac4e730ac26d02c566aaa6cc934`).
- No simulator, GUI, device, training, network, or Git action ran. The technical status does not grant
  human long-run authority and does not establish policy quality.
