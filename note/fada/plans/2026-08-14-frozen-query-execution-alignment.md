# Frozen Query Execution Alignment Implementation Plan

Status: `CLOSED-OFFLINE` (2026-08-15). All production tasks completed under the reviewed scope;
Task 7 records the producer receipts and governance closeout. No live, training, policy-quality, or
Git authority is implied.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align production code with the human-confirmed `ICA-DP-08` query-conditioned
per-control-cycle `delta_z_t` semantics while preserving complete-Support ownership, six-step
decoding, and first-action-only execution/supervision.

**Architecture:** The existing Planner and Tracker interfaces remain unchanged. Context Encoder
fuses one complete Support trajectory with the current Query State/Action histories and emits one
residual for that Query timestep or control cycle. Policy, trainer, evaluator, and playback all call
this same owner boundary; fixed-`delta_z` artifacts fail closed instead of being silently
reinterpreted.

**Tech Stack:** PyTorch, dataclasses, pytest, Hydra entrypoints, `uv run`, repository Contract and
workflow-governance documents.

---

## Authority and scope

This plan is volatile and not implementation authority until the active Contracts, synchronized
Concept Figure/Inspector, confirmed Module Test Cards, and `code-review-expert: READY` receipt are
current. It does not authorize Git actions, simulation, training, checkpoint publication, or
policy-quality claims.

Preserve the complete Support collector, Query per-timestep causal-sample schema, pair-balanced masked
first-action loss, frozen Tracker latent seam, and unrelated FADA/UniLab paths. Do not introduce
Support slicing, residual aggregation, direct Context Actions, online
updates, or legacy-checkpoint coercion.

### Task 0: Reconcile active authority before production work

**Files:**
- Modify: `note/fada/contracts/README.md`
- Modify: `note/fada/task_canvas.md`
- Modify: `note/fada/checklists/current.md`
- Modify: `note/governance.json`
- Modify: `note/architecture/atlas_manifest.json`

- [ ] **Step 1: Complete the admitted Contract activation transaction**

Keep `FADA-METHOD-v005` and `FADA-TRAIN-v005` active. Move only the fixed-residual Context v005/v004
Contracts to history, activate Context v006/v005, and archive the exact proposal content and hashes.

- [ ] **Step 2: Confirm one current authority state**

Registry, Inspector, Atlas, task canvas and governance must all identify active Context v006/v005;
the production implementation is still explicitly `not-implemented` at this point. No current
surface may point to removed proposal or old active-Contract paths.

- [ ] **Step 3: Validate the authority transaction**

```bash
uv run python /Users/chengyuxuan/.codex/skills/workflow-governance/scripts/validate_governance_manifest.py note/governance.json --target engineering-ready
uv run python /Users/chengyuxuan/.codex/skills/repo-architecture-atlas/scripts/validate_atlas_manifest.py note/architecture/atlas_manifest.json --target design-inspector
npm --prefix note/architecture/auxiliary/atlas_app run check
git diff --check
```

Stop before production code if any authority identity or content hash disagrees.

### Task 1: Protect the existing Planner–Tracker boundary

**Files:**
- Test: `tests/algos/test_fada_planner_idm.py`
- Test: `tests/algos/test_fada_context_support_query.py`

- [ ] **Step 1: Keep the existing Planner characterization**

```python
def test_planner_keeps_state_history_and_command_inputs():
    config = _config()
    planner = FADAPlanner(config).eval()
    batch = _batch(config)
    with torch.inference_mode():
        intent = planner(batch.observation_history, batch.command)
    assert intent.shape == (
        batch.observation_history.shape[0],
        config.prediction_horizon,
        config.obs_dim,
    )
```

- [ ] **Step 2: Run the non-regression boundary**

Add a production-dimension fixture with `H=30,K=6` that proves both State History and Command reach
Planner, six Intent tokens reach Tracker, and a strict healthy source state-dict round trip preserves
the existing Planner/Tracker keys and tensors.

```bash
uv run pytest tests/algos/test_fada_planner_idm.py -q
```

Expected: PASS before and after the Context repair.

- [ ] **Step 3: Freeze Planner as non-scope**

The implementation impact set must exclude Planner architecture, `planner_source_loss`, source
replay, source checkpoint schema, and source training. Context policy callers must continue using:

```python
planner_intent = self.planner(observation_history, command)
```

- [ ] **Step 4: Stop on Planner drift**

Stop the implementation immediately if a patch changes the Planner signature, Planner parameters,
source checkpoint compatibility, or the existing residual-to-latest-observation behavior.

### Task 2: Make Context Encoder emit query-conditioned `delta_z_t`

**Files:**
- Modify: `src/unilab/algos/torch/fada_context/support_query.py`
- Test: `tests/algos/test_fada_context_support_query.py`

- [ ] **Step 1: Preserve zero initialization and add a failing query-conditioned interface test**

```python
def test_context_delta_depends_on_current_query_histories():
    config = _config()
    batch = _batch(config, batch_size=2)
    encoder = FADASupportContextEncoder(
        config,
        SupportQueryContextConfig(support_length=8, context_hidden_dim=12, context_layers=1),
    ).eval()
    _install_deterministic_nonzero_query_path(encoder)
    state_a = torch.zeros(2, config.history_length, config.obs_dim)
    state_b = state_a.clone()
    state_b[:, -1, 0] = 1.0
    action = torch.zeros(2, config.history_length, config.action_dim)
    with torch.inference_mode():
        delta_a = encoder(batch.support, state_a, action)
        delta_b = encoder(batch.support, state_b, action)
    assert delta_a.shape == (2, config.hidden_dim)
    assert not torch.allclose(delta_a, delta_b)
```

- [ ] **Step 2: Run the interface test and record RED**

```bash
uv run pytest tests/algos/test_fada_context_support_query.py::test_context_delta_depends_on_current_query_histories -q
```

Expected before repair: `TypeError` because Context accepts only Support.

Keep a separate characterization requiring a freshly initialized Context Encoder to emit exact zero.
The differential fixture installs deterministic nonzero test weights (or an equivalent controlled
Jacobian oracle) for both State and Action history paths; production zero initialization must not
change merely to make the test pass.

- [ ] **Step 3: Add the query-history encoder and fail-closed validation**

Implement one Support sequence encoder and one current-history sequence encoder. Fuse their final
representations before the bounded residual head:

```python
def forward(
    self,
    support: SupportContextBatch,
    observation_history: torch.Tensor,
    action_history: torch.Tensor,
) -> torch.Tensor:
    support.validate(self.fada_config, support_length=self.context_config.support_length)
    self._validate_query_histories(observation_history, action_history)
    support_summary = self._encode_support(support)
    query_summary = self._encode_query_history(observation_history, action_history)
    return self.context_config.delta_scale * torch.tanh(
        self.delta_head(torch.cat((support_summary, query_summary), dim=-1))
    )
```

- [ ] **Step 4: Prove forbidden semantics remain absent**

Add tests requiring the full Support length on every call; reject Support/history batch, shape,
dtype, device and non-finite mismatches; demonstrate that Support has no sample axis and remains
complete on every call. Query Future and executed-Action labels are not accepted inputs.

- [ ] **Step 5: Run the Context owner suite**

```bash
uv run pytest tests/algos/test_fada_context_support_query.py -k 'context or support' -q
uv run ruff check src/unilab/algos/torch/fada_context/support_query.py tests/algos/test_fada_context_support_query.py
```

Stop at this task if a Query-history perturbation cannot change `delta_z_t` while Support is fixed.

### Task 3: Rewire training and deployment through one policy owner

**Files:**
- Modify: `src/unilab/algos/torch/fada_context/support_query.py`
- Test: `tests/algos/test_fada_context_support_query.py`

- [ ] **Step 1: Add failing valid-timestep ownership and differential tests**

Build one pair with two distinct valid Query histories, install deterministic nonzero Context test
weights, and require two distinct residuals while the complete Support object remains identical.
Include padded rows with sentinel values and prove they never enter Context or Tracker. Also assert:

```python
assert torch.equal(output.action, output.action_chunk[..., 0, :])
assert output.action_chunk.shape[-2] == fada_config.prediction_horizon == 6
```

Keep the deployment-path assertion rank-specific: for `[B,K,A]` output, prove
`torch.equal(output.action, output.action_chunk[:, 0, :])` separately.

- [ ] **Step 2: Gather only valid Query rows and scatter without gradient leakage**

In `reconstruct_query`, gather the indices where `valid_window_mask` is true, expand complete Support
rows only by those owning pair indices, and invoke Context/Tracker on exactly that many rows. Scatter
valid results into public `[P,N,...]` tensors using defined zero placeholders for invalid rows;
invalid rows enter neither forward and contribute neither value nor gradient. Never slice Support
along `S`. Evidence is input-row cardinality and pair identity, not Python `forward` call count.

- [ ] **Step 3: Move live Context invocation inside `act_with_context`**

Use the public signature:

```python
def act_with_context(
    self,
    support: SupportContextBatch,
    observation_history: torch.Tensor,
    action_history: torch.Tensor,
    command: torch.Tensor,
) -> ContextActionOutput:
```

Compute Planner Intent with `self.planner(observation_history, command)`, then compute Tracker latent
and Context residual during the same call; add the residual to the current latent and decode six
Actions.

- [ ] **Step 4: Run policy semantic tests**

```bash
uv run pytest tests/algos/test_fada_context_support_query.py -k 'query or action or delta' -q
```

Stop if Context input row count differs from `valid_window_mask.sum()`, owning-pair identity is lost,
or any of the five nonexecuted Actions enters the loss.

### Task 4: Preserve pair-balanced first-action Calibration Learning

**Files:**
- Modify: `src/unilab/algos/torch/fada_context/support_query_training.py`
- Modify: `scripts/train_fada_context_support_query.py`
- Modify: `scripts/preflight_fada_context_support_query.py`
- Test: `tests/algos/test_fada_context_support_query.py`
- Test: `tests/scripts/test_preflight_fada_context_support_query.py`

- [ ] **Step 1: Add regression tests before changing the trainer**

Keep the existing unequal-Query-sample-count fixture and require pair-equal weighting. Retain the decoded
Action chunk gradient in a focused fixture, call `context_first_action_loss`, and require zero
gradient for `action_chunk[..., 1:, :]`. Snapshot every frozen Planner and Tracker parameter
before/after one optimizer step and require exact equality. Inject a frozen-owner mutation inside
`optimizer.step()` and require immediate detection plus frozen-state rollback before validation,
event emission, or any step checkpoint is externally visible; the pre-step checkpoint may remain.

- [ ] **Step 2: Run the training-owner tests**

```bash
uv run pytest tests/algos/test_fada_context_support_query.py -k 'pair or first_action or training_owner or frozen' -q
```

- [ ] **Step 3: Route the new per-timestep residual through the existing owner**

Keep sampling, optimizer, validation, interval/best/final checkpointing and the complete preflight
transaction in `support_query_training.py`. The scripts remain composition roots. Do not add optimizer,
loss, backward, frozen-sentinel, or schema rules to either training script. Expose one owner-level
preflight result API; the preflight script may only resolve dependencies and render it. Give preflight
an explicit artifact-admission mode that calls the shared policy-only artifact owner and reports its
validated method ID, checkpoint schema/step, Query provenance, and `delta_z [P,N,D]`. Add a direct
entrypoint test using real dataset/checkpoint persistence and admission for a minimal v006 artifact,
then supply historical schemas and prove rejection before Context policy, optimizer, or mutable state
construction. Do not monkeypatch persistence or the admission owner; only the external healthy-source
loader may be replaced.

- [ ] **Step 4: Re-run the focused training suite**

```bash
uv run pytest tests/algos/test_fada_context_support_query.py -q
uv run ruff check src/unilab/algos/torch/fada_context/support_query_training.py scripts/train_fada_context_support_query.py
```

Stop on any frozen-parameter mutation, pair-weight drift, or gradient contribution from Actions
`1..5`.

### Task 5: Version artifacts and reject fixed-residual checkpoints

**Files:**
- Modify: `src/unilab/algos/torch/fada_context/support_query_training.py`
- Modify: `scripts/evaluate_fada_context_support_query.py`
- Modify: `scripts/play_fada_context_viser.py`
- Test: `tests/algos/test_fada_context_support_query.py`

- [ ] **Step 1: Add failing schema-isolation tests**

Require a checkpoint identity field equal to `FADA-CONTEXT-METHOD-v006`. Attempt to load v004/v005
fixed-residual checkpoints and missing/wrong method IDs through the active preparation helper and
require explicit rejection before policy construction, optimizer construction, state-dict loading or
any mutable state change. Also construct a schema-v1 single-anchor dataset plus a matching schema-v4
checkpoint digest and require the active helper to reject the dataset before those boundaries.

- [ ] **Step 2: Bump the checkpoint schema in the owner module**

Persist the method Contract ID, Context architecture, `H`, `K`, Support length, source-checkpoint
digest, dataset/split digests, and Context state. Do not implement an automatic tensor/key migration.
Validate raw checkpoint schema and identity before constructing a policy. Provide a policy-only
artifact preparation route for evaluator/playback; they must not materialize an optimizer. Remove the
legacy-dataset flag from that active helper. Replace the exported setup-first resume loader with one
public admission-first resume API: read and validate raw schema/identities, then construct the fresh
Context policy/optimizer and load state. A failed load exposes no partially prepared object.

- [ ] **Step 3: Keep scripts on the shared artifact API**

Both evaluator and playback must continue to call the policy-only
`prepare_context_support_query_artifact`; neither script may load `context_state_dict` or interpret
schema fields directly. Remove legacy-schema opt-ins from every active train/eval/play entrypoint.
Any inspection-only historical reader must have a separate name and remain unreachable from these
routes. The prepared artifact exposes typed, already-validated method/schema/step fields so scripts do
not retain or inspect a raw checkpoint payload.

- [ ] **Step 4: Run persistence tests**

```bash
uv run pytest tests/algos/test_fada_context_support_query.py -k 'artifact or checkpoint or schema' -q
```

Stop if any fixed-residual artifact reaches a policy/optimizer constructor or mutable load boundary.

### Task 6: Recompute `delta_z_t` in evaluation and playback

**Files:**
- Modify: `src/unilab/algos/torch/fada_context/support_query_evaluation.py`
- Modify: `src/unilab/algos/torch/fada_context/support_query.py`
- Modify: `src/unilab/visualization/interactive_playback.py`
- Modify: `scripts/play_fada_context_viser.py`
- Modify: `conf/distill/context_playback/left_knee_070.yaml`
- Test: `tests/algos/test_fada_context_support_query_evaluation.py`
- Test: `tests/algos/test_fada_playback.py`

- [ ] **Step 1: Add a failing closed-loop call-sequence test**

Use a fake Context Encoder that records histories. For a three-step second rollout require three
Context input rows, unchanged complete Support identity, and three successively updated State/Action
histories. Healthy and fault-zero branches must produce zero Context calls.

- [ ] **Step 2: Remove precomputed fixed residuals**

Delete evaluator/playback calls that compute `delta_z = context_encoder(support)` before rollout.
Healthy and fault-zero use the existing frozen Planner-IDM policy with no Context call. Only the
fault-Context branch retains complete Support and passes it to `act_with_context` every control
cycle. Record the Context residual trace as `[T,B,D]`, aggregate its norms deterministically, and
version the report schema because the former single-residual diagnostic meaning changed.

- [ ] **Step 3: Put Support binding in a policy-owned playback seam**

Add a narrow Support-bound callable adjacent to `FrozenIDMSupportQueryPolicy`. It retains immutable
Support plus its outer `support_command` provenance, validates the current Command before the first
Context-conditioned Action, and adapts to `FADAPlaybackController`'s existing three-input callable.
The playback script only selects artifacts and assembles this owner; it must not contain residual or
Command-matching rules.

Give `FADAPlaybackSession` a public controller-binding method owned by the visualization layer; it
must reject a non-`policy` action mode and update the public policy callable without script access to
`_fada_policy`. Make the Context playback preset compose `interactive.action_mode=policy` explicitly.
Add a config/session-factory test that composes the real preset, binds through the public method, steps
the session, and proves the Context controller reaches the first Action. A Context-enabled non-policy
combination must fail closed instead of silently producing zero Actions.

- [ ] **Step 4: Prove frozen receding-horizon behavior**

Run under `torch.inference_mode()`, require Planner/Tracker/Context parameters unchanged, require one
six-step chunk per cycle, and send only `output.action` to `env.step`.

- [ ] **Step 5: Run consumer suites**

```bash
uv run pytest tests/algos/test_fada_context_support_query_evaluation.py tests/algos/test_fada_playback.py -q
uv run ruff check src/unilab/algos/torch/fada_context/support_query_evaluation.py src/unilab/visualization/interactive_playback.py scripts/play_fada_context_viser.py
```

Stop before simulator work if the fake closed loop does not prove history advancement and per-cycle
Context invocation.

### Task 7: Complete bounded verification and governance closeout

**Files:**
- Modify after evidence exists: `note/fada/checklists/current.md`
- Modify after evidence exists: `note/fada/task_canvas.md`
- Modify after evidence exists: active Context Contract implementation status
- Modify after evidence exists: `note/governance.json`

- [x] **Step 1: Run the focused static gates**

```bash
uv run ruff check src/unilab/algos/torch/distill/fada.py src/unilab/algos/torch/fada_context src/unilab/visualization/interactive_playback.py scripts/train_fada_context_support_query.py scripts/evaluate_fada_context_support_query.py scripts/play_fada_context_viser.py scripts/preflight_fada_context_support_query.py tests/algos/test_fada_planner_idm.py tests/algos/test_fada_context_support_query.py tests/algos/test_fada_context_support_query_evaluation.py tests/algos/test_fada_playback.py tests/scripts/test_preflight_fada_context_support_query.py
uv run ruff format --check src/unilab/algos/torch/distill/fada.py src/unilab/algos/torch/fada_context src/unilab/visualization/interactive_playback.py scripts/train_fada_context_support_query.py scripts/evaluate_fada_context_support_query.py scripts/play_fada_context_viser.py scripts/preflight_fada_context_support_query.py tests/algos/test_fada_planner_idm.py tests/algos/test_fada_context_support_query.py tests/algos/test_fada_context_support_query_evaluation.py tests/algos/test_fada_playback.py tests/scripts/test_preflight_fada_context_support_query.py
```

- [x] **Step 2: Run the complete affected test set**

```bash
uv run pytest tests/algos/test_fada_planner_idm.py tests/algos/test_fada_context_support_query.py tests/algos/test_fada_context_support_query_evaluation.py tests/algos/test_fada_playback.py tests/scripts/test_preflight_fada_context_support_query.py -q
npm --prefix note/architecture/auxiliary/atlas_app run check
uv run python /Users/chengyuxuan/.codex/skills/repo-architecture-atlas/scripts/validate_atlas_manifest.py note/architecture/atlas_manifest.json --target design-inspector
uv run python /Users/chengyuxuan/.codex/skills/workflow-governance/scripts/validate_governance_manifest.py note/governance.json --target closeout
git diff --check
```

- [x] **Step 3: Obtain producer-owned receipts**

Require `module-alignment-test: MODULE-CORRECT` and `code-review-expert: FINAL_GATE_PASS`. Only then
request `formal-runtime-audit` for the official entrypoints. Governance must not infer these statuses
from pytest output.

- [x] **Step 4: Update authority atomically**

Do not perform a second Contract activation. Update implementation status, checklist, Atlas,
test inventory/control board, module/formal/code-review receipts and evidence ledger against the
already-active Context v006/v005 identities. Preserve fixed-residual Contracts and evidence only
under history.

- [x] **Step 5: Keep expensive work separately authorized**

Do not run `make test-all`, MuJoCo, Context training, evaluation, commit, or push unless their later
transition and authority gates explicitly require and authorize them. The existing compatible v005
source checkpoint remains the frozen Planner–Tracker source for this repair.
