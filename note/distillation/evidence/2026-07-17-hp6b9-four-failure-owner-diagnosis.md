# E81 — HP-6b9 Remaining Four-Failure Owner Diagnosis

Result: **PASS (diagnosis only)**

The four exact E79 nodes reproduce as four failures in 0.48s. No implementation,
test, dependency, generated doc, or environment state was changed.

## Owner classification

### Stewart (2)

- First failed boundary: `create_backend()` raises `ImportError: MotrixSim not
  available` before env construction/step/IK.
- Owner: optional dependency provider plus Stewart test selection/marker.
- Causality: environment/baseline, not current branch code. The two tests,
  Stewart owner, and backend factory are HEAD-identical. `pyproject.toml`
  declares Motrix under optional extra `motrix`, while these two non-slow tests
  do not skip when the provider is absent.
- Not proven: Stewart runtime correctness with Motrix installed.

### Generated support matrix (1)

- First failed boundary: docs checker reports generated support matrix stale.
- Owner: derived support-matrix document.
- Exact drift: generated output adds the already tracked SAC tasks
  `g1_stand_still` and `g1_walk_height` after `g1_wall_flip_tracking`.
- Causality: HEAD-baseline derived-doc drift. Generator, document, and both SAC
  owner configs are unchanged by this branch.

### CLI local checkpoint (1)

- First failed boundary: the fake subprocess expects the temporary checkout's
  `.venv`, but receives the outer frozen worktree environment path.
- Owner: repository test execution-environment identity.
- Causality: invocation-induced, not checkpoint resolution and not a branch
  code change. `run_demo()` uses `env.setdefault("UV_PROJECT_ENVIRONMENT", ...)`;
  the exact E79 command exports that variable globally, so the test's temporary
  `_repo_root()` cannot replace it. `demo.py` and the test are HEAD-identical.
- Not proven: the same test under an environment with that variable unset; no
  fifth probe was run in this diagnosis-only gate.

## Next owner gates

These are three independent gates: Motrix test/provider policy, generated docs
refresh, and test execution-env isolation. They must not be repaired together.
