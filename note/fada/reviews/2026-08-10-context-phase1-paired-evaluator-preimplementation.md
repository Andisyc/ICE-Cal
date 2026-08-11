# Context Phase-1 Paired Evaluator Preimplementation Review

Date: 2026-08-10
Mode: `plan_review`
Verdict: approved for bounded implementation

## Findings

No Critical blocker was found.

Important: separate environment construction with the same integer seed is weaker than exact branch
pairing because constructor/reset call order can drift. The evaluator must capture one initialized
`NpEnv` snapshot and restore that snapshot between nominal and teacher branches.

Important: autoreset would replace a fallen row with a new episode and contaminate displacement and
fall statistics. Both branches must disable autoreset and stop accumulating a row after its first
termination or truncation. Termination and truncation must remain separate.

Important: world-frame lateral displacement is ambiguous when reset yaw is nonzero. Position must be
rotated into each row's initial yaw frame and yaw drift wrapped to `[-pi, pi)`.

Important: the teacher checkpoint must reconstruct the actor through the Phase-1 runtime config and
must pass nominal SHA/tensor identity validation before simulation.

## Owner map

- `G1BaseEnv`: public base pose access used by evaluation; no metric policy.
- `fada_context.paired_evaluation`: snapshot pairing, rollout lifecycle, metrics, strata, aggregation.
- `evaluate_context_teacher_phase1.py`: Hydra composition, checkpoint/env construction, seed loop,
  structured JSON output.
- Tests: fake deterministic dynamics for metric semantics plus one bounded real MuJoCo sentinel.

The evaluator is an explicit entrypoint and therefore default-off. No threshold or training launch is
authorized by this review.
