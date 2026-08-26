# FADA v012 Module Test Cards

Status: Unit A module tests implemented and passing; formal runtime and policy-quality evidence pending.

## MTC-A — Privileged Oracle and lineage

- **Input:** one G1WalkFlat config, typed privilege bundle, seed, and iteration.
- **Output:** SAC actor input plus a checkpoint carrying one `oracle_lineage_id`.
- **Invariant:** the actor directly consumes privileges; exactly iterations `240…4800` are intermediate and `5000` is final; all share one lineage.
- **Negative cases:** distilled teacher, mixed lineage, wrong/missing index, critic-only privilege, non-finite privilege.
- **Evidence owner:** Oracle contract tests, formal runtime audit, then policy-quality audit.

## MTC-B — Reward and domain distribution

- **Input:** task Reward and DR config before environment construction.
- **Output:** admitted single-task Oracle config.
- **Invariant:** no gait/feet-phase reward owns credit; gait phase may only be observed; agreed DR families are active.
- **Negative cases:** any non-zero `feet_phase` or equivalent phase-conditioned scale, missing DR family, backend-private hot-path probe.
- **Evidence owner:** config composition tests plus runtime DR probes.

## MTC-C — Planner–IDM causal input

- **Input:** raw 98-vector history and command.
- **Output:** Planner `H×95 + command3 → K×66`; IDM `H×66 + H×29 + K×66 → K×29`.
- **Invariant:** previous action is visible in Planner history but absent from future; residual anchor is latest state66; no supervised action leakage.
- **Negative cases:** 66-only Planner history, 98-vector passed unsplit, action appended to future, command embedded ambiguously.
- **Evidence owner:** exact-shape tests, action-leakage counterfactual, model forward/gradient tests.

## MTC-D — Source ownership and optimization

- **Input:** admitted 20+1 Oracle lineage and causal windows.
- **Output:** new-schema Planner–IDM checkpoint.
- **Invariant:** intermediate checkpoints are IDM-only coverage; final Oracle owns all action labels; each round is IDM then Planner; IDM is frozen during Planner update without detaching the future gradient; first action is supervised and executed.
- **Negative cases:** intermediate Planner label, standing/transition role, mixed lineage, simultaneous optimizer ownership, v011 resume.
- **Evidence owner:** collection/admission tests, optimizer mutation tests, persistence tests, formal runtime audit.
