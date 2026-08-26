# FADA v011 Formal Runtime Audit Plan

> Scope: one offline Global Simplified Formal Test. No server, simulator, training campaign,
> deployment, Git commit, or push is authorized.

## Admitted identity

- Design: `FADA-METHOD-v011:FADA-TRAIN-v011`
- Checkout: `main@2b5278fc3987a9696a4946abc13c0b6a948e840d` plus production diff
  `sha256:0eaa57f3d45e89559ff3bda28ae1984554f15b02682217b5d3f4bd2f6e19f61e`
- Module receipt: `2026-08-26-unified-oracle-alternating-module-test.json`, status
  `MODULE-CORRECT`
- Official entry: `scripts/train_distill.py:main -> run_fada_training_owner -> persistent_async`
- Student starts cold; final Oracle is a deterministic local schema-valid 98-to-29 distillation
  checkpoint; 20 intermediate checkpoint identities use a deterministic external-policy adapter.

## Critical design-point matrix

| ID | Contract and owners | Producer -> carrier -> consumer | Necessary capability and witness | Falsifier |
|---|---|---|---|---|
| DP-01 | v011 unified Oracle; `load_fada_oracle_policy`, worker | distillation checkpoint -> composition and spawned worker -> all main scenarios | one actual frozen checkpoint identity labels walk, static stand, and walk-to-stand | second final policy, wrong format/dimensions, or scenario-specific final identity |
| DP-02 | schema-4 source and v005 replay; collector, artifact admission, replay | spawned fake simulator state -> production collector/artifact -> replay sampler | iteration 0 Oracle and later student rollouts produce 50/25/25 scenarios, cold-start roles, 1:2 retention, and historical replay consumption | semantic artifact synthesized by the rig, zero intermediate rows, rejected quotas, or no later replay use |
| DP-03 | alternating trainer; `FADATrainer` | admitted replay -> IDM updates then fixed-IDM Planner updates -> both losses and parameter owners | every one of three iterations executes exactly one IDM update before one Planner update | missing loss, simultaneous step, wrong order, or frozen owner mutation |
| DP-04 | persistent lifecycle; production runtime, shared weights, worker | saved current student -> shared-weight activation -> spawned collector request | versions 1/2/3 activate before matching collection and spawned resources close once | version mismatch, process leak, duplicate collection, or stale student rollout |
| DP-05 | schema-5 persistence; checkpoint writer/reader/policy | trained policy and optimizers -> atomic checkpoint -> strict reload and first action | schedule, both optimizer states, counters and 29-D finite first action survive round trip | temp residue, missing owner state, wrong schedule/counter, load failure, or invalid action |

## Global Simplified Formal Test

Modify only `tests/algos/test_fada_formal_runtime.py`:

1. Keep the production Hydra entrypoint, FADA composition root, `persistent_async` branch,
   production `build_persistent_fada_runtime`, `PersistentDistillationRuntime`, spawned runner,
   `PersistentFADACollectorWorker`, collector, artifact, replay, trainer, and checkpoint owners.
2. Replace only external effects through the existing worker-factory dependency: a top-level,
   spawn-picklable wrapper constructs the same production worker while injecting a deterministic
   98-observation/29-action fake simulator and deterministic intermediate-policy loader.
3. Do not replace the final Oracle loader. Materialize a real distillation checkpoint and let the
   parent composition and spawned worker load it through `load_fada_oracle_policy`.
4. Run three outer iterations, 12 main plus 24 intermediate windows per iteration, one IDM and one
   Planner update per iteration, replay capacity 96. This is the minimum horizon that crosses
   Oracle bootstrap, later student rollout, replay eviction/retention, repeated shared-weight
   activation, and final persistence. Each collection is bounded at 72 environment steps, enough
   to cross the configured 30-step active history plus 36-step stopped future and emit all rows.
5. Assert route identities, scenario/role counts, rollout modes, active replay, both losses,
   schema-5 fields, strict reload, finite first action, process cleanup, and no temporary residue.

The fake simulator owns only deterministic external state transition and lifecycle. It does not
construct `FADASourceBatch`, choose source roles, implement replay, losses, updates, or persistence.
The intermediate policy adapter owns only external policy outputs; source identity and eligibility
remain production-owned.

## Stop conditions

Stop on the first production branch mismatch, non-picklable seam, worker startup failure, semantic
artifact generated outside the collector, quota/role mismatch, zero update, persistence mismatch,
resource leak, or test-only production hook requirement. A failure is evidence, not permission to
weaken the test. PASS proves only R1 composition and applicable R2 persistence; it does not prove
MuJoCo behavior, convergence, locomotion quality, robustness, or server checkpoint availability.
