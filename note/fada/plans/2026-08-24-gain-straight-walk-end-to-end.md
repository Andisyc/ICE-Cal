# Gain × Straight-Walk End-to-End Pipeline Plan

> **For agentic workers:** this plan is documentation only. The code/training freeze remains in
> effect; no production edit, retraining, or Stage execution is authorized by this file.

**Status:** ACTIVE VOLATILE PLAN — first transaction under v009/v008 (2026-08-24)
**Design:** Design Inspector 09, seven-card layout (`ICA-DP-01..07`)
**Contract:** `FADA-CONTEXT-METHOD-v009` + `FADA-CONTEXT-TRAIN-v008`; this plan is the first bounded
`m=1` transaction and does not authorize multi-component evidence, real-robot work, or long training.

**Goal:** Run the full calibration chain once at minimum complexity — single task (straight walk) ×
single fault (gain attenuation) × single correction direction — to validate chain connectivity and
correctness end to end before admitting multi-fault joint DR and multi-component separation.

**Why scoped:** The settled design (Basis Discovery card) samples all perturbation axes jointly and
separates k principal components with PCA/SVD. Both the separation and the denoising machinery are
deferred: with gain as the only fault there is nothing to separate, and the smoke direction is the
mean of a small batch of command-normalized per-sample `δz*` vectors — no SVD, no PCA. What is NOT
deferred is state dependence: a fixed state-independent vector is the known-failed v1 mode
(shared-vector compensation ratio 0.81 < 0.9 on gain), so even the smoke direction is applied scaled
by current command magnitude.

## Scope

- Task family: straight-line walking commands only.
- Fault: action-channel gain `g ∈ [g_min, g_max]` (normalized `c = 0 ≡ nominal`, `c = ±1 ≡ max
  covered strength`; range values pinned in Task 0).
- Chain: DR rollout → per-sample `δz*` optimization → direction extraction (mean of
  command-normalized `δz*`, no SVD/PCA) → operator freeze → Coefficient Encoder training → σᵢ sweep
  fit → frozen deployment readout.

## Non-scope

- Multi-fault joint sampling and multi-component separation of mixed signatures.
- Delay/offset axes (the delay-operator convention `Δ(z) = shift_k(z) − z` with constant
  identification-determined `k` is recorded but not exercised).
- Held-out combination evidence (requires `m ≥ 2`; single-axis transaction proves chain connectivity
  and the single-axis method case only).
- Real-robot anything; all evidence is simulation-bound.

## Tasks

- [x] **Task 0 — Engineering parameters (PINNED 2026-08-24, human-approved).**
  - `δz*` solver: Adam, lr = 0.01, max 200 steps, `δ` init at 0, early stop at relative residual
    < 1e-3 (min-norm via zero-init + early stop, no explicit L2).
  - Target: full 6-step executed action chunk (aligned with K = 6 future tokens).
  - Gain range: `g ∈ U(0.8, 1.2)`, `c = (g − 1)/0.2` (`c = 0 ≡ nominal`).
  - Sampling: 32 gain values × 4 rollouts (≈ 60k per-sample solves).
  - Command-magnitude normalization: `s = ‖cmd‖` (velocity-command norm); extract `δz*/s`, apply
    `u × s`.
  - Straight-walk command range: forward `v ∈ U(0.3, 1.0)` m/s, no turning.
  - Direction averaging: all converged samples (final relative residual < 1e-2).
  - σ sweep grid: `c ∈ [−1, +1]`, 21 points (step 0.1) × 32 rollouts per point.
- [ ] **Task 1 — Data audit / collection.** Check whether the existing server-side gain rollout
  (`calibration_dataset_gain_v2.pt`) retains the full causal tuple `(history, realized future,
  executed action, θ = g)` per step. If complete, reuse; if any field is missing, recollect with a
  gain-only DR rollout on straight-walk commands. Server access requires separate user authority.
- [ ] **Task 2 — Per-sample `δz*` solver.** Target factory exactly as the Basis Discovery card:
  `z_failed = Encoder(history, realized future)`; objective is non-self-consistent because the input
  is the realized future; solve from `δ = 0` for the min-norm solution.
- [ ] **Task 3 — Direction extraction (smoke form).** Solve a small batch of `δz*` on straight-walk
  data, normalize each by its command magnitude, take the mean unit direction `u` — no SVD, no PCA.
  Sanity check (not a gate): per-sample normalized directions cluster tightly around `u`, and their
  pre-normalization magnitudes correlate with injected `g` (identification is trivial here since
  only one axis exists).
- [ ] **Task 4 — Stage 1 operator freeze.** Type the component as rank-1 multiplicative:
  `Δ(z) = u · cmd_mag` with the same command-magnitude scalar used at extraction (fallback rank-1
  latent form `u·(uᵀz)` pinned in Task 0 if the command scalar proves insufficient); freeze it. A
  fixed state-independent vector is explicitly excluded (known v1 failure: 0.81 < 0.9).
  **S1 gate:** shared-operator compensation ratio ≥ 0.9 across states; no threshold
  relaxation permitted.
- [ ] **Task 5 — Stage 2 Coefficient Encoder.** `c_true` = least-squares projection of `δz*` onto
  the frozen operator output; loss = coefficient anchor + `0.1` action-consistency safety net;
  gradient enters the Coefficient Encoder only. **S2 gate:** `|ĉ − c_true| ≤ 0.05` on the validation
  grid.
- [ ] **Task 6 — Stage 3 σ fit.** Sweep the normalized grid, record `(ĉ, error)`, fit monotone
  PCHIP; saturation + alarm at endpoints, no extrapolation. **S3 gate:** `R² ≥ 0.95` and monotone.
- [ ] **Task 7 — Deployment smoke.** Frozen readout → `z̄ = z + σ(ĉ)·Δ(z)` → frozen Tracker;
  verify `c = 0` is exactly the nominal path; confirm a held-out gain value inside the covered range
  recovers tracking error versus the uncalibrated baseline.

## Acceptance

- S1/S2/S3 gates all pass without threshold relaxation.
- `c = 0` identity holds exactly.
- One held-out gain strength inside `[g_min, g_max]` shows end-to-end error recovery.

## Deferred after this plan

- Joint DR + PCA separation (the full Basis Discovery design), plus the SVD/denoising machinery —
  the smoke uses a plain mean of normalized `δz*` instead.
- Delay/offset axes and their operator conventions.
- Contract changes after this run; any semantic change requires a new human review receipt.
- Real-robot residual monitoring and basis growth (new-component coefficients read by residual
  projection — the default recorded 2026-08-24).
