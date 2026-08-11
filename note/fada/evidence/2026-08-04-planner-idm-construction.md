# Planner-IDM Construction Evidence

Date: 2026-08-04

Scope: generic Planner, IDM, composed receding-horizon policy, and paper-defined source loss boundaries.

## Evidence

- E1: local extraction of `/Users/sss9999/locomotion/paper/精确控制/FADA.pdf`, Sections 4.1 and Appendix B.1-B.2.
- E2: `uv run --frozen --no-sync pytest tests/algos/test_fada_planner_idm.py -q` -> `6 passed`.
- E3: focused `ruff check` -> `All checks passed!`.
- E4: focused `pyright src/unilab/algos/torch/distill/fada.py` -> `0 errors, 0 warnings, 0 informations`; public import smoke returned `FADAPlannerIDMPolicy`.
- E5: `npm --prefix note/architecture/auxiliary/atlas_app run check` -> viewer/data contracts OK and Atlas OK; browser verification rendered all six Inspector tabs and switched from Command Coverage to Planner Interface with the shared spine unchanged.
- E6: `note/fada/reviews/2026-08-04-planner-idm-final-gate.md` -> APPROVE, no P0/P1/P2 findings after annotation pass.

## Facts

- Planner defaults match the paper: `H=30`, `K=6`, hidden size 128, 4 heads, 3 encoder layers.
- IDM defaults match the paper: 3 history-encoder layers, 2 future-decoder layers, 4 heads, hidden size 128, full non-causal future self-attention.
- Planner reconstructs future proprioception as a residual from the latest observation.
- IDM source loss uses a causally matched realized-future/executed-action pair and supervises only the first action.
- Planner source loss uses the oracle first-action relabel through a fixed IDM; Planner receives gradients and IDM parameters do not.
- The combined policy outputs a full future/action chunk and exposes only the first action for receding-horizon execution.

## Decisions

- Task-owned `obs_dim`, `action_dim`, and complete `command_dim` remain constructor inputs rather than hardcoded G1 values.
- Paper-unspecified Transformer details are recorded as local implementation choices in `FADA-METHOD-v001`.

## Open risks

- No formal simulator collector or snapshot/oracle-shadow integration was implemented; it is outside the requested module-construction scope.
- Task-specific command ranges, valid combinations, temporal schedule, and rollout coverage acceptance remain unresolved and therefore block calling a future rollout dataset complete.

## Next

Begin rollout-owner design only after the command coverage details are fixed. Do not begin target adaptation or LoRA work under the current scope.
