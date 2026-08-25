---
contract_id: FADA-TRAIN-v010
status: active
effective_date: 2026-08-25
supersedes: FADA-TRAIN-v009
method_contract: FADA-METHOD-v010
scope: fresh persistent-async two-stage Planner-IDM training preparation
---

# FADA Two-Stage Source Training Contract v010

The official route remains `persistent_async`, but each command owns exactly one phase through
`training.fada.phase`.

## IDM pretraining command contract

- `phase=idm_pretrain`;
- fresh initialization only; resume and warm start remain disabled;
- Oracle rollout for every main-source iteration plus the configured 20 intermediate Oracles;
- only `idm_updates` is active;
- output checkpoint records completed IDM-pretrain identity.

## Planner training command contract

- `phase=planner`;
- `pretrained_idm_path` is required and must name a completed schema-4 `idm_pretrain` checkpoint;
- `checkpoint_path` must resolve to a different file than `pretrained_idm_path`;
- only IDM weights transfer and Planner starts fresh;
- IDM is permanently frozen in eval mode and canonical tensor identity is checked across every
  update/save;
- iteration zero uses Oracle rollout and later iterations use current Planner-IDM rollout;
- intermediate Oracle collection and 1:2 IDM replay retention are disabled;
- only `planner_updates` is active.

The exact `66/29/3`, `H=30`, `K=6` dimensions, source schema 4, source roles, final and intermediate
Oracle identities, replay retention, and scenario/cold-start quotas remain unchanged. Schema-4
training checkpoints contain exactly one phase-owned optimizer. Playback of older schema
checkpoints remains supported but cannot admit them into v010 training. v010 training resume is
not supported.

Long training requires fresh formal-runtime audit, exact server paths, and separate launch
authorization. This contract authorizes only local reversible implementation and offline proof.
