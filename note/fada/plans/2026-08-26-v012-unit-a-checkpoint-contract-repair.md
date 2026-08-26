# FADA v012 Unit A checkpoint-contract repair

Status: `READY FOR LOCAL CONSTRUCTION`

Authority receipt: `USER-CODE-CONSTRUCTION-20260826`. Local production code, tests, review,
module evidence, and formal-audit evidence are authorized. Training, server work, simulation,
Git operations, and policy-quality claims remain unauthorized.

## Requested and preserved behavior

Requested:

1. Every Unit A checkpoint is self-identifying under `FADA-TRAIN-v012`.
2. Missing or incompatible identity rejects before learner/model/optimizer mutation.
3. The production persistence owner validates exactly twenty `idm_coverage` checkpoints plus one
   `final_oracle` checkpoint from one lineage.
4. Module evidence is validator-compliant and tied to the exact design and checkout.

Preserved:

- `scripts/train_offpolicy.py` remains the official entrypoint and assembly layer.
- Generic SAC checkpoint persistence remains byte-for-byte equivalent when no custom saver is
  injected.
- Unit A remains cold-start only, G1WalkFlat/MuJoCo, 5,000 iterations, save interval 240, and no
  gait/phase Reward.
- No v011 or incomplete v012 checkpoint becomes a compatibility fallback.

## Engineering boundary record

| Concern | Unique owner | Public boundary |
|---|---|---|
| Canonical checkpoint identity and strict validation | `fada_privileged_oracle.py` | immutable contract/identity records and strict payload reader |
| FADA learner pre-mutation load guard | `FADAPrivilegedSACLearner` | `load_state_dict` |
| FADA save, reload verification, and final 20+1 manifest | FADA checkpoint gateway | callable saver injected into the runner |
| Generic persistence dispatch | `DoubleBufferOffPolicyRunner` | optional checkpoint-saver dependency; default remains `torch.save` |
| Full-config and env-materialized identity assembly | `FADAPrivilegedSACRuntime` | training-only kwargs and checkpoint saver construction |
| Asset/layout/order cold-path projection | `G1WalkEnv` plus declared `SimBackend` capability | immutable public identity snapshot created during env initialization |

Allowed dependencies point from the training composition root to the optional runner port and from
the FADA runtime to the G1 public identity snapshot. The runner must not import FADA. The env must
not import learner or checkpoint code. No hot-path asset reads, `getattr`/`hasattr` capability
probing, filename-only role inference, or script-owned business validation is allowed.

## State-schema migration contract

Primary mode: `STATE_SCHEMA`.

- Donor: incomplete local v012 payload schema, never admitted for training or publication.
- Target: sealed v012 Oracle checkpoint schema owned by the FADA checkpoint gateway.
- Mixed-version policy: forbidden.
- Old/incomplete writer → new reader: reject before mutation.
- New writer → old reader: unsupported and isolated by the first-campaign cold-start/no-resume rule.
- New writer → new reader: strict save/load round-trip.
- Rollback: revert the local uncommitted repair; no trained v012 artifacts exist to migrate.

## Confirmed module Test Card

Card identity: `FADA-V012-UNIT-A-CHECKPOINT-20260826`, confirmed by the active Contract and the
user's authorization to execute the three repairs identified by the formal audit.

- Ordinary: an asymmetric identity with distinct hashes, dimensions, body order, actuator-joint
  order, and iteration 240 round-trips exactly.
- Boundary: iterations 240 and 5000 map only to `idm_coverage` and `final_oracle` respectively;
  iteration 0/4801 rejects.
- Invalid: one missing or mismatched field rejects before a learner parameter or optimizer state is
  changed.
- Identity/order: changing lineage, config hash, body order, actuator order, task/backend/action
  scale, layout, or dimensions rejects despite shape-compatible tensors.
- Lifecycle: the saver writes/reloads each payload, and finalization admits exactly the required
  20+1 records; a missing, extra, duplicate, mixed-lineage, or filename/payload disagreement fails.

Independent oracle: explicit field equality, exact iteration-role table, exact mutation count zero
on rejection, and the hand-enumerated set `{240,480,...,4800,5000}`. The pre-repair payload is the
controlled counterexample.

## TDD and implementation sequence

1. Add RED tests for complete identity, strict pre-mutation rejection, exact 20+1 finalization, and
   default generic runner persistence.
2. Add the immutable identity/contract and canonical JSON hashing in the FADA owner.
3. Add the FADA checkpoint gateway and learner load guard.
4. Add the narrow optional saver port to the double-buffer runner; keep the default path unchanged.
5. Add runtime composition and one cold-path G1 identity projection, declaring any needed backend
   method in `SimBackend` before implementing MuJoCo.
6. Run focused GREEN tests, affected runner/runtime tests, G1 tests, lint, and `git diff --check`.
7. Emit and validate a current `module-alignment-test` manifest.
8. Perform `code-review-expert` migration and final-gate reviews.
9. Rerun the same formal runtime audit. It may return PASS, HOLD, or one `LIVE_REQUIRED` fact; it
   must not start long training.

## Stop conditions

Stop and ask the user only if implementation reveals a new semantic choice about accepted fields,
mixed-version support, task schedule, or live execution. Ordinary owner-layer mechanics and failing
regressions remain inside this authorization.
