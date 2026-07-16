contract_id: DISTILL-TRAIN-v002
status: active
effective_date: 2026-07-16
updated_date: 2026-07-16
supersedes: DISTILL-TRAIN-v001
method_contract: DISTILL-METHOD-v001
concept_figure: note/architecture/concept/03_g1_multiteacher_distillation_method.data.json

# Single-Entry Multi-Role Distillation With Transition-State Evidence

## Public Contract

The formal training workflow remains one resumable entry over immutable role
artifacts. Existing stand and walk artifacts may be reused independently. This
version activates one additional scenario contract: `walk_to_stop`. It is a
scenario in the student-state DAgger loop, not a third teacher or a third MoE
expert.

The existing `DISTILL-TRAIN-v001` stage machine, resume/fork identity rules,
role preflight, and cumulative DAgger rules remain active unless this contract
states a transition-specific rule.

## Transition Scenario

Every transition row has a canonical `scenario_label` and
`transition_age`:

| Scenario | Teacher/intent semantics | `transition_age` |
| --- | --- | --- |
| `static_stand` | no velocity command; standing teacher; `inactive` intent | `-1` |
| `walk_flat` | any velocity command; walking teacher; `active` intent | `-1` |
| `walk_to_stop` | walking teacher before the atomic zero-command switch, standing teacher after it | pre-switch `-1`, first post-switch row `0`, then increasing |

`role_labels` continues to identify the teacher role used for the target row:
`walk_flat` before the switch and `stand` after it. `scenario_labels` records
the episode-level reason for the row and must not be inferred from a
checkpoint filename.

For `walk_to_stop`, each row also preserves `command_before` and
`command_after`. The switch is one environment command update; post-switch
`command_after` is exactly the zero command. The teacher action is detached
and corresponds to the teacher selected for that row. `student_obs` is the
state presented to the distilled student, while `teacher_obs` follows the
selected teacher contract.

## Ownership

- `data.py` owns validation, batch slicing, persistence, and fail-closed
  multi-source merging of transition fields.
- `collector.py` owns the transition scenario state machine, atomic command
  switch, done/reset handling, teacher selection, and collection metadata.
- `dagger.py` owns cumulative iteration/update sequencing and checkpoint
  lineage; it does not reinterpret transition fields.
- `trainer.py` consumes the fields as batch metadata and keeps behavior loss
  ownership unchanged.
- `scripts/train_distill.py` and Hydra only dispatch/configure the owner; they
  do not implement transition semantics.

## Required Schema

The new fields are optional for legacy stand/walk artifacts so compatible
datasets remain loadable. If any transition field is present, the schema must
validate the complete transition contract. The transition fields are:

- `scenario_labels: tuple[str, ...]`
- `transition_ages: int64 tensor[N]`
- `command_before: float tensor[N, 3]`
- `command_after: float tensor[N, 3]`

Missing fields are not silently padded. A multi-source merge fails closed when
sources disagree about transition-field presence or dimensions.

## Forbidden Behavior

- Do not add a transition expert or treat `walk_to_stop` as a third role.
- Do not use a soft MoE mixture as proof of pure stand/walk teacher fidelity.
- Do not collect active rows for `static_stand` or zero-command rows for
  `walk_flat`.
- Do not silently synthesize transition fields for legacy artifacts.
- Do not claim student recovery from schema, offline MSE, or finite actions;
  repeated-reset and walk-to-stop live evidence remain separate gates.

## Required Evidence

1. S1 schema tests cover validation, batch slicing, save/load roundtrip,
   multi-source merge, and malformed transition rows.
2. S2 collector tests prove the atomic command switch, teacher-role change,
   age reset, done/reset handling, and bounded sample counts.
3. A runtime probe records pre/post command, selected teacher role,
   transition age, finite actions, and reset boundaries.
4. Formal workflow evidence records whether legacy role artifacts were reused
   and which transition artifact was collected.
5. Physical acceptance remains a separate repeated-reset/live sentinel gate.

## Stop Conditions

Do not start a long transition DAgger run until the schema and collector owner
tests pass, the transition dataset has an inspectable manifest, and the
student checkpoint lineage identifies its parent and teacher artifacts.
