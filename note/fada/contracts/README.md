# FADA Contract Registry

Default recall reads only the active contracts listed here.

| Contract | Category | Status | Scope |
|---|---|---|---|
| [FADA-METHOD-v024](active/method/FADA-METHOD-v024.md) | method | active / implemented / offline-tested | Exact command gate, deterministic regime-transition phase state machine and bounded contact cost |
| [FADA-TRAIN-v024](active/training/FADA-TRAIN-v024.md) | training | formal-runtime-blocked / not-run | Offline module proof complete; formal route and bounded-policy evidence remain |
| [FADA-METHOD-v022](active/method/FADA-METHOD-v022.md) | method | active | Live normalized privileged Actor with iteration-based grouped DR curriculum, then Planner–IDM |
| [FADA-TRAIN-v022](active/training/FADA-TRAIN-v022.md) | training | active / persistence-blocked | Successful validation route exists; sealed 20+1 grouped-DR lineage is still missing |
| [FADA-ADAPT-METHOD-v003](active/method/FADA-ADAPT-METHOD-v003.md) | method | active | Q/V-only LoRA across all IDM attention from non-leaking target windows |
| [FADA-ADAPT-TRAIN-v003](active/training/FADA-ADAPT-TRAIN-v003.md) | training | active | Schema-5 source, v3 adapted checkpoint, legacy isolation, and 6000-step target budget |
| [FADA-CONTEXT-METHOD-v009](active/method/FADA-CONTEXT-METHOD-v009.md) | method | active | Data-driven task-relevant correction basis, frozen Tracker injection and coefficient readout |
| [FADA-CONTEXT-TRAIN-v008](active/training/FADA-CONTEXT-TRAIN-v008.md) | training | active | Serial basis discovery, operator freeze, coefficient training and scale evidence |

The v024 pair supersedes the unrun v023 phase-only route. v024 restores one mixed standing/walking
Oracle without restoring competing Standing and Gait regime owners: exact command normalization
selects the regime, null commands expose `[pi,pi]`, non-null transitions restore pi separation, and
a bounded negative contact-mismatch term owns gait timing. Translation and yaw jointly determine
the non-null clock. The confirmed module and Reward ordering specifications are
`../testing/v024_module_test_cards.md` and `../testing/v024_reward_ordering_card.json`.
The design preserves v022 privilege, broad physical DR and sealed `240…4800 + 5000` lineage
requirements. The left-knee-only actuator-strength axis and its curriculum are disabled; generic
all-joint Kp/Kd randomization remains enabled. The v024 implementation and
`mujoco_fada_phase_contact` selector exist with offline Module Test evidence and a bounded official
reset/step smoke. There is no full v024 formal runtime receipt, checkpoint, training, or policy
evidence.
`mujoco_fada_phase` remains a historical v023 compatibility profile, not a v024 launch command.

The active Context pair is a separate data-driven-basis engineering proposal. v008/v007 analytic-axis
implementation and review receipts became historical when the target and component identity changed.
Official-route formal audit, simulator/training execution, and policy-quality evidence remain
separate and have not run.

The superseded analytic-axis v008/v007 pair, fixed-three-axis v007/v006 pair, query-conditioned
v006/v005 pair, and receipts bound to them remain history. They cannot establish correctness for the
active data-driven Contracts.

Superseded method/training contracts, including v023, FADA-ADAPT v002, v017, v016, v015, v014,
v013, v012, and v011, are retained under `history/` and excluded from default recall.

The 15-degree slope demonstration is an implementation of the active
FADA-ADAPT v003 contract. Its target artifact is target-only schema v3; legacy
actuator-gain schema v2 remains readable only through the explicit
`actuator_gain` compatibility boundary.
