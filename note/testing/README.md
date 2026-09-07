# ICE-Cal test-design registry

There is currently no Module Test receipt for the active Source v024 or Context v009 Contracts.
Design activation and documentation checks do not establish module correctness.

## Historical Context evidence

The files in this directory bind earlier Context implementations:

- `module_test_cards.md`, `module_test_manifest.json`,
  `calibratable_tracker_module_test_evidence.json`, and
  `calibratable_tracker_module_test_control_board.json` cover
  `FADA-CONTEXT-METHOD-v008 + FADA-CONTEXT-TRAIN-v007`.
- `module_test_control_board.json`, `module_test_evidence.json`, and the formal-audit JSON files
  cover the still older v006/v005 route.

Those receipts remain immutable historical evidence. Their internal `MODULE-CORRECT` or formal
statuses do not transfer to the active data-driven Context v009 design or to the v024 Oracle.

## Next test-design owner

The next v024 engineering unit must create new Module Test Cards covering command dead-zone order,
exact null-command identity, phase freeze/advance, contact truth tables, bounded Reward, observation
visibility, checkpoint behavior profile and target-route isolation. Until those cards and tests are
confirmed, module correctness is unclaimed.
