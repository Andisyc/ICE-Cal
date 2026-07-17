# E82 — HP-6b10 Motrix Provider/Test Selection

Result: **PASS**

The Stewart runtime helper now uses `pytest.importorskip("motrixsim")`. Static
backend-contract and Hydra-owner tests remain active; only tests that construct
the real optional Motrix backend skip when the provider is absent.

Evidence: Stewart module `4 passed, 2 skipped, 1 deselected`; targeted Ruff
passes. This does not prove Stewart runtime behavior with Motrix installed.
