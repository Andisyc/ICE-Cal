from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_materializer():
    path = Path("scripts/deploy/materialize_hp7c3_gate0.py")
    spec = importlib.util.spec_from_file_location("materialize_hp7c3_gate0", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_oracle_reads_materializer_identity_schema() -> None:
    module = _load_materializer()
    source = module.oracle_source()

    compile(source, "hp7c3_bounded_persistent_oracle_v4.py", "exec")
    assert 'freeze["compose"]["observed_sha256"]' in source
    assert 'freeze["compose"]["sha256"]' not in source
    assert 'freeze["hard_artifacts"]' in source
