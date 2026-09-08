"""The selected weights and restored environment must come from the same run."""

import json
from types import SimpleNamespace as NS

import pytest

from unilab.visualization import playback_checkpoint_contract as contract
from unilab.visualization.playback_policy_sessions import (
    _resolve_task_checkpoint_from_playback_cfg,
)


def _args(checkpoint_path):
    return NS(
        task="g1_walk_flat", algo="sac", checkpoint_path=checkpoint_path,
        load_run="-1", algo_log_name="fast_sac", checkpoint=None, log_root=None,
    )


@pytest.mark.parametrize("absolute", [True, False])
def test_explicit_checkpoint_restores_its_own_contract(tmp_path, monkeypatch, absolute):
    selected = tmp_path / "model" / "selected"
    selected.mkdir(parents=True)
    checkpoint = selected / "model_5000.pt"
    checkpoint.touch()
    saved = {"config": {
        "env": {"gait_clock_mode": "continuous", "mode_observation": False},
        "reward": {"tracking_sigma": .37},
    }}
    (selected / "run_config.json").write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(contract, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(
        contract, "resolve_task_checkpoint_path",
        lambda *a, **kw: pytest.fail("Explicit paths must not search other runs"),
    )
    monkeypatch.chdir(selected)  # Relative paths are repository-relative, not cwd-relative.
    args = _args(str(checkpoint if absolute else checkpoint.relative_to(tmp_path)))
    assert contract._resolve_play_checkpoint_path(args) == (
        _resolve_task_checkpoint_from_playback_cfg(args, NS(), tmp_path)
    ) == checkpoint
    assert contract._load_checkpoint_run_config(args) == saved
    restored = contract.apply_checkpoint_env_contract(
        {"gait_clock_mode": "command_fixed", "reward_config": {"tracking_sigma": .25}},
        args,
    )
    assert restored["gait_clock_mode"] == "continuous"
    assert restored["reward_config"] == {"tracking_sigma": .37}


def test_missing_explicit_checkpoint_fails_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "ROOT_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="training.play_checkpoint_path"):
        contract._load_checkpoint_run_config(_args("missing.pt"))


def test_missing_sidecar_does_not_search_another_run(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model_5000.pt"
    checkpoint.touch()
    monkeypatch.setattr(
        contract, "resolve_task_checkpoint_path",
        lambda *a, **kw: pytest.fail("A missing sidecar must not select another run"),
    )
    assert contract._load_checkpoint_run_config(_args(str(checkpoint))) is None


def test_unspecified_checkpoint_preserves_log_lookup(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model_5000.pt"
    calls = []

    def resolve(root, **kwargs):
        calls.append(kwargs)
        return checkpoint, tmp_path

    monkeypatch.setattr(contract, "resolve_task_checkpoint_path", resolve)
    assert contract._resolve_play_checkpoint_path(_args(None)) == checkpoint
    assert calls[0]["load_run"] == "-1"
