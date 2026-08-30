from __future__ import annotations


def test_checkpoint_contract_helpers_have_one_visualization_owner() -> None:
    from scripts.play_interactive import _apply_checkpoint_env_contract as legacy

    from unilab.visualization.playback_checkpoint_contract import (
        apply_checkpoint_env_contract,
    )

    assert legacy is apply_checkpoint_env_contract


def test_interactive_session_factory_has_one_visualization_owner() -> None:
    from scripts.play_interactive import _create_interactive_session as legacy

    from unilab.visualization.playback_viewer import create_interactive_session

    assert legacy is create_interactive_session
