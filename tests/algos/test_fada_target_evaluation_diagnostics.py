from __future__ import annotations

import torch

from unilab.algos.torch.distill.fada.model import FADAArchitectureConfig
from unilab.algos.torch.distill.fada.target_data import FADATargetBatch
from unilab.algos.torch.distill.fada.target_evaluation_diagnostics import (
    compare_fada_rollout_diagnostics,
    summarize_fada_own_rollout,
)


class _Planner(torch.nn.Module):
    def forward(self, history: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        del command
        return history[:, -1:].repeat(1, 2, 1) + 1.0


class _IDM(torch.nn.Module):
    def forward(
        self,
        history: torch.Tensor,
        action_history: torch.Tensor,
        future: torch.Tensor,
    ) -> torch.Tensor:
        del history, action_history
        return future[..., :2]


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = FADAArchitectureConfig(
            obs_dim=3,
            action_dim=2,
            command_dim=3,
            history_length=2,
            prediction_horizon=2,
            hidden_dim=8,
            num_heads=2,
            planner_layers=1,
            idm_encoder_layers=1,
            idm_decoder_layers=1,
            feedforward_dim=16,
        )
        self.planner = _Planner()
        self.idm = _IDM()


def _batch(*, realized_offset: float, executed_offset: float) -> FADATargetBatch:
    history = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]], [[1.0, 1.0, 1.0], [4.0, 5.0, 6.0]]])
    predicted = history[:, -1:].repeat(1, 2, 1) + 1.0
    realized = predicted + realized_offset
    executed = realized[..., :2] + executed_offset
    return FADATargetBatch(
        observation_history=history,
        action_history=torch.zeros(2, 2, 2),
        command=torch.zeros(2, 3),
        realized_future=realized,
        executed_action_chunk=executed,
        episode_id=torch.tensor([0, 1]),
        start_timestep=torch.tensor([1, 1]),
    )


def test_own_rollout_diagnostics_separate_idm_fit_from_consistency_gap() -> None:
    result = summarize_fada_own_rollout(_Policy(), _batch(realized_offset=2.0, executed_offset=0.5))

    assert result == {
        "num_windows": 2,
        "own_rollout_idm_loss": 0.25,
        "planner_realized_action_gap_rmse": 2.0,
        "planner_executed_action_replay_rmse": 2.5,
    }


def test_consistency_gap_is_sensitive_to_realized_future() -> None:
    aligned = summarize_fada_own_rollout(
        _Policy(), _batch(realized_offset=0.0, executed_offset=0.0)
    )
    shifted = summarize_fada_own_rollout(
        _Policy(), _batch(realized_offset=3.0, executed_offset=0.0)
    )

    assert aligned["planner_realized_action_gap_rmse"] == 0.0
    assert shifted["planner_realized_action_gap_rmse"] == 3.0


def test_diagnostic_comparison_uses_positive_for_adaptation_improvement() -> None:
    zero = summarize_fada_own_rollout(_Policy(), _batch(realized_offset=2.0, executed_offset=0.5))
    adapted = summarize_fada_own_rollout(
        _Policy(), _batch(realized_offset=1.0, executed_offset=0.25)
    )

    result = compare_fada_rollout_diagnostics(zero, adapted)

    assert result["own_rollout_idm_loss"] > 0.0
    assert result["planner_realized_action_gap_rmse"] == 1.0
