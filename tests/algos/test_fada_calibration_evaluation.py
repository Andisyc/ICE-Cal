from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from unilab.algos.torch.distill.fada import FADAArchitectureConfig, PlannerIDMOutput
from unilab.algos.torch.fada_context.calibration import CalibrationRolloutBatch
from unilab.algos.torch.fada_context.calibration_evaluation import (
    CalibrationFullFinetuneUpperBound,
    evaluate_held_out_calibration,
    load_calibration_full_finetune_upper_bound,
    save_calibration_full_finetune_upper_bound,
)


def _config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=4,
        action_dim=2,
        command_dim=2,
        hidden_dim=8,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=16,
    )


def _batch() -> CalibrationRolloutBatch:
    config = _config()
    rows = 3
    target = torch.zeros(rows, 6, 2)
    target[1, :, 0] = 2.0
    target[2, :, 0] = 4.0
    return CalibrationRolloutBatch(
        observation_history=torch.zeros(rows, 30, 4),
        action_history=torch.zeros(rows, 30, 2),
        command=torch.zeros(rows, 2),
        nominal_action_chunk=torch.zeros(rows, 6, 2),
        target_action_chunk=target,
        c_true=torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.4, 0.0], [0.7, 0.0, -0.3]]),
        axis_id=torch.tensor([0, -1, -1], dtype=torch.int64),
        is_held_out_combination=torch.tensor([False, True, True]),
        injected_strength=torch.ones(rows),
        planner_intent=torch.zeros(rows, 6, 4),
        rollout_id=torch.arange(rows, dtype=torch.int64),
        seed=torch.arange(rows, dtype=torch.int64),
        split_id=torch.tensor([0, 2, 2], dtype=torch.int64),
    ).validate(config, axis_count=3)


class _NominalPolicy:
    config = _config()

    def __call__(self, observation_history, action_history, command):
        chunk = torch.zeros(observation_history.shape[0], 6, 2)
        return PlannerIDMOutput(torch.zeros(observation_history.shape[0], 6, 4), chunk, chunk[:, 0])


class _CalibratedPolicy(_NominalPolicy):
    def __init__(self) -> None:
        self.reconstructed_coefficients: torch.Tensor | None = None

    def __call__(self, observation_history, action_history, command):
        chunk = torch.zeros(observation_history.shape[0], 6, 2)
        chunk[0, :, 0] = 1.5
        chunk[1, :, 0] = 3.5
        return PlannerIDMOutput(torch.zeros(observation_history.shape[0], 6, 4), chunk, chunk[:, 0])

    def reconstruct_with_coefficients(
        self,
        observation_history,
        action_history,
        command,
        coefficients,
    ):
        self.reconstructed_coefficients = coefficients.detach().clone()
        chunk = coefficients[:, :1, None].expand(-1, 6, 2).clone()
        return PlannerIDMOutput(torch.zeros(observation_history.shape[0], 6, 4), chunk, chunk[:, 0])


def test_held_out_evaluation_routes_only_combination_rows_and_uses_first_action() -> None:
    calibrated = _CalibratedPolicy()
    full_finetune = torch.zeros(3, 6, 2)
    full_finetune[1:, :, 0] = torch.tensor([2.0, 4.0])[:, None]
    report = evaluate_held_out_calibration(
        _NominalPolicy(),
        calibrated,
        _batch(),
        full_finetune=CalibrationFullFinetuneUpperBound(
            action_chunk=full_finetune,
            rollout_id=_batch().rollout_id,
        ),
    )
    assert report["held_out_rows"] == 2
    assert report["executed_action_index"] == 0
    assert report["first_action_mse"]["nominal"] == pytest.approx(5.0)
    assert report["first_action_mse"]["calibrated"] == pytest.approx(0.125)
    assert report["first_action_mse"]["full_finetune_upper_bound"] == 0.0
    assert calibrated.reconstructed_coefficients is not None
    torch.testing.assert_close(
        calibrated.reconstructed_coefficients,
        _batch().c_true[torch.tensor([2, 1])],
    )


def test_held_out_evaluation_rejects_training_only_or_wrong_upper_bound_shape() -> None:
    batch = _batch()
    with pytest.raises(ValueError, match="held-out combination"):
        evaluate_held_out_calibration(
            _NominalPolicy(),
            _CalibratedPolicy(),
            replace(
                batch,
                c_true=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, -0.3]]),
                axis_id=torch.tensor([0, 1, 2], dtype=torch.int64),
                is_held_out_combination=torch.zeros(3, dtype=torch.bool),
            ),
            full_finetune=CalibrationFullFinetuneUpperBound(
                action_chunk=torch.zeros(3, 6, 2),
                rollout_id=batch.rollout_id,
            ),
        )
    with pytest.raises(ValueError, match="full-finetune"):
        evaluate_held_out_calibration(
            _NominalPolicy(),
            _CalibratedPolicy(),
            batch,
            full_finetune=CalibrationFullFinetuneUpperBound(
                action_chunk=torch.zeros(2, 6, 2),
                rollout_id=torch.arange(2, dtype=torch.int64),
            ),
        )


def test_one_axis_evaluation_is_explicitly_not_applicable() -> None:
    source = _batch().index_select(torch.tensor([0]))
    batch = replace(source, c_true=source.c_true[:, :1]).validate(_config(), axis_count=1)
    with pytest.raises(ValueError, match="not applicable"):
        evaluate_held_out_calibration(
            _NominalPolicy(),
            _CalibratedPolicy(),
            batch,
            full_finetune=CalibrationFullFinetuneUpperBound(
                action_chunk=batch.target_action_chunk,
                rollout_id=batch.rollout_id,
            ),
        )


def test_full_finetune_upper_bound_round_trip_binds_row_identity(tmp_path) -> None:
    upper = CalibrationFullFinetuneUpperBound(
        action_chunk=torch.zeros(3, 6, 2),
        rollout_id=torch.tensor([10, 11, 12], dtype=torch.int64),
    )
    metadata = {
        "source_tracker_sha256": "source",
        "dataset_sha256": "dataset",
        "split_sha256": "split",
    }
    path = save_calibration_full_finetune_upper_bound(
        tmp_path / "upper.pt",
        upper,
        metadata=metadata,
    )
    restored = load_calibration_full_finetune_upper_bound(
        path,
        expected_metadata=metadata,
    )
    torch.testing.assert_close(restored.action_chunk, upper.action_chunk)
    torch.testing.assert_close(restored.rollout_id, upper.rollout_id)
