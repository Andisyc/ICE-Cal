from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from unilab.algos.torch.distill.fada import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.calibration import (
    CalibrationRolloutBatch,
    CoefficientEncoder,
    DirectionBank,
    fit_scale_curve_bank,
)
from unilab.algos.torch.fada_context.calibration_training import (
    CalibrationScaleEvidence,
    SerialCalibrationConfig,
    calibration_compensation_ratio,
    coefficient_stage_loss,
    coefficient_validation_error,
    direction_stage_loss,
    fit_scale_stage,
    load_calibration_scale_evidence,
    load_calibration_training_checkpoint,
    run_serial_calibration_training,
    save_calibration_scale_evidence,
    save_calibration_training_checkpoint,
    validate_calibration_source_projection,
    validate_encoder_gradients,
)


def _config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=4,
        action_dim=3,
        command_dim=2,
        hidden_dim=8,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=16,
    )


def _batch(config: FADAArchitectureConfig, rows: int = 2) -> CalibrationRolloutBatch:
    return CalibrationRolloutBatch(
        observation_history=torch.randn(rows, 30, config.obs_dim),
        action_history=torch.randn(rows, 30, config.action_dim),
        command=torch.randn(rows, config.command_dim),
        nominal_action_chunk=torch.randn(rows, 6, config.action_dim),
        target_action_chunk=torch.randn(rows, 6, config.action_dim),
        c_true=torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.2, 0.0]])[:rows],
        axis_id=torch.arange(rows, dtype=torch.int64),
        is_held_out_combination=torch.zeros(rows, dtype=torch.bool),
        injected_strength=torch.ones(rows),
        planner_intent=torch.randn(rows, 6, config.obs_dim),
        rollout_id=torch.arange(rows, dtype=torch.int64),
        seed=torch.arange(rows, dtype=torch.int64) + 10,
        split_id=torch.zeros(rows, dtype=torch.int64),
    )


def _metadata() -> dict[str, str]:
    return {
        "source_tracker_sha256": "tracker",
        "dataset_sha256": "dataset",
        "split_sha256": "split",
        "axis_catalog_version": "gain-delay-offset-v1",
    }


def _scale_evidence() -> CalibrationScaleEvidence:
    readings = torch.linspace(-1.0, 1.0, 21).view(1, 21, 1).repeat(3, 1, 32)
    candidates = torch.linspace(-0.5, 0.5, 41)
    desired = 0.2 * readings
    errors = (candidates.view(1, 1, 1, -1) - desired.unsqueeze(-1)).square()
    return CalibrationScaleEvidence(
        readings=readings,
        candidate_scales=candidates,
        action_errors=errors,
        metadata=_metadata(),
    )


def test_stage1_direction_loss_only_exposes_selected_direction_gradient() -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    bank = DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=8)
    policy_snapshot = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    direction_snapshot = bank.directions.detach().clone()
    optimizer = torch.optim.Adam([bank.directions], lr=1e-3)
    loss = direction_stage_loss(policy, bank, _batch(config), axis_index=1)
    loss.backward()
    assert bank.directions.grad is not None
    assert torch.count_nonzero(bank.directions.grad[0]) == 0
    assert torch.count_nonzero(bank.directions.grad[2]) == 0
    assert torch.count_nonzero(bank.directions.grad[1]) > 0
    assert all(parameter.grad is None for parameter in policy.parameters())
    optimizer.step()
    torch.testing.assert_close(bank.directions[0], direction_snapshot[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(bank.directions[2], direction_snapshot[2], rtol=0.0, atol=0.0)
    assert not torch.equal(bank.directions[1], direction_snapshot[1])
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, policy_snapshot[name], rtol=0.0, atol=0.0)


def test_stage1_normalizes_only_the_admitted_axis() -> None:
    bank = DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=8)
    bank.directions.data[1].fill_(2.0)
    before = bank.directions.detach().clone()
    latent = torch.randn(2, 6, 8)
    coefficients = torch.tensor([[0.0, 0.4, 0.0], [0.0, -0.2, 0.0]])
    expected_composition = bank.compose(latent, coefficients)
    bank.normalize_axis_(1)
    torch.testing.assert_close(bank.directions[0], before[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(bank.directions[2], before[2], rtol=0.0, atol=0.0)
    torch.testing.assert_close(bank.directions[1].norm(), torch.tensor(1.0))
    torch.testing.assert_close(bank.normalization_scale[1], before[1].norm())
    torch.testing.assert_close(bank.compose(latent, coefficients), expected_composition)
    bank.normalize_axis_(1)
    torch.testing.assert_close(bank.normalization_scale[1], before[1].norm())
    torch.testing.assert_close(bank.compose(latent, coefficients), expected_composition)


def test_stage1_compensation_ratio_has_a_direct_hand_oracle() -> None:
    target = torch.tensor([[[1.0, -1.0]]])
    nominal = torch.tensor([[[0.0, 0.0]]])
    compensated = torch.tensor([[[0.8, -0.8]]])
    ratio = calibration_compensation_ratio(nominal, compensated, target)
    torch.testing.assert_close(ratio, torch.tensor(0.04))
    with pytest.raises(ValueError, match="uncompensated error"):
        calibration_compensation_ratio(target, compensated, target)


def test_stage2_coefficient_loss_only_exposes_encoder_gradient() -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    bank = DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=8)
    bank.directions.data.normal_()
    encoder = CoefficientEncoder(state_dim=4, action_dim=3, axis_count=3, hidden_dim=128, layers=2)
    frozen_snapshots = (
        {name: value.detach().clone() for name, value in policy.state_dict().items()},
        {name: value.detach().clone() for name, value in bank.state_dict().items()},
    )
    before_encoder = {name: value.detach().clone() for name, value in encoder.state_dict().items()}
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    loss = coefficient_stage_loss(policy, bank, encoder, _batch(config))
    loss.backward()
    assert any(parameter.grad is not None for parameter in encoder.parameters())
    assert all(parameter.grad is None for parameter in policy.parameters())
    assert bank.directions.grad is None
    optimizer.step()
    assert any(
        not torch.equal(value, before_encoder[name]) for name, value in encoder.state_dict().items()
    )
    for owner, snapshot in zip((policy, bank), frozen_snapshots, strict=True):
        for name, value in owner.state_dict().items():
            torch.testing.assert_close(value, snapshot[name], rtol=0.0, atol=0.0)


def test_stage2_admission_uses_worst_normalized_coefficient_error() -> None:
    predicted = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.2, 0.0]])
    target = torch.zeros_like(predicted)
    torch.testing.assert_close(coefficient_validation_error(predicted, target), torch.tensor(0.2))


def test_stage2_rejects_zero_or_nonfinite_encoder_gradients() -> None:
    encoder = CoefficientEncoder(state_dim=4, action_dim=3, axis_count=3)
    for parameter in encoder.parameters():
        parameter.grad = torch.zeros_like(parameter)
    with pytest.raises(ValueError, match="finite nonzero"):
        validate_encoder_gradients(encoder)
    first_parameter = next(encoder.parameters())
    first_parameter.grad.fill_(torch.nan)
    with pytest.raises(ValueError, match="finite nonzero"):
        validate_encoder_gradients(encoder)
    first_parameter.grad.fill_(1.0)
    validate_encoder_gradients(encoder)


def test_stage3_scale_fit_does_not_construct_or_update_neural_owner() -> None:
    readings = torch.linspace(-1.0, 1.0, 21).view(1, 21, 1).repeat(3, 1, 32)
    candidates = torch.linspace(-0.4, 0.4, 81)
    desired = 0.2 * readings
    action_errors = (candidates.view(1, 1, 1, -1) - desired.unsqueeze(-1)).square()
    artifact = fit_scale_stage(readings, candidates, action_errors)
    assert len(artifact) == 3
    assert all(curve.kind == "pchip" for curve in artifact)


def test_stage3_rejects_missing_32_repeat_evidence() -> None:
    readings = torch.linspace(-1.0, 1.0, 21).repeat(3, 1)
    with pytest.raises(ValueError, match="32 repetitions"):
        fit_scale_stage(readings, torch.linspace(-1.0, 1.0, 3), torch.zeros(3, 21, 3))


def test_stage3_rejects_low_quality_monotone_fit() -> None:
    readings = torch.linspace(-1.0, 1.0, 21).view(1, 21, 1).repeat(3, 1, 32)
    candidates = torch.tensor([-1.0, 0.0, 1.0])
    desired = readings.sign()
    desired[:, :, ::2] *= -1
    action_errors = (candidates.view(1, 1, 1, -1) - desired.unsqueeze(-1)).square()
    with pytest.raises(ValueError, match=r"R\^2"):
        fit_scale_stage(readings, candidates, action_errors)


def test_stage3_evidence_round_trip_binds_transaction_identity(tmp_path) -> None:
    evidence = _scale_evidence()
    path = save_calibration_scale_evidence(tmp_path / "scale-evidence.pt", evidence)
    restored = load_calibration_scale_evidence(path, expected_metadata=_metadata())
    torch.testing.assert_close(restored.readings, evidence.readings)
    torch.testing.assert_close(restored.action_errors, evidence.action_errors)
    with pytest.raises(ValueError, match="metadata identity"):
        load_calibration_scale_evidence(
            path,
            expected_metadata={**_metadata(), "dataset_sha256": "other-dataset"},
        )


def test_serial_checkpoint_rejects_out_of_order_stage_and_round_trips(tmp_path) -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config)
    bank = DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=8)
    bank.directions.data.normal_()
    bank.normalize_()
    encoder = CoefficientEncoder(state_dim=4, action_dim=3, axis_count=3, hidden_dim=128, layers=2)
    path = save_calibration_training_checkpoint(
        tmp_path / "stage2.pt",
        policy=policy,
        direction_bank=bank,
        coefficient_encoder=encoder,
        stage="coefficient_frozen",
        metadata=_metadata(),
    )
    restored_policy = FADAPlannerIDMPolicy(config)
    restored_bank = DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=8)
    restored_encoder = CoefficientEncoder(
        state_dim=4,
        action_dim=3,
        axis_count=3,
        hidden_dim=128,
        layers=2,
    )
    restored = load_calibration_training_checkpoint(
        path,
        restored_policy,
        restored_bank,
        restored_encoder,
        expected_metadata=_metadata(),
        expected_stage="coefficient_frozen",
    )
    assert restored["stage"] == "coefficient_frozen"
    for source, target in (
        (policy.planner, restored_policy.planner),
        (policy.idm, restored_policy.idm),
        (bank, restored_bank),
        (encoder, restored_encoder),
    ):
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[name], rtol=0.0, atol=0.0)
    direction_path = save_calibration_training_checkpoint(
        tmp_path / "stage1.pt",
        policy=policy,
        direction_bank=bank,
        coefficient_encoder=encoder,
        stage="direction_frozen",
        metadata=_metadata(),
    )
    direction_payload = load_calibration_training_checkpoint(
        direction_path,
        restored_policy,
        restored_bank,
        restored_encoder,
        expected_metadata=_metadata(),
        expected_stage="direction_frozen",
    )
    assert direction_payload["stage"] == "direction_frozen"
    with pytest.raises(ValueError, match="stage mismatch"):
        load_calibration_training_checkpoint(
            path,
            restored_policy,
            restored_bank,
            restored_encoder,
            expected_metadata=_metadata(),
            expected_stage="direction_frozen",
        )
    bad = tmp_path / "bad.pt"
    torch.save({"schema_version": "unilab_fada_calibration_checkpoint_v0"}, bad)
    with pytest.raises(ValueError, match="unsupported calibration checkpoint schema"):
        load_calibration_training_checkpoint(
            bad,
            policy,
            bank,
            encoder,
            expected_metadata=_metadata(),
            expected_stage="coefficient_frozen",
        )


def test_checkpoint_load_rolls_back_every_borrowed_owner_on_late_failure(tmp_path) -> None:
    config = _config()
    source_policy = FADAPlannerIDMPolicy(config)
    source_bank = DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=8)
    source_bank.directions.data.normal_()
    source_bank.normalize_()
    source_encoder = CoefficientEncoder(
        state_dim=4,
        action_dim=3,
        axis_count=3,
        hidden_dim=128,
        layers=2,
    )
    path = save_calibration_training_checkpoint(
        tmp_path / "checkpoint.pt",
        policy=source_policy,
        direction_bank=source_bank,
        coefficient_encoder=source_encoder,
        stage="coefficient_frozen",
        metadata=_metadata(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    first_planner_key = next(iter(payload["planner_state_dict"]))
    payload["planner_state_dict"][first_planner_key] = torch.full_like(
        payload["planner_state_dict"][first_planner_key],
        7.0,
    )
    payload["coefficient_encoder_state_dict"] = {"wrong": torch.tensor(1.0)}
    torch.save(payload, path)

    target_policy = FADAPlannerIDMPolicy(config)
    target_bank = DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=8)
    target_encoder = CoefficientEncoder(
        state_dim=4,
        action_dim=3,
        axis_count=3,
        hidden_dim=128,
        layers=2,
    )
    owners = (target_policy.planner, target_policy.idm, target_bank, target_encoder)
    snapshots = tuple(
        {name: value.detach().clone() for name, value in owner.state_dict().items()}
        for owner in owners
    )
    with pytest.raises(RuntimeError, match="state_dict"):
        load_calibration_training_checkpoint(
            path,
            target_policy,
            target_bank,
            target_encoder,
            expected_metadata=_metadata(),
            expected_stage="coefficient_frozen",
        )
    for owner, snapshot in zip(owners, snapshots, strict=True):
        for name, value in owner.state_dict().items():
            torch.testing.assert_close(value, snapshot[name], rtol=0.0, atol=0.0)


def test_serial_training_rejects_missing_validation_before_optimizer(
    tmp_path,
    monkeypatch,
) -> None:
    calls = 0

    def poison(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("optimizer construction must be unreachable")

    monkeypatch.setattr(torch.optim, "Adam", poison)
    with pytest.raises(ValueError, match="train and validation"):
        run_serial_calibration_training(
            FADAPlannerIDMPolicy(_config()),
            _batch(_config()),
            output_dir=tmp_path,
            source_tracker_sha256="tracker",
            dataset_sha256="dataset",
            split_sha256="split",
            axis_catalog_version="gain-delay-offset-v1",
            scale_evidence=_scale_evidence(),
            config=SerialCalibrationConfig(stage1_steps_per_axis=1, stage2_steps=1),
        )
    assert calls == 0


def test_serial_training_rejects_wrong_scale_evidence_before_optimizer(
    tmp_path,
    monkeypatch,
) -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config).eval()
    batch = replace(_batch(config), split_id=torch.tensor([0, 1], dtype=torch.int64))
    with torch.no_grad():
        intent = policy.planner(batch.observation_history, batch.command)
        latent = policy.idm.encode_latent(
            batch.observation_history,
            batch.action_history,
            intent,
        )
        nominal = policy.idm.decode_latent(latent)
    batch = replace(batch, planner_intent=intent, nominal_action_chunk=nominal)
    calls = 0

    def poison(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("optimizer construction must be unreachable")

    monkeypatch.setattr(torch.optim, "Adam", poison)
    wrong_evidence = replace(
        _scale_evidence(),
        metadata={**_metadata(), "dataset_sha256": "other-dataset"},
    )
    with pytest.raises(ValueError, match="scale evidence metadata identity"):
        run_serial_calibration_training(
            policy,
            batch,
            output_dir=tmp_path,
            source_tracker_sha256="tracker",
            dataset_sha256="dataset",
            split_sha256="split",
            axis_catalog_version="gain-delay-offset-v1",
            scale_evidence=wrong_evidence,
            config=SerialCalibrationConfig(stage1_steps_per_axis=1, stage2_steps=1),
        )
    assert calls == 0


def test_source_projection_rejects_dataset_from_a_different_tracker() -> None:
    config = _config()
    policy = FADAPlannerIDMPolicy(config).eval()
    batch = _batch(config)
    with torch.no_grad():
        intent = policy.planner(batch.observation_history, batch.command)
        latent = policy.idm.encode_latent(
            batch.observation_history,
            batch.action_history,
            intent,
        )
        nominal = policy.idm.decode_latent(latent)
    valid = replace(batch, planner_intent=intent, nominal_action_chunk=nominal)
    validate_calibration_source_projection(policy, valid)
    with pytest.raises(ValueError, match="nominal Action"):
        validate_calibration_source_projection(
            policy,
            replace(valid, nominal_action_chunk=nominal + 0.1),
        )


def test_serial_training_completes_all_three_real_owner_stages(tmp_path) -> None:
    torch.manual_seed(13)
    config = _config()
    policy = FADAPlannerIDMPolicy(config).eval()
    base_observation = torch.randn(3, 30, config.obs_dim)
    base_actions = torch.randn(3, 30, config.action_dim)
    base_command = torch.randn(3, config.command_dim)
    with torch.no_grad():
        intent = policy.planner(base_observation, base_command)
        latent = policy.idm.encode_latent(base_observation, base_actions, intent)
        nominal = policy.idm.decode_latent(latent)
        true_directions = 0.2 * torch.randn(3, 6, config.hidden_dim)
        coefficients = torch.diag(torch.tensor([0.2, 0.3, 0.4]))
        target = policy.idm.decode_latent(
            latent + torch.einsum("bm,mkd->bkd", coefficients, true_directions)
        )
    batch = CalibrationRolloutBatch(
        observation_history=base_observation.repeat(2, 1, 1),
        action_history=base_actions.repeat(2, 1, 1),
        command=base_command.repeat(2, 1),
        nominal_action_chunk=nominal.repeat(2, 1, 1),
        target_action_chunk=target.repeat(2, 1, 1),
        c_true=coefficients.repeat(2, 1),
        axis_id=torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64),
        is_held_out_combination=torch.zeros(6, dtype=torch.bool),
        injected_strength=torch.ones(6),
        planner_intent=intent.repeat(2, 1, 1),
        rollout_id=torch.arange(6, dtype=torch.int64),
        seed=torch.arange(6, dtype=torch.int64) + 100,
        split_id=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64),
    )
    result = run_serial_calibration_training(
        policy,
        batch,
        output_dir=tmp_path,
        source_tracker_sha256="tracker",
        dataset_sha256="dataset",
        split_sha256="split",
        axis_catalog_version="gain-delay-offset-v1",
        scale_evidence=_scale_evidence(),
        config=SerialCalibrationConfig(
            stage1_steps_per_axis=200,
            stage2_steps=1200,
            learning_rate=3e-3,
        ),
    )
    assert result["stage"] == "complete"
    assert float(result["coefficient_error"]) <= 0.05
    assert (tmp_path / "stage1_direction_frozen.pt").is_file()
    assert (tmp_path / "stage2_coefficient_frozen.pt").is_file()
    assert (tmp_path / "calibration_artifact.pt").is_file()
    assert (tmp_path / "stage3_complete.pt").is_file()
    complete = torch.load(tmp_path / "stage3_complete.pt", map_location="cpu", weights_only=True)
    assert len(complete["scale_curves"]) == 3
    restored_policy = FADAPlannerIDMPolicy(config)
    restored_bank = DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=8)
    restored_encoder = CoefficientEncoder(
        state_dim=4,
        action_dim=3,
        axis_count=3,
    )
    loaded = load_calibration_training_checkpoint(
        tmp_path / "stage3_complete.pt",
        restored_policy,
        restored_bank,
        restored_encoder,
        expected_metadata=_metadata(),
        expected_stage="complete",
    )
    assert len(loaded["scale_curves"]) == 3
