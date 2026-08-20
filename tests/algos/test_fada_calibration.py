from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from unilab.algos.torch.distill.fada import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.calibration import (
    CALIBRATION_ARTIFACT_SCHEMA,
    CALIBRATION_AXIS_CATALOG_VERSION,
    CALIBRATION_METHOD_CONTRACT_ID,
    CalibratedFADAPolicy,
    CalibrationReadoutState,
    CalibrationRolloutBatch,
    CoefficientEncoder,
    DirectionBank,
    FaultAxisCatalog,
    MonotoneScaleCurve,
    fit_scale_curve_bank,
    load_calibration_artifact,
    save_calibration_artifact,
)
from unilab.algos.torch.fada_context.calibration_data import (
    calibration_split_identity_sha256,
    load_fault_axis_catalog,
    prepare_calibration_rollout_batch,
)
from unilab.algos.torch.fada_context.calibration_runtime import (
    CalibratedFADAPlaybackController,
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


def _batch(config: FADAArchitectureConfig, rows: int = 3) -> CalibrationRolloutBatch:
    return CalibrationRolloutBatch(
        observation_history=torch.randn(rows, config.history_length, config.obs_dim),
        action_history=torch.randn(rows, config.history_length, config.action_dim),
        command=torch.randn(rows, config.command_dim),
        nominal_action_chunk=torch.randn(rows, config.prediction_horizon, config.action_dim),
        target_action_chunk=torch.randn(rows, config.prediction_horizon, config.action_dim),
        c_true=torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.7]]),
        axis_id=torch.tensor([0, 1, 2], dtype=torch.int64),
        is_held_out_combination=torch.zeros(rows, dtype=torch.bool),
        injected_strength=torch.tensor([1.2, 1.0, 0.2]),
        planner_intent=torch.randn(rows, config.prediction_horizon, config.obs_dim),
        rollout_id=torch.tensor([10, 11, 12], dtype=torch.int64),
        seed=torch.tensor([101, 102, 103], dtype=torch.int64),
        split_id=torch.tensor([0, 0, 1], dtype=torch.int64),
    )


def _normalized_bank(*, latent_dim: int = 128) -> DirectionBank:
    bank = DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=latent_dim)
    bank.directions.data.normal_()
    return bank.normalize_()


def test_axis_catalog_has_analytic_gain_delay_offset_targets() -> None:
    catalog = FaultAxisCatalog.default()
    nominal = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    assert catalog.names == ("gain", "delay", "offset")
    torch.testing.assert_close(catalog.analytic_target("gain", nominal, 2.0), nominal / 2.0)
    torch.testing.assert_close(
        catalog.analytic_target("delay", nominal, 1.0),
        torch.tensor([[[4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [7.0, 8.0, 9.0]]]),
    )
    torch.testing.assert_close(catalog.analytic_target("offset", nominal, 0.5), nominal - 0.5)
    with pytest.raises(ValueError, match="gain strength"):
        catalog.analytic_target("gain", nominal, 0.0)
    with pytest.raises(ValueError, match="prediction horizon"):
        catalog.analytic_target("delay", nominal, 3.0)
    with pytest.raises(ValueError, match="unregistered"):
        catalog.analytic_target("friction", nominal, 1.0)


def test_axis_catalog_is_loaded_from_the_active_config() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "conf/fada_context/calibration_axes/gain_delay_offset_v1.yaml"
    )
    catalog = load_fault_axis_catalog(path)
    assert catalog.version == CALIBRATION_AXIS_CATALOG_VERSION
    assert catalog.names == ("gain", "delay", "offset")


def test_axis_catalog_rejects_same_axes_in_the_wrong_declared_order(tmp_path: Path) -> None:
    path = tmp_path / "axes.yaml"
    path.write_text(
        """catalog_version: gain-delay-offset-v1
axis_order: [delay, gain, offset]
axes:
  - {name: gain, normalized_range: [-1, 1], units: x, injection: x}
  - {name: delay, normalized_range: [-1, 1], units: x, injection: x}
  - {name: offset, normalized_range: [-1, 1], units: x, injection: x}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="order"):
        load_fault_axis_catalog(path)


def test_raw_single_axis_rollouts_get_analytic_targets_and_split_identity() -> None:
    config = _config()
    source = _batch(config)
    raw = {
        name: value
        for name, value in source.__dict__.items()
        if name not in {"target_action_chunk", "axis_id"}
    }
    raw["axis_name"] = ["gain", "delay", "offset"]
    raw["injected_strength"] = torch.tensor([1.2, 1.0, 0.2])
    catalog = FaultAxisCatalog.default()
    sealed = prepare_calibration_rollout_batch(raw, config, catalog)
    torch.testing.assert_close(
        sealed.target_action_chunk[0],
        source.nominal_action_chunk[0] / 1.2,
    )
    assert sealed.axis_id.tolist() == [0, 1, 2]
    split_identity = calibration_split_identity_sha256(sealed)
    assert len(split_identity) == 64
    assert (
        calibration_split_identity_sha256(
            replace(sealed, split_id=torch.tensor([1, 0, 1], dtype=torch.int64))
        )
        != split_identity
    )
    missing = dict(raw)
    missing.pop("command")
    with pytest.raises(ValueError, match="missing tensor fields"):
        prepare_calibration_rollout_batch(missing, config, catalog)


def test_direction_bank_is_six_token_field_and_zero_is_nominal() -> None:
    bank = DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=128)
    assert tuple(bank.directions.shape) == (3, 6, 128)
    latent = torch.randn(2, 6, 128)
    zero = torch.zeros(2, 3)
    torch.testing.assert_close(bank.compose(latent, zero), latent, rtol=0.0, atol=0.0)
    bank.directions = nn.Parameter(torch.zeros(3, 128))
    with pytest.raises(ValueError, match=r"\[axis, horizon, latent\]"):
        bank.compose(latent, zero)


def test_coefficient_encoder_requires_30_frame_histories_and_returns_raw_axis_readings() -> None:
    encoder = CoefficientEncoder(state_dim=4, action_dim=3, axis_count=3, hidden_dim=128, layers=2)
    state = torch.randn(2, 30, 4)
    action = torch.randn(2, 30, 3)
    output = encoder(state, action)
    assert output.shape == (2, 3)
    with pytest.raises(ValueError, match="history length"):
        encoder(state[:, :-1], action[:, :-1])


def test_coefficient_encoder_uses_both_histories_is_row_covariant_and_does_not_clamp() -> None:
    torch.manual_seed(7)
    encoder = CoefficientEncoder(state_dim=4, action_dim=3, axis_count=3)
    state = torch.randn(2, 30, 4)
    action = torch.randn(2, 30, 3)
    baseline = encoder(state, action)
    assert not torch.equal(encoder(state + 0.5, action), baseline)
    assert not torch.equal(encoder(state, action - 0.5), baseline)
    permutation = torch.tensor([1, 0])
    torch.testing.assert_close(
        encoder(state[permutation], action[permutation]),
        baseline[permutation],
    )
    encoder.readout.weight.data.zero_()
    encoder.readout.bias.data.fill_(3.0)
    torch.testing.assert_close(encoder(state, action), torch.full((2, 3), 3.0))


def test_scale_curve_is_monotone_bounded_and_reports_range() -> None:
    curve = MonotoneScaleCurve.fit(torch.linspace(-1.0, 1.0, 21), torch.linspace(-0.4, 0.4, 21))
    values, events = curve.map(torch.tensor([-2.0, -0.5, 0.5, 2.0]))
    assert curve.kind == "pchip"
    assert torch.all(values[1:] >= values[:-1])
    assert events.tolist() == [True, False, False, True]
    assert float(values.min()) >= -0.40001 and float(values.max()) <= 0.40001
    with pytest.raises(ValueError, match="monotone"):
        MonotoneScaleCurve.fit(
            torch.arange(4, dtype=torch.float32),
            torch.tensor([0.0, 1.0, 1.0, 0.0]),
        )


def test_scale_curve_bank_is_axis_permutation_covariant() -> None:
    readings = torch.stack(
        (
            torch.linspace(-1.0, 1.0, 21),
            torch.linspace(-0.8, 0.8, 21),
            torch.linspace(-0.6, 0.6, 21),
        )
    )
    scales = torch.stack(
        (
            torch.linspace(-0.3, 0.3, 21),
            torch.linspace(-0.2, 0.2, 21),
            torch.linspace(-0.1, 0.1, 21),
        )
    )
    baseline = fit_scale_curve_bank(readings, scales)
    permutation = torch.tensor([2, 0, 1])
    permuted = fit_scale_curve_bank(readings[permutation], scales[permutation])
    for output_index, source_index in enumerate(permutation.tolist()):
        torch.testing.assert_close(permuted[output_index].x, baseline[source_index].x)
        torch.testing.assert_close(permuted[output_index].y, baseline[source_index].y)


def test_stage_three_artifact_round_trip_binds_contract_and_is_not_a_neural_checkpoint(
    tmp_path: Path,
) -> None:
    bank = fit_scale_curve_bank(
        torch.linspace(-1.0, 1.0, 21).repeat(3, 1),
        torch.linspace(-0.4, 0.4, 21).repeat(3, 1),
    )
    config = FADAArchitectureConfig(obs_dim=4, action_dim=3, command_dim=2)
    path = save_calibration_artifact(
        tmp_path / "calibration.pt",
        config=config,
        direction_bank=_normalized_bank(),
        coefficient_encoder=CoefficientEncoder(state_dim=4, action_dim=3, axis_count=3),
        scale_curves=bank,
        metadata={
            "source_tracker_sha256": "abc",
            "dataset_sha256": "def",
            "split_sha256": "ghi",
            "axis_catalog_version": CALIBRATION_AXIS_CATALOG_VERSION,
        },
    )
    payload = load_calibration_artifact(path)
    assert payload["schema_version"] == CALIBRATION_ARTIFACT_SCHEMA
    assert payload["method_contract_id"] == CALIBRATION_METHOD_CONTRACT_ID
    assert payload["metadata"]["source_tracker_sha256"] == "abc"
    assert payload["architecture"] == config.__dict__


def test_artifact_rejects_nonfinite_state_before_policy_construction(tmp_path: Path) -> None:
    config = FADAArchitectureConfig(obs_dim=4, action_dim=3, command_dim=2)
    curves = fit_scale_curve_bank(
        torch.linspace(-1.0, 1.0, 21).repeat(3, 1),
        torch.linspace(-0.4, 0.4, 21).repeat(3, 1),
    )
    path = save_calibration_artifact(
        tmp_path / "calibration.pt",
        config=config,
        direction_bank=_normalized_bank(),
        coefficient_encoder=CoefficientEncoder(state_dim=4, action_dim=3, axis_count=3),
        scale_curves=curves,
        metadata={
            "source_tracker_sha256": "abc",
            "dataset_sha256": "def",
            "split_sha256": "ghi",
            "axis_catalog_version": CALIBRATION_AXIS_CATALOG_VERSION,
        },
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["direction_bank"]["directions"][0, 0, 0] = torch.nan
    torch.save(payload, path)
    with pytest.raises(ValueError, match="finite"):
        load_calibration_artifact(path)


def test_artifact_rejects_finite_but_nonmonotone_curve_state(tmp_path: Path) -> None:
    config = FADAArchitectureConfig(obs_dim=4, action_dim=3, command_dim=2)
    curves = fit_scale_curve_bank(
        torch.linspace(-1.0, 1.0, 21).repeat(3, 1),
        torch.linspace(-0.4, 0.4, 21).repeat(3, 1),
    )
    path = save_calibration_artifact(
        tmp_path / "calibration.pt",
        config=config,
        direction_bank=_normalized_bank(),
        coefficient_encoder=CoefficientEncoder(state_dim=4, action_dim=3, axis_count=3),
        scale_curves=curves,
        metadata={
            "source_tracker_sha256": "abc",
            "dataset_sha256": "def",
            "split_sha256": "ghi",
            "axis_catalog_version": CALIBRATION_AXIS_CATALOG_VERSION,
        },
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["scale_curves"][0]["y"][10] = 1.0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="monotone"):
        load_calibration_artifact(path)


def test_legacy_artifact_rejects_before_runtime_owner_construction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from unilab.algos.torch.fada_context import calibration_runtime

    legacy = tmp_path / "legacy.pt"
    torch.save(
        {
            "schema_version": 4,
            "method_contract_id": "FADA-CONTEXT-METHOD-v006",
            "context_state_dict": {},
        },
        legacy,
    )
    calls = 0

    def poison(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("runtime owner construction must be unreachable")

    monkeypatch.setattr(calibration_runtime, "CoefficientEncoder", poison)
    with pytest.raises(ValueError, match="unsupported calibration artifact schema"):
        calibration_runtime.load_calibrated_policy(
            FADAPlannerIDMPolicy(_config()),
            legacy,
            expected_metadata={"source_tracker_sha256": "source"},
        )
    assert calls == 0


def test_calibration_rollout_validation_binds_first_action_and_axis_width() -> None:
    batch = _batch(_config())
    batch.validate(_config(), axis_count=3)
    with pytest.raises(ValueError, match="axis count"):
        batch.validate(_config(), axis_count=2)
    out_of_range = batch.c_true.clone()
    out_of_range[0, 0] = 1.01
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        replace(batch, c_true=out_of_range).validate(_config(), axis_count=3)


def test_calibration_dataset_round_trip_and_legacy_rejection(tmp_path: Path) -> None:
    from unilab.algos.torch.fada_context.calibration_data import (
        load_calibration_dataset,
        save_calibration_dataset,
    )

    config = _config()
    batch = _batch(config)
    split_identity = calibration_split_identity_sha256(batch)
    path = save_calibration_dataset(
        tmp_path / "dataset.pt",
        batch,
        config,
        metadata={
            "source_tracker_sha256": "tracker",
            "axis_catalog_version": CALIBRATION_AXIS_CATALOG_VERSION,
            "split_identity_sha256": split_identity,
        },
    )
    loaded, metadata = load_calibration_dataset(path, config)
    torch.testing.assert_close(loaded.c_true, batch.c_true)
    assert metadata["source_tracker_sha256"] == "tracker"
    with pytest.raises(ValueError, match="split identity"):
        save_calibration_dataset(
            tmp_path / "wrong-split.pt",
            batch,
            config,
            metadata={
                "source_tracker_sha256": "tracker",
                "axis_catalog_version": CALIBRATION_AXIS_CATALOG_VERSION,
                "split_identity_sha256": "wrong",
            },
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["split_id"][0] = 1
    corrupted = tmp_path / "corrupted-split.pt"
    torch.save(payload, corrupted)
    with pytest.raises(ValueError, match="split identity"):
        load_calibration_dataset(corrupted, config)
    legacy = tmp_path / "legacy.pt"
    torch.save({"schema_version": 2, "support_target_future": torch.zeros(1)}, legacy)
    with pytest.raises(ValueError, match="unsupported calibration dataset schema"):
        load_calibration_dataset(legacy, config)


def test_calibrated_policy_zero_readout_preserves_frozen_tracker_and_first_action() -> None:
    config = _config()
    healthy = FADAPlannerIDMPolicy(config).eval()
    bank = DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=config.hidden_dim)
    encoder = CoefficientEncoder(
        state_dim=config.obs_dim, action_dim=config.action_dim, axis_count=3
    )
    curves = tuple(
        MonotoneScaleCurve.fit(torch.linspace(-1.0, 1.0, 21), torch.zeros(21)) for _ in range(3)
    )
    calibrated = CalibratedFADAPolicy(
        config,
        direction_bank=bank,
        coefficient_encoder=encoder,
        scale_curves=curves,
        planner=healthy.planner,
        idm=healthy.idm,
    ).eval()
    observation = torch.randn(2, 30, config.obs_dim)
    action_history = torch.randn(2, 30, config.action_dim)
    command = torch.randn(2, config.command_dim)
    expected = healthy(observation, action_history, command)
    observed = calibrated(observation, action_history, command)
    torch.testing.assert_close(observed.action_chunk, expected.action_chunk)
    torch.testing.assert_close(observed.action, observed.action_chunk[:, 0])


def test_calibrated_policy_explicit_coefficients_match_manual_latent_composition() -> None:
    config = _config()
    healthy = FADAPlannerIDMPolicy(config).eval()
    bank = DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=config.hidden_dim)
    bank.directions.data.normal_()
    encoder = CoefficientEncoder(
        state_dim=config.obs_dim,
        action_dim=config.action_dim,
        axis_count=3,
    )
    curves = tuple(
        MonotoneScaleCurve.fit(torch.linspace(-1.0, 1.0, 21), torch.linspace(-1.0, 1.0, 21))
        for _ in range(3)
    )
    calibrated = CalibratedFADAPolicy(
        config,
        direction_bank=bank,
        coefficient_encoder=encoder,
        scale_curves=curves,
        planner=healthy.planner,
        idm=healthy.idm,
    )
    observation = torch.randn(2, 30, config.obs_dim)
    actions = torch.randn(2, 30, config.action_dim)
    command = torch.randn(2, config.command_dim)
    coefficients = torch.tensor([[0.2, -0.1, 0.3], [-0.4, 0.0, 0.1]])
    observed = calibrated.reconstruct_with_coefficients(
        observation,
        actions,
        command,
        coefficients,
    )
    predicted_future = healthy.planner(observation, command)
    latent = healthy.idm.encode_latent(observation, actions, predicted_future)
    expected_chunk = healthy.idm.decode_latent(bank.compose(latent, coefficients))
    torch.testing.assert_close(observed.action_chunk, expected_chunk)
    torch.testing.assert_close(observed.action, expected_chunk[:, 0])
    permutation = torch.tensor([1, 0])
    permuted = calibrated.reconstruct_with_coefficients(
        observation[permutation],
        actions[permutation],
        command[permutation],
        coefficients[permutation],
    )
    torch.testing.assert_close(permuted.action_chunk, observed.action_chunk[permutation])
    with pytest.raises(ValueError, match="coefficients"):
        calibrated.reconstruct_with_coefficients(
            observation,
            actions,
            command,
            torch.zeros(2, 2),
        )
    assert all(not parameter.requires_grad for parameter in calibrated.parameters())


def test_readout_state_cold_start_range_and_jump_freeze_are_explicit() -> None:
    curves = tuple(
        MonotoneScaleCurve.fit(torch.linspace(-1.0, 1.0, 21), torch.linspace(-0.2, 0.2, 21))
        for _ in range(3)
    )
    state = CalibrationReadoutState(axis_count=3, jump_threshold=torch.tensor([0.5, 0.5, 0.5]))
    cold = state.apply(torch.tensor([[0.2, 0.3, 0.4]]), curves, ready=torch.tensor([False]))
    torch.testing.assert_close(cold.scales, torch.zeros(1, 3), rtol=0.0, atol=0.0)
    assert cold.cold_start.tolist() == [True]
    first = state.apply(torch.tensor([[0.2, 2.0, 0.4]]), curves, ready=torch.tensor([True]))
    assert first.range_events.tolist() == [[False, True, False]]
    second = state.apply(torch.tensor([[0.9, 0.9, 0.4]]), curves, ready=torch.tensor([True]))
    assert second.jump_events.tolist() == [[True, True, False]]
    torch.testing.assert_close(second.scales[0, :2], first.scales[0, :2])
    state.reset()
    assert state.previous_coefficients is None
    with pytest.raises(ValueError, match="ready mask"):
        state.apply(torch.zeros(1, 3), curves, ready=torch.tensor([True, False]))


def test_calibrated_playback_uses_true_history_count_before_encoder(monkeypatch) -> None:
    config = _config()
    healthy = FADAPlannerIDMPolicy(config).eval()
    encoder = CoefficientEncoder(state_dim=4, action_dim=3, axis_count=3)
    calls = 0
    original = encoder.forward

    def record(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(encoder, "forward", record)
    curves = tuple(
        MonotoneScaleCurve.fit(torch.linspace(-1.0, 1.0, 21), torch.zeros(21)) for _ in range(3)
    )
    policy = CalibratedFADAPolicy(
        config,
        direction_bank=DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=8),
        coefficient_encoder=encoder,
        scale_curves=curves,
        planner=healthy.planner,
        idm=healthy.idm,
    )
    controller = CalibratedFADAPlaybackController(
        policy,
        device="cpu",
        jump_threshold=torch.ones(3),
    )
    for _ in range(29):
        controller.act(torch.zeros(4), torch.zeros(2))
    assert calls == 0
    assert controller.last_readout is not None
    assert controller.last_readout.cold_start.tolist() == [True]
    controller.act(torch.zeros(4), torch.zeros(2))
    assert calls == 1
    assert controller.last_readout is not None
    assert controller.last_readout.cold_start.tolist() == [False]
    controller.reset()
    assert controller.last_readout is None
    with pytest.raises(ValueError, match="finite"):
        controller.act(torch.full((4,), torch.nan), torch.zeros(2))
    controller.act(torch.zeros(4), torch.zeros(2))
    assert calls == 1


def test_calibrated_playback_partial_reset_is_row_local(monkeypatch) -> None:
    config = _config()
    healthy = FADAPlannerIDMPolicy(config).eval()
    encoder = CoefficientEncoder(state_dim=4, action_dim=3, axis_count=3)
    observed_batch_sizes = []
    original = encoder.forward

    def record(state, action):
        observed_batch_sizes.append(int(state.shape[0]))
        return original(state, action)

    monkeypatch.setattr(encoder, "forward", record)
    curves = tuple(
        MonotoneScaleCurve.fit(torch.linspace(-1.0, 1.0, 21), torch.zeros(21)) for _ in range(3)
    )
    controller = CalibratedFADAPlaybackController(
        CalibratedFADAPolicy(
            config,
            direction_bank=DirectionBank(axis_count=3, prediction_horizon=6, latent_dim=8),
            coefficient_encoder=encoder,
            scale_curves=curves,
            planner=healthy.planner,
            idm=healthy.idm,
        ),
        device="cpu",
        jump_threshold=torch.ones(3),
    )
    for _ in range(29):
        controller.act(torch.zeros(2, 4), torch.zeros(2, 2))
    controller.reset(torch.tensor([True, False]))
    controller.act(torch.zeros(2, 4), torch.zeros(2, 2))
    assert observed_batch_sizes == [1]
    assert controller.last_readout is not None
    assert controller.last_readout.cold_start.tolist() == [True, False]
