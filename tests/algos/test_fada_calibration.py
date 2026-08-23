from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from unilab.algos.torch.distill.fada import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context import calibration as calibration_owner
from unilab.algos.torch.fada_context import calibration_data as calibration_data_owner
from unilab.algos.torch.fada_context.calibration import (
    CALIBRATION_ARTIFACT_SCHEMA,
    CALIBRATION_AXIS_CATALOG_VERSION,
    CALIBRATION_METHOD_CONTRACT_ID,
    CalibratedFADAPolicy,
    CalibrationAxisSpec,
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
    LoadedCalibrationDataset,
    calibration_split_identity_sha256,
    load_calibration_dataset,
    load_fault_axis_catalog,
    prepare_calibration_rollout_batch,
    project_calibration_rollout_batch,
    save_calibration_dataset,
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


def _axis_spec(names: tuple[str, ...] | None = None) -> CalibrationAxisSpec:
    return CalibrationAxisSpec.from_catalog(FaultAxisCatalog.default(), names)


def _artifact_metadata() -> dict[str, str]:
    return {
        "source_tracker_sha256": "1" * 64,
        "dataset_sha256": "2" * 64,
        "split_sha256": "3" * 64,
        "stage": "complete",
        "parent_stage_sha256": "4" * 64,
        "scale_evidence_sha256": "5" * 64,
    }


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


def test_axis_spec_preserves_the_requested_non_catalog_order() -> None:
    catalog = FaultAxisCatalog.default()
    spec = CalibrationAxisSpec.from_catalog(catalog, ("offset", "gain"))
    assert spec.catalog_version == CALIBRATION_AXIS_CATALOG_VERSION
    assert spec.names == ("offset", "gain")
    assert spec.axis_count == 2
    assert spec.catalog_indices(catalog) == (2, 0)


def test_axis_spec_has_one_canonical_round_trip_payload() -> None:
    catalog = FaultAxisCatalog.default()
    spec = CalibrationAxisSpec.from_catalog(catalog, ("delay", "gain"))
    payload = {"catalog_version": CALIBRATION_AXIS_CATALOG_VERSION, "names": ["delay", "gain"]}
    assert spec.to_payload() == payload
    assert CalibrationAxisSpec.from_payload(payload, catalog) == spec


@pytest.mark.parametrize(
    ("names", "message"),
    [
        ((), "non-empty"),
        (("gain", "gain"), "unique"),
        (("gain", "friction"), "unregistered"),
    ],
)
def test_axis_spec_rejects_invalid_selections(names: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CalibrationAxisSpec.from_catalog(FaultAxisCatalog.default(), names)


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


def test_rollout_projection_preserves_requested_axis_order_and_remaps_rows() -> None:
    catalog = FaultAxisCatalog.default()
    spec = CalibrationAxisSpec.from_catalog(catalog, ("offset", "gain"))
    projected = project_calibration_rollout_batch(_batch(_config()), catalog, spec)
    assert projected.rollout_id.tolist() == [10, 12]
    assert projected.axis_id.tolist() == [1, 0]
    torch.testing.assert_close(
        projected.c_true,
        torch.tensor([[0.0, 0.2], [0.7, 0.0]]),
    )
    projected.validate(_config(), axis_count=2)


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
        axis_spec=_axis_spec(),
        metadata=_artifact_metadata(),
    )
    payload = load_calibration_artifact(path, FaultAxisCatalog.default())
    assert payload["schema_version"] == CALIBRATION_ARTIFACT_SCHEMA
    assert payload["method_contract_id"] == CALIBRATION_METHOD_CONTRACT_ID
    assert payload["metadata"] == _artifact_metadata()
    assert payload["architecture"] == config.__dict__


def test_artifact_writer_rejects_axis_count_in_provenance_metadata(tmp_path: Path) -> None:
    metadata = {**_artifact_metadata(), "axis_count": 3}
    with pytest.raises(ValueError, match="reserved axis identity"):
        save_calibration_artifact(
            tmp_path / "calibration.pt",
            config=FADAArchitectureConfig(obs_dim=4, action_dim=3, command_dim=2),
            direction_bank=_normalized_bank(),
            coefficient_encoder=CoefficientEncoder(state_dim=4, action_dim=3, axis_count=3),
            scale_curves=fit_scale_curve_bank(
                torch.linspace(-1.0, 1.0, 21).repeat(3, 1),
                torch.linspace(-0.4, 0.4, 21).repeat(3, 1),
            ),
            axis_spec=_axis_spec(),
            metadata=metadata,
        )


@pytest.mark.parametrize("preexisting", [False, True])
def test_artifact_serialization_failure_cleans_temp_and_preserves_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting: bool,
) -> None:
    target = tmp_path / "calibration.pt"
    if preexisting:
        target.write_bytes(b"old-artifact")

    def fail_after_partial_write(value, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("injected artifact serialization failure")

    monkeypatch.setattr(calibration_owner.torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="artifact serialization failure"):
        save_calibration_artifact(
            target,
            config=FADAArchitectureConfig(obs_dim=4, action_dim=3, command_dim=2),
            direction_bank=_normalized_bank(),
            coefficient_encoder=CoefficientEncoder(state_dim=4, action_dim=3, axis_count=3),
            scale_curves=fit_scale_curve_bank(
                torch.linspace(-1.0, 1.0, 21).repeat(3, 1),
                torch.linspace(-0.4, 0.4, 21).repeat(3, 1),
            ),
            axis_spec=_axis_spec(),
            metadata=_artifact_metadata(),
        )
    assert list(tmp_path.glob(".calibration.pt.*.tmp")) == []
    if preexisting:
        assert target.read_bytes() == b"old-artifact"
    else:
        assert not target.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage", None),
        ("stage", "coefficient_frozen"),
        ("source_tracker_sha256", "a" * 63),
        ("source_tracker_sha256", "A" * 64),
        ("source_tracker_sha256", "g" * 64),
        ("dataset_sha256", "b" * 63),
        ("dataset_sha256", "B" * 64),
        ("dataset_sha256", "h" * 64),
        ("split_sha256", "c" * 63),
        ("split_sha256", "C" * 64),
        ("split_sha256", "i" * 64),
        ("parent_stage_sha256", None),
        ("parent_stage_sha256", "a" * 63),
        ("parent_stage_sha256", "A" * 64),
        ("parent_stage_sha256", "g" * 64),
        ("scale_evidence_sha256", None),
        ("scale_evidence_sha256", "b" * 63),
        ("scale_evidence_sha256", "B" * 64),
        ("scale_evidence_sha256", "h" * 64),
    ],
)
def test_artifact_writer_rejects_missing_or_malformed_lineage(
    tmp_path: Path,
    field: str,
    value: str | None,
) -> None:
    metadata = _artifact_metadata()
    if value is None:
        metadata.pop(field)
    else:
        metadata[field] = value
    target = tmp_path / "calibration.pt"
    with pytest.raises(ValueError, match="lineage"):
        save_calibration_artifact(
            target,
            config=FADAArchitectureConfig(obs_dim=4, action_dim=3, command_dim=2),
            direction_bank=_normalized_bank(),
            coefficient_encoder=CoefficientEncoder(state_dim=4, action_dim=3, axis_count=3),
            scale_curves=fit_scale_curve_bank(
                torch.linspace(-1.0, 1.0, 21).repeat(3, 1),
                torch.linspace(-0.4, 0.4, 21).repeat(3, 1),
            ),
            axis_spec=_axis_spec(),
            metadata=metadata,
        )
    assert not target.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage", None),
        ("stage", "direction_frozen"),
        ("source_tracker_sha256", "a" * 63),
        ("source_tracker_sha256", "A" * 64),
        ("source_tracker_sha256", "g" * 64),
        ("dataset_sha256", "b" * 63),
        ("dataset_sha256", "B" * 64),
        ("dataset_sha256", "h" * 64),
        ("split_sha256", "c" * 63),
        ("split_sha256", "C" * 64),
        ("split_sha256", "i" * 64),
        ("parent_stage_sha256", None),
        ("parent_stage_sha256", "a" * 63),
        ("parent_stage_sha256", "A" * 64),
        ("parent_stage_sha256", "g" * 64),
        ("scale_evidence_sha256", None),
        ("scale_evidence_sha256", "b" * 63),
        ("scale_evidence_sha256", "B" * 64),
        ("scale_evidence_sha256", "h" * 64),
    ],
)
def test_artifact_loader_rejects_missing_or_malformed_lineage(
    tmp_path: Path,
    field: str,
    value: str | None,
) -> None:
    target = save_calibration_artifact(
        tmp_path / "calibration.pt",
        config=FADAArchitectureConfig(obs_dim=4, action_dim=3, command_dim=2),
        direction_bank=_normalized_bank(),
        coefficient_encoder=CoefficientEncoder(state_dim=4, action_dim=3, axis_count=3),
        scale_curves=fit_scale_curve_bank(
            torch.linspace(-1.0, 1.0, 21).repeat(3, 1),
            torch.linspace(-0.4, 0.4, 21).repeat(3, 1),
        ),
        axis_spec=_axis_spec(),
        metadata=_artifact_metadata(),
    )
    payload = torch.load(target, map_location="cpu", weights_only=True)
    if value is None:
        payload["metadata"].pop(field)
    else:
        payload["metadata"][field] = value
    torch.save(payload, target)
    with pytest.raises(ValueError, match="lineage"):
        load_calibration_artifact(target, FaultAxisCatalog.default())


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
        axis_spec=_axis_spec(),
        metadata=_artifact_metadata(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["direction_bank"]["directions"][0, 0, 0] = torch.nan
    torch.save(payload, path)
    with pytest.raises(ValueError, match="finite"):
        load_calibration_artifact(path, FaultAxisCatalog.default())


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
        axis_spec=_axis_spec(),
        metadata=_artifact_metadata(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["scale_curves"][0]["y"][10] = 1.0
    torch.save(payload, path)
    with pytest.raises(ValueError, match="monotone"):
        load_calibration_artifact(path, FaultAxisCatalog.default())


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
            catalog=FaultAxisCatalog.default(),
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
    config = _config()
    batch = _batch(config)
    catalog = FaultAxisCatalog.default()
    axis_spec = CalibrationAxisSpec.from_catalog(catalog)
    split_identity = calibration_split_identity_sha256(batch)
    path = save_calibration_dataset(
        tmp_path / "dataset.pt",
        batch,
        config,
        axis_spec=axis_spec,
        metadata={
            "source_tracker_sha256": "tracker",
            "split_identity_sha256": split_identity,
        },
    )
    loaded = load_calibration_dataset(path, config, catalog)
    assert isinstance(loaded, LoadedCalibrationDataset)
    assert loaded.axis_spec == axis_spec
    torch.testing.assert_close(loaded.batch.c_true, batch.c_true)
    assert loaded.metadata["source_tracker_sha256"] == "tracker"
    with pytest.raises(ValueError, match="split identity"):
        save_calibration_dataset(
            tmp_path / "wrong-split.pt",
            batch,
            config,
            axis_spec=axis_spec,
            metadata={
                "source_tracker_sha256": "tracker",
                "split_identity_sha256": "wrong",
            },
        )
    with pytest.raises(ValueError, match="reserved axis identity"):
        save_calibration_dataset(
            tmp_path / "reserved-metadata.pt",
            batch,
            config,
            axis_spec=axis_spec,
            metadata={
                "source_tracker_sha256": "tracker",
                "split_identity_sha256": split_identity,
                "axis_names": ["gain", "delay", "offset"],
            },
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["split_id"][0] = 1
    corrupted = tmp_path / "corrupted-split.pt"
    torch.save(payload, corrupted)
    with pytest.raises(ValueError, match="split identity"):
        load_calibration_dataset(corrupted, config, catalog)
    legacy = tmp_path / "legacy.pt"
    torch.save({"schema_version": 2, "support_target_future": torch.zeros(1)}, legacy)
    with pytest.raises(ValueError, match="unsupported calibration dataset schema"):
        load_calibration_dataset(legacy, config, catalog)


@pytest.mark.parametrize("preexisting", [False, True])
def test_dataset_serialization_failure_cleans_temp_and_preserves_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preexisting: bool,
) -> None:
    config = _config()
    batch = _batch(config)
    target = tmp_path / "dataset.pt"
    if preexisting:
        target.write_bytes(b"old-dataset")

    def fail_after_partial_write(value, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("injected dataset serialization failure")

    monkeypatch.setattr(calibration_data_owner.torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="dataset serialization failure"):
        save_calibration_dataset(
            target,
            batch,
            config,
            axis_spec=_axis_spec(),
            metadata={
                "source_tracker_sha256": "tracker",
                "split_identity_sha256": calibration_split_identity_sha256(batch),
            },
        )
    assert list(tmp_path.glob(".dataset.pt.*.tmp")) == []
    if preexisting:
        assert target.read_bytes() == b"old-dataset"
    else:
        assert not target.exists()


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
        axis_spec=_axis_spec(),
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
        axis_spec=_axis_spec(),
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
        axis_spec=_axis_spec(),
        planner=healthy.planner,
        idm=healthy.idm,
    )
    controller = CalibratedFADAPlaybackController(
        policy,
        device="cpu",
        jump_threshold={"gain": 1.0, "delay": 1.0, "offset": 1.0},
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
            axis_spec=_axis_spec(),
            planner=healthy.planner,
            idm=healthy.idm,
        ),
        device="cpu",
        jump_threshold={"gain": 1.0, "delay": 1.0, "offset": 1.0},
    )
    for _ in range(29):
        controller.act(torch.zeros(2, 4), torch.zeros(2, 2))
    controller.reset(torch.tensor([True, False]))
    controller.act(torch.zeros(2, 4), torch.zeros(2, 2))
    assert observed_batch_sizes == [1]
    assert controller.last_readout is not None
    assert controller.last_readout.cold_start.tolist() == [True, False]


def test_playback_resolves_named_jump_thresholds_in_artifact_axis_order() -> None:
    config = _config()
    healthy = FADAPlannerIDMPolicy(config).eval()
    axis_spec = _axis_spec(("offset", "gain"))
    curves = tuple(
        MonotoneScaleCurve.fit(torch.linspace(-1.0, 1.0, 21), torch.zeros(21))
        for _ in axis_spec.names
    )
    policy = CalibratedFADAPolicy(
        config,
        direction_bank=DirectionBank(axis_count=2, prediction_horizon=6, latent_dim=8),
        coefficient_encoder=CoefficientEncoder(state_dim=4, action_dim=3, axis_count=2),
        scale_curves=curves,
        axis_spec=axis_spec,
        planner=healthy.planner,
        idm=healthy.idm,
    )
    controller = CalibratedFADAPlaybackController(
        policy,
        device="cpu",
        jump_threshold={"gain": 0.1, "offset": 0.3},
    )
    torch.testing.assert_close(
        controller.readout_state.jump_threshold,
        torch.tensor([0.3, 0.1]),
    )
