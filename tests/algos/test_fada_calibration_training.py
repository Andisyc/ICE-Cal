from __future__ import annotations

import hashlib
import importlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import unilab.algos.torch.fada_context as fada_context
from unilab.algos.torch.distill.fada import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context import calibration_training
from unilab.algos.torch.fada_context.calibration_training import (
    io as training_io_owner,
)
from unilab.algos.torch.fada_context.calibration_training import (
    stage1 as stage1_owner,
)
from unilab.algos.torch.fada_context.calibration_training import (
    stage2 as stage2_owner,
)
from unilab.algos.torch.fada_context.calibration_training import (
    stage3 as stage3_owner,
)

_SOURCE_SHA256 = "1" * 64
_DATASET_SHA256 = "2" * 64
_SPLIT_SHA256 = "3" * 64
from unilab.algos.torch.fada_context.calibration import (
    CalibrationAxisSpec,
    CalibrationRolloutBatch,
    CoefficientEncoder,
    DirectionBank,
    FaultAxisCatalog,
    fit_scale_curve_bank,
    load_calibration_artifact,
)
from unilab.algos.torch.fada_context.calibration_data import (
    project_calibration_rollout_batch,
)
from unilab.algos.torch.fada_context.calibration_runtime import load_calibrated_policy
from unilab.algos.torch.fada_context.calibration_training import (
    CALIBRATION_STAGE_ARTIFACT_SCHEMA,
    CalibrationScaleEvidence,
    CalibrationStageIdentity,
    CoefficientStageConfig,
    DirectionGeometryConfig,
    DirectionStageConfig,
    SerialCalibrationConfig,
    calibration_compensation_ratio,
    coefficient_stage_loss,
    coefficient_validation_error,
    direction_stage_loss,
    fit_scale_stage,
    load_calibration_scale_evidence,
    run_coefficient_stage_training,
    run_direction_stage_training,
    run_scale_stage_fitting,
    run_serial_calibration_training,
    save_calibration_scale_evidence,
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
        "source_tracker_sha256": _SOURCE_SHA256,
        "dataset_sha256": _DATASET_SHA256,
        "split_sha256": _SPLIT_SHA256,
    }


def _axis_spec(names: tuple[str, ...] | None = None) -> CalibrationAxisSpec:
    return CalibrationAxisSpec.from_catalog(FaultAxisCatalog.default(), names)


def _identity(names: tuple[str, ...] | None = None) -> CalibrationStageIdentity:
    return CalibrationStageIdentity(**_metadata(), axis_spec=_axis_spec(names))


def _scale_evidence(
    names: tuple[str, ...] | None = None,
) -> CalibrationScaleEvidence:
    axis_spec = _axis_spec(names)
    coefficient_scan_grid = torch.linspace(-1.0, 1.0, 21).repeat(axis_spec.axis_count, 1)
    readings = torch.linspace(-1.0, 1.0, 21).view(1, 21, 1).repeat(axis_spec.axis_count, 1, 32)
    candidates = torch.linspace(-0.5, 0.5, 41)
    desired = 0.2 * readings
    errors = (candidates.view(1, 1, 1, -1) - desired.unsqueeze(-1)).square()
    return CalibrationScaleEvidence(
        coefficient_scan_grid=coefficient_scan_grid,
        readings=readings,
        candidate_scales=candidates,
        action_errors=errors,
        axis_spec=axis_spec,
        metadata=_metadata(),
    )


def _admitted_batch(
    policy: FADAPlannerIDMPolicy,
) -> CalibrationRolloutBatch:
    config = policy.config
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
    return CalibrationRolloutBatch(
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


def _gain_geometry_batch(policy: FADAPlannerIDMPolicy) -> CalibrationRolloutBatch:
    source = project_calibration_rollout_batch(
        _admitted_batch(policy),
        FaultAxisCatalog.default(),
        _axis_spec(("gain",)),
        config=policy.config,
    )
    indices = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64)
    selected = source.index_select(indices)
    c_true = selected.c_true.clone()
    c_true[[2, 5], 0] = 0.0
    target = selected.target_action_chunk.clone()
    target[[2, 5]] = selected.nominal_action_chunk[[2, 5]]
    return replace(
        selected,
        target_action_chunk=target,
        c_true=c_true,
        rollout_id=torch.arange(6, dtype=torch.int64) + 500,
        seed=torch.arange(6, dtype=torch.int64) + 600,
        split_id=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int64),
    )


def _append_held_out_combinations(batch: CalibrationRolloutBatch) -> CalibrationRolloutBatch:
    extras = {
        name: getattr(batch, name)[:2].clone()
        for name in CalibrationRolloutBatch.__dataclass_fields__
    }
    extras["c_true"] = torch.tensor(
        [[0.2, 0.0, 0.4], [0.2, 0.3, 0.0]],
        dtype=batch.c_true.dtype,
    )
    extras["axis_id"] = torch.full((2,), -1, dtype=torch.int64)
    extras["is_held_out_combination"] = torch.ones(2, dtype=torch.bool)
    extras["rollout_id"] = torch.tensor([1000, 1001], dtype=torch.int64)
    extras["seed"] = torch.tensor([2000, 2001], dtype=torch.int64)
    extras["split_id"] = torch.zeros(2, dtype=torch.int64)
    return CalibrationRolloutBatch(
        **{
            name: torch.cat((getattr(batch, name), extras[name]), dim=0)
            for name in CalibrationRolloutBatch.__dataclass_fields__
        }
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _assert_snapshot(
    module: torch.nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> None:
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, snapshot[name], rtol=0.0, atol=0.0)


def _make_direction_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: FADAPlannerIDMPolicy,
    batch: CalibrationRolloutBatch,
) -> Path:
    monkeypatch.setattr(
        stage1_owner,
        "direction_stage_compensation_ratio",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    path = tmp_path / "stage1.pt"
    run_direction_stage_training(
        policy,
        batch,
        path,
        _identity(),
        DirectionStageConfig(steps_per_axis=1),
    )
    return path


def _make_coefficient_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: FADAPlannerIDMPolicy,
    batch: CalibrationRolloutBatch,
) -> Path:
    direction_path = _make_direction_artifact(tmp_path, monkeypatch, policy, batch)
    monkeypatch.setattr(
        stage2_owner,
        "coefficient_validation_error",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    path = tmp_path / "stage2.pt"
    run_coefficient_stage_training(
        policy,
        batch,
        direction_path,
        path,
        _identity(),
        CoefficientStageConfig(steps=1),
    )
    return path


def test_stage_isolation_public_export_surface_has_no_generic_checkpoint_api() -> None:
    for name in (
        "CALIBRATION_STAGE_ARTIFACT_SCHEMA",
        "CalibrationStageIdentity",
        "DirectionStageConfig",
        "CoefficientStageConfig",
        "DirectionStageResult",
        "CoefficientStageResult",
        "ScaleStageResult",
        "run_direction_stage_training",
        "run_coefficient_stage_training",
        "run_scale_stage_fitting",
    ):
        assert hasattr(fada_context, name), name
    for name in (
        "CALIBRATION_CHECKPOINT_SCHEMA",
        "save_calibration_training_checkpoint",
        "load_calibration_training_checkpoint",
    ):
        assert not hasattr(fada_context, name), name


def test_calibration_training_is_split_into_stage_owner_modules() -> None:
    assert hasattr(calibration_training, "__path__")
    owners = {
        "stage1": run_direction_stage_training,
        "stage2": run_coefficient_stage_training,
        "stage3": run_scale_stage_fitting,
        "pipeline": run_serial_calibration_training,
    }
    for module_name, entrypoint in owners.items():
        module = importlib.import_module(f"{calibration_training.__name__}.{module_name}")
        assert entrypoint is getattr(module, entrypoint.__name__)


def test_stage_owned_configs_cannot_relax_active_contract_gates() -> None:
    with pytest.raises(ValueError, match=r"0\.1"):
        DirectionStageConfig(compensation_ratio_threshold=0.11)
    with pytest.raises(ValueError, match=r"0\.05"):
        CoefficientStageConfig(coefficient_error_threshold=0.051)
    with pytest.raises(ValueError, match="compensation_ratio_threshold"):
        SerialCalibrationConfig(compensation_ratio_threshold=0.11)
    with pytest.raises(ValueError, match="coefficient_error_threshold"):
        SerialCalibrationConfig(coefficient_error_threshold=0.051)


@pytest.mark.parametrize("owner", ["direction", "coefficient", "serial"])
@pytest.mark.parametrize("learning_rate", [float("nan"), float("inf")])
def test_stage_configs_reject_nonfinite_learning_rate_before_adam(
    monkeypatch,
    owner,
    learning_rate,
) -> None:
    calls = 0

    def poison(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("Adam must be unreachable")

    monkeypatch.setattr(torch.optim, "Adam", poison)
    constructor = {
        "direction": DirectionStageConfig,
        "coefficient": CoefficientStageConfig,
        "serial": SerialCalibrationConfig,
    }[owner]
    with pytest.raises(ValueError, match="learning_rate must be finite and positive"):
        constructor(learning_rate=learning_rate)
    assert calls == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"training_split_id": 1, "validation_split_id": 1},
        {"minimum_abs_coefficient": 0.0},
        {"minimum_abs_coefficient": float("nan")},
    ],
)
def test_direction_geometry_config_rejects_unidentifiable_boundaries(kwargs) -> None:
    with pytest.raises(ValueError, match="direction geometry"):
        DirectionGeometryConfig(**kwargs)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("source_tracker_sha256", "A" * 64),
        ("dataset_sha256", "b" * 63),
        ("split_sha256", "g" * 64),
    ],
)
def test_stage_identity_rejects_noncanonical_digest_before_adam_or_publication(
    tmp_path,
    monkeypatch,
    field,
    invalid,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    target = tmp_path / "stage1.pt"
    calls = 0

    def poison(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("Adam must be unreachable")

    monkeypatch.setattr(torch.optim, "Adam", poison)
    with pytest.raises(ValueError, match="64-character lowercase hexadecimal"):
        run_direction_stage_training(
            policy,
            batch,
            target,
            replace(_identity(), **{field: invalid}),
            DirectionStageConfig(steps_per_axis=1),
        )
    assert calls == 0
    assert not target.exists()


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


def test_stage1_diagnostics_record_requested_steps_without_mutating_policy() -> None:
    torch.manual_seed(17)
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    policy_snapshot = _snapshot(policy)
    gradients = [parameter.grad for parameter in policy.parameters()]
    requires_grad = [parameter.requires_grad for parameter in policy.parameters()]
    training_modes = [module.training for module in policy.modules()]
    config_type = getattr(calibration_training, "DirectionDiagnosticConfig")
    diagnose = getattr(calibration_training, "diagnose_direction_stage_training")

    points = diagnose(
        policy,
        batch,
        _identity(),
        config_type(checkpoint_steps=(0, 1), learning_rate=1.0e-3),
    )

    assert [(point.axis_index, point.step) for point in points] == [
        (axis_index, step)
        for axis_index in range(3)
        for step in (0, 1)
    ]
    assert all(
        torch.isfinite(
            torch.tensor(
                [
                    point.training_loss,
                    point.training_compensation_ratio,
                    point.validation_compensation_ratio,
                    point.direction_norm,
                ]
            )
        ).all()
        for point in points
    )
    assert all(point.direction_norm == 0.0 for point in points[::2])
    assert all(point.direction_norm > 0.0 for point in points[1::2])
    _assert_snapshot(policy, policy_snapshot)
    assert [parameter.grad for parameter in policy.parameters()] == gradients
    assert [parameter.requires_grad for parameter in policy.parameters()] == requires_grad
    assert [module.training for module in policy.modules()] == training_modes


def test_direction_geometry_summary_distinguishes_aligned_orthogonal_and_opposite() -> None:
    owner = importlib.import_module(
        "unilab.algos.torch.fada_context.calibration_training.direction_geometry"
    )
    ratios = torch.tensor([0.01, 0.04])

    aligned = owner.summarize_direction_geometry(
        torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        ratios,
        split_id=0,
        excluded_zero_coefficient_count=1,
        excluded_zero_target_error_count=2,
    )
    assert aligned.sample_count == 2
    assert aligned.excluded_zero_coefficient_count == 1
    assert aligned.excluded_zero_target_error_count == 2
    assert aligned.top1_energy_fraction == pytest.approx(1.0)
    assert aligned.cosine_to_consensus_mean == pytest.approx(1.0)
    assert aligned.cosine_to_consensus_p10 == pytest.approx(1.0)
    assert aligned.opposing_direction_fraction == pytest.approx(0.0)
    assert aligned.individual_gate_fraction == pytest.approx(1.0)
    assert aligned.direction_norm_p10 == pytest.approx(1.1)
    assert aligned.direction_norm_median == pytest.approx(1.5)
    assert aligned.direction_norm_p90 == pytest.approx(1.9)
    assert aligned.direction_norm_p90_p10_ratio == pytest.approx(1.9 / 1.1)

    orthogonal = owner.summarize_direction_geometry(
        torch.eye(2),
        ratios,
        split_id=0,
        excluded_zero_coefficient_count=0,
        excluded_zero_target_error_count=0,
    )
    assert orthogonal.top1_energy_fraction == pytest.approx(0.5)

    opposite = owner.summarize_direction_geometry(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
        ratios,
        split_id=0,
        excluded_zero_coefficient_count=0,
        excluded_zero_target_error_count=0,
    )
    assert opposite.top1_energy_fraction == pytest.approx(1.0)
    assert opposite.opposing_direction_fraction == pytest.approx(0.5)


def test_direction_geometry_summary_rejects_unidentifiable_or_nonfinite_inputs() -> None:
    owner = importlib.import_module(
        "unilab.algos.torch.fada_context.calibration_training.direction_geometry"
    )
    with pytest.raises(ValueError, match="two nonzero"):
        owner.summarize_direction_geometry(
            torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            torch.tensor([0.1, 1.0]),
            split_id=0,
            excluded_zero_coefficient_count=0,
            excluded_zero_target_error_count=0,
        )
    with pytest.raises(ValueError, match="finite"):
        owner.summarize_direction_geometry(
            torch.tensor([[1.0, 0.0], [float("nan"), 1.0]]),
            torch.tensor([0.1, 0.2]),
            split_id=0,
            excluded_zero_coefficient_count=0,
            excluded_zero_target_error_count=0,
        )


def test_direction_geometry_diagnostic_is_exact_first_action_only_and_restores_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(23)
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _gain_geometry_batch(policy)
    policy_snapshot = _snapshot(policy)
    gradients = [parameter.grad for parameter in policy.parameters()]
    requires_grad = [parameter.requires_grad for parameter in policy.parameters()]
    training_modes = [module.training for module in policy.modules()]
    owner = importlib.import_module(
        "unilab.algos.torch.fada_context.calibration_training.direction_geometry"
    )
    config_type = getattr(calibration_training, "DirectionGeometryConfig")

    def poison_adam(*args, **kwargs):
        raise AssertionError("analytic direction geometry must not construct Adam")

    monkeypatch.setattr(torch.optim, "Adam", poison_adam)

    reports = owner.diagnose_direction_geometry(
        policy,
        batch,
        _identity(("gain",)),
        config_type(),
    )
    changed_future_target = batch.target_action_chunk.clone()
    changed_future_target[:, 1:] += 100.0
    counterfactual_reports = owner.diagnose_direction_geometry(
        policy,
        replace(batch, target_action_chunk=changed_future_target),
        _identity(("gain",)),
        config_type(),
    )

    assert len(reports) == 1
    report = reports[0]
    assert report.axis_index == 0
    assert report.supervision_scope == "executed_first_action"
    assert report.solver == "linear_decoder_minimum_norm"
    assert report.training.split_id == 0
    assert report.validation.split_id == 1
    assert report.training.sample_count == 2
    assert report.validation.sample_count == 2
    assert report.training.excluded_zero_coefficient_count == 1
    assert report.validation.excluded_zero_coefficient_count == 1
    assert reports == counterfactual_reports
    assert report.shared_training_ratio < 1.0e-8
    assert report.shared_validation_ratio < 1.0e-8
    assert report.training.individual_ratio_max < 1.0e-8
    assert report.validation.individual_ratio_max < 1.0e-8
    assert torch.isfinite(
        torch.tensor(
            [
                report.shared_training_ratio,
                report.shared_validation_ratio,
                report.training.top1_energy_fraction,
                report.validation.top1_energy_fraction,
            ]
        )
    ).all()
    _assert_snapshot(policy, policy_snapshot)
    assert [parameter.grad for parameter in policy.parameters()] == gradients
    assert [parameter.requires_grad for parameter in policy.parameters()] == requires_grad
    assert [module.training for module in policy.modules()] == training_modes


def test_stage1_formal_and_diagnostic_paths_share_optimizer_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    shared_step = getattr(stage1_owner, "_direction_stage_step")
    calls: list[int] = []

    def spy(*args, **kwargs):
        calls.append(int(kwargs["axis_index"]))
        return shared_step(*args, **kwargs)

    monkeypatch.setattr(stage1_owner, "_direction_stage_step", spy)
    monkeypatch.setattr(
        stage1_owner,
        "direction_stage_compensation_ratio",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    run_direction_stage_training(
        policy,
        batch,
        tmp_path / "stage1.pt",
        _identity(),
        DirectionStageConfig(steps_per_axis=1),
    )
    assert calls == [0, 1, 2]

    calls.clear()
    config_type = getattr(calibration_training, "DirectionDiagnosticConfig")
    diagnose = getattr(calibration_training, "diagnose_direction_stage_training")
    diagnose(
        policy,
        batch,
        _identity(),
        config_type(checkpoint_steps=(0, 1)),
    )
    assert calls == [0, 1, 2]


def test_stage1_diagnostic_checkpoint_uses_the_post_step_direction_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    updates = [0, 0, 0]

    def deterministic_step(
        policy,
        direction_bank,
        training_batch,
        optimizer,
        policy_snapshot,
        *,
        axis_index,
    ):
        del policy, training_batch, optimizer, policy_snapshot
        updates[axis_index] += 1
        direction_bank.directions.data[axis_index].fill_(float(updates[axis_index]))
        return torch.tensor(float(updates[axis_index]))

    def norm_metric(policy, direction_bank, batch, *, axis_index):
        del policy, batch
        return direction_bank.directions[axis_index].norm()

    monkeypatch.setattr(stage1_owner, "_direction_stage_step", deterministic_step)
    monkeypatch.setattr(stage1_owner, "direction_stage_loss", norm_metric)
    monkeypatch.setattr(stage1_owner, "direction_stage_compensation_ratio", norm_metric)
    config_type = getattr(calibration_training, "DirectionDiagnosticConfig")
    diagnose = getattr(calibration_training, "diagnose_direction_stage_training")

    points = diagnose(
        policy,
        batch,
        _identity(),
        config_type(checkpoint_steps=(0, 1, 2)),
    )

    latent_elements = policy.config.prediction_horizon * policy.config.hidden_dim
    expected = {
        0: 0.0,
        1: latent_elements**0.5,
        2: 2.0 * latent_elements**0.5,
    }
    for point in points:
        assert point.direction_norm == pytest.approx(expected[point.step])
        assert point.training_loss == pytest.approx(expected[point.step])
        assert point.training_compensation_ratio == pytest.approx(expected[point.step])
        assert point.validation_compensation_ratio == pytest.approx(expected[point.step])


def test_stage1_diagnostic_restores_borrowed_policy_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    policy_snapshot = _snapshot(policy)
    gradients = [parameter.grad for parameter in policy.parameters()]
    requires_grad = [parameter.requires_grad for parameter in policy.parameters()]
    training_modes = [module.training for module in policy.modules()]

    def corrupt_then_fail(*args, **kwargs):
        del args, kwargs
        parameter = next(policy.parameters())
        parameter.data.add_(1.0)
        parameter.grad = torch.ones_like(parameter)
        parameter.requires_grad_(False)
        policy.train()
        raise RuntimeError("injected diagnostic failure")

    monkeypatch.setattr(stage1_owner, "_direction_stage_step", corrupt_then_fail)
    config_type = getattr(calibration_training, "DirectionDiagnosticConfig")
    diagnose = getattr(calibration_training, "diagnose_direction_stage_training")
    with pytest.raises(RuntimeError, match="injected diagnostic failure"):
        diagnose(
            policy,
            batch,
            _identity(),
            config_type(checkpoint_steps=(0, 1)),
        )
    _assert_snapshot(policy, policy_snapshot)
    assert [parameter.grad for parameter in policy.parameters()] == gradients
    assert [parameter.requires_grad for parameter in policy.parameters()] == requires_grad
    assert [module.training for module in policy.modules()] == training_modes


@pytest.mark.parametrize("checkpoint_steps", [(), (-1,), (0, 0), (1, 0)])
def test_stage1_diagnostic_config_rejects_invalid_steps_before_adam(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_steps: tuple[int, ...],
) -> None:
    calls = 0

    def poison(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("Adam must be unreachable")

    monkeypatch.setattr(torch.optim, "Adam", poison)
    config_type = getattr(calibration_training, "DirectionDiagnosticConfig")
    with pytest.raises(ValueError, match="checkpoint_steps"):
        config_type(checkpoint_steps=checkpoint_steps)
    assert calls == 0


def test_stage1_diagnostics_report_high_ratio_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)

    def poison(*args, **kwargs):
        del args, kwargs
        raise AssertionError("diagnostics must not publish a Stage artifact")

    monkeypatch.setattr(stage1_owner, "_atomic_torch_save", poison)
    monkeypatch.setattr(
        stage1_owner,
        "direction_stage_compensation_ratio",
        lambda *args, **kwargs: torch.tensor(0.8),
    )
    config_type = getattr(calibration_training, "DirectionDiagnosticConfig")
    diagnose = getattr(calibration_training, "diagnose_direction_stage_training")
    points = diagnose(
        policy,
        batch,
        _identity(),
        config_type(checkpoint_steps=(0, 1)),
    )
    assert all(point.validation_compensation_ratio == pytest.approx(0.8) for point in points)

    with pytest.raises(ValueError, match="compensation ratio"):
        run_direction_stage_training(
            policy,
            batch,
            tmp_path / "stage1.pt",
            _identity(),
            DirectionStageConfig(steps_per_axis=1),
        )
    assert not (tmp_path / "stage1.pt").exists()


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
    artifact = fit_scale_stage(readings, candidates, action_errors, _axis_spec())
    assert len(artifact) == 3
    assert all(curve.kind == "pchip" for curve in artifact)


def test_stage3_rejects_missing_32_repeat_evidence() -> None:
    readings = torch.linspace(-1.0, 1.0, 21).repeat(3, 1)
    with pytest.raises(ValueError, match="32 repetitions"):
        fit_scale_stage(
            readings,
            torch.linspace(-1.0, 1.0, 3),
            torch.zeros(3, 21, 3),
            _axis_spec(),
        )


def test_stage3_rejects_low_quality_monotone_fit() -> None:
    readings = torch.linspace(-1.0, 1.0, 21).view(1, 21, 1).repeat(3, 1, 32)
    candidates = torch.tensor([-1.0, 0.0, 1.0])
    desired = readings.sign()
    desired[:, :, ::2] *= -1
    action_errors = (candidates.view(1, 1, 1, -1) - desired.unsqueeze(-1)).square()
    with pytest.raises(ValueError, match=r"R\^2"):
        fit_scale_stage(readings, candidates, action_errors, _axis_spec())


def test_stage3_evidence_round_trip_binds_transaction_identity(tmp_path) -> None:
    evidence = _scale_evidence()
    path = save_calibration_scale_evidence(tmp_path / "scale-evidence.pt", evidence)
    restored = load_calibration_scale_evidence(path, expected_identity=_identity())
    torch.testing.assert_close(
        restored.coefficient_scan_grid,
        evidence.coefficient_scan_grid,
    )
    torch.testing.assert_close(restored.readings, evidence.readings)
    torch.testing.assert_close(restored.action_errors, evidence.action_errors)
    with pytest.raises(ValueError, match="metadata identity"):
        load_calibration_scale_evidence(
            path,
            expected_identity=replace(_identity(), dataset_sha256="4" * 64),
        )


def test_stage3_evidence_rejects_a_noncanonical_coefficient_scan_grid() -> None:
    evidence = _scale_evidence()
    bad_grid = evidence.coefficient_scan_grid.clone()
    bad_grid[1, 10] = 0.01
    with pytest.raises(ValueError, match="coefficient scan grid"):
        replace(evidence, coefficient_scan_grid=bad_grid).validate()


def test_scale_evidence_rejects_axis_count_in_provenance_metadata() -> None:
    evidence = _scale_evidence()
    with pytest.raises(ValueError, match="reserved axis identity"):
        replace(evidence, metadata={**evidence.metadata, "axis_count": "3"}).validate()


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "method_contract",
        "training_contract",
        "axis_order",
        "missing_grid",
        "missing_readings",
        "missing_candidates",
        "missing_errors",
        "nonfinite",
    ],
)
def test_scale_evidence_loader_rejects_invalid_persisted_envelope(
    tmp_path,
    mutation,
) -> None:
    path = save_calibration_scale_evidence(tmp_path / "evidence.pt", _scale_evidence())
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if mutation == "schema":
        payload["schema_version"] = "wrong-schema"
    elif mutation == "method_contract":
        payload["method_contract_id"] = "wrong-method"
    elif mutation == "training_contract":
        payload["training_contract_id"] = "wrong-training"
    elif mutation == "axis_order":
        payload["axis_spec"]["names"] = list(reversed(payload["axis_spec"]["names"]))
    elif mutation.startswith("missing_"):
        field = {
            "missing_grid": "coefficient_scan_grid",
            "missing_readings": "readings",
            "missing_candidates": "candidate_scales",
            "missing_errors": "action_errors",
        }[mutation]
        payload.pop(field)
    else:
        payload["readings"][0, 0, 0] = torch.nan
    torch.save(payload, path)
    with pytest.raises(ValueError):
        load_calibration_scale_evidence(path, expected_identity=_identity())


@pytest.mark.parametrize("preexisting", [False, True])
def test_scale_evidence_serialization_failure_cleans_temp_and_preserves_target(
    tmp_path,
    monkeypatch,
    preexisting,
) -> None:
    target = tmp_path / "evidence.pt"
    if preexisting:
        target.write_bytes(b"old-evidence")

    def fail_after_partial_write(value, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("injected scale evidence serialization failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="scale evidence serialization failure"):
        save_calibration_scale_evidence(target, _scale_evidence())
    assert list(tmp_path.glob(".evidence.pt.*.tmp")) == []
    if preexisting:
        assert target.read_bytes() == b"old-evidence"
    else:
        assert not target.exists()


def test_stage1_public_transaction_constructs_no_encoder_and_seals_only_directions(
    tmp_path,
    monkeypatch,
) -> None:
    torch.manual_seed(21)
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    policy_snapshot = _snapshot(policy)
    monkeypatch.setattr(
        stage1_owner,
        "direction_stage_compensation_ratio",
        lambda *args, **kwargs: torch.tensor(0.0),
    )

    def poison(*args, **kwargs):
        raise AssertionError("Stage 1 must not construct a Coefficient Encoder")

    assert not hasattr(stage1_owner, "CoefficientEncoder")
    result = run_direction_stage_training(
        policy,
        batch,
        tmp_path / "stage1.pt",
        _identity(),
        DirectionStageConfig(steps_per_axis=1),
    )

    assert result.stage == "direction_frozen"
    assert result.artifact_sha256 == _sha256(result.artifact_path)
    payload = torch.load(result.artifact_path, map_location="cpu", weights_only=True)
    assert payload["schema_version"] == CALIBRATION_STAGE_ARTIFACT_SCHEMA
    assert payload["stage"] == "direction_frozen"
    assert set(payload["owners"]) == {"direction_bank"}
    assert "coefficient_encoder" not in payload["owners"]
    directions = payload["owners"]["direction_bank"]["state_dict"]["directions"]
    torch.testing.assert_close(directions.flatten(1).norm(dim=1), torch.ones(3))
    _assert_snapshot(policy, policy_snapshot)


def test_stage2_strictly_loads_stage1_and_seals_parent_identity(
    tmp_path,
    monkeypatch,
) -> None:
    torch.manual_seed(22)
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    stage1_path = _make_direction_artifact(tmp_path, monkeypatch, policy, batch)
    stage1_payload = torch.load(stage1_path, map_location="cpu", weights_only=True)
    direction_snapshot = {
        name: value.clone()
        for name, value in stage1_payload["owners"]["direction_bank"]["state_dict"].items()
    }
    policy_snapshot = _snapshot(policy)
    monkeypatch.setattr(
        stage2_owner,
        "coefficient_validation_error",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    result = run_coefficient_stage_training(
        policy,
        batch,
        stage1_path,
        tmp_path / "stage2.pt",
        _identity(),
        CoefficientStageConfig(steps=1),
    )

    assert result.stage == "coefficient_frozen"
    assert result.parent_stage_sha256 == _sha256(stage1_path)
    payload = torch.load(result.artifact_path, map_location="cpu", weights_only=True)
    assert payload["stage"] == "coefficient_frozen"
    assert set(payload["owners"]) == {"direction_bank", "coefficient_encoder"}
    for name, value in direction_snapshot.items():
        torch.testing.assert_close(
            payload["owners"]["direction_bank"]["state_dict"][name],
            value,
            rtol=0.0,
            atol=0.0,
        )
    _assert_snapshot(policy, policy_snapshot)


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "method_contract",
        "training_contract",
        "stage",
        "source",
        "dataset",
        "split",
        "catalog",
        "architecture",
        "history_dimension",
        "horizon_dimension",
        "latent_dimension",
        "axis_order",
        "direction_config",
        "owner_set",
        "direction_shape",
        "direction_norm",
        "normalization_zero",
        "normalization_nan",
        "gate_threshold",
        "gate_nan",
        "nan",
    ],
)
def test_stage2_rejects_invalid_predecessor_before_optimizer(
    tmp_path,
    monkeypatch,
    mutation,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    stage1_path = _make_direction_artifact(tmp_path, monkeypatch, policy, batch)
    payload = torch.load(stage1_path, map_location="cpu", weights_only=True)
    if mutation == "schema":
        payload["schema_version"] = "unilab_fada_calibration_checkpoint_v1"
    elif mutation == "method_contract":
        payload["method_contract_id"] = "wrong-method"
    elif mutation == "training_contract":
        payload["training_contract_id"] = "wrong-training"
    elif mutation == "stage":
        payload["stage"] = "coefficient_frozen"
    elif mutation in {"source", "dataset", "split"}:
        identity_field = {
            "source": "source_tracker_sha256",
            "dataset": "dataset_sha256",
            "split": "split_sha256",
        }[mutation]
        payload["identity"][identity_field] = "other"
    elif mutation == "catalog":
        payload["identity"]["axis_spec"]["catalog_version"] = "other"
    elif mutation == "architecture":
        payload["architecture"]["hidden_dim"] = 9
    elif mutation in {"history_dimension", "horizon_dimension", "latent_dimension"}:
        dimension = {
            "history_dimension": "history_length",
            "horizon_dimension": "prediction_horizon",
            "latent_dimension": "latent_dim",
        }[mutation]
        payload["dimensions"][dimension] += 1
    elif mutation == "axis_order":
        payload["identity"]["axis_spec"]["names"] = list(
            reversed(payload["identity"]["axis_spec"]["names"])
        )
    elif mutation == "direction_config":
        payload["owners"]["direction_bank"]["config"]["axis_count"] = 4
    elif mutation == "owner_set":
        payload["owners"]["coefficient_encoder"] = {}
    elif mutation == "direction_shape":
        state = payload["owners"]["direction_bank"]["state_dict"]
        state["directions"] = state["directions"][:2]
    elif mutation == "direction_norm":
        state = payload["owners"]["direction_bank"]["state_dict"]
        state["directions"][0].mul_(2.0)
    elif mutation == "normalization_zero":
        state = payload["owners"]["direction_bank"]["state_dict"]
        state["normalization_scale"][0] = 0.0
    elif mutation == "normalization_nan":
        state = payload["owners"]["direction_bank"]["state_dict"]
        state["normalization_scale"][0] = torch.nan
    elif mutation == "gate_threshold":
        payload["gate"]["threshold"] = 0.2
    elif mutation == "gate_nan":
        payload["gate"]["result"][0] = float("nan")
    else:
        payload["owners"]["direction_bank"]["state_dict"]["directions"][0, 0, 0] = torch.nan
    torch.save(payload, stage1_path)
    calls = 0

    def poison(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("optimizer construction must be unreachable")

    monkeypatch.setattr(torch.optim, "Adam", poison)
    with pytest.raises((ValueError, RuntimeError, EOFError)):
        run_coefficient_stage_training(
            policy,
            batch,
            stage1_path,
            tmp_path / "stage2.pt",
            _identity(),
            CoefficientStageConfig(steps=1),
        )
    assert calls == 0
    assert not (tmp_path / "stage2.pt").exists()


def test_stage2_binds_exact_copied_parent_bytes_and_rejects_corruption_before_optimizer(
    tmp_path,
    monkeypatch,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    original = _make_direction_artifact(tmp_path, monkeypatch, policy, batch)
    copied = tmp_path / "copied-stage1.pt"
    copied.write_bytes(original.read_bytes())
    monkeypatch.setattr(
        stage2_owner,
        "coefficient_validation_error",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    result = run_coefficient_stage_training(
        policy,
        batch,
        copied,
        tmp_path / "stage2.pt",
        _identity(),
        CoefficientStageConfig(steps=1),
    )
    assert result.parent_stage_sha256 == _sha256(original) == _sha256(copied)

    valid_payload = torch.load(copied, map_location="cpu", weights_only=True)
    valid_payload["gate"]["result"][0] = 0.01
    torch.save(valid_payload, copied)
    changed_digest = _sha256(copied)
    assert changed_digest != result.parent_stage_sha256
    rebound = run_coefficient_stage_training(
        policy,
        batch,
        copied,
        tmp_path / "rebound-stage2.pt",
        _identity(),
        CoefficientStageConfig(steps=1),
    )
    assert rebound.parent_stage_sha256 == changed_digest

    raw = bytearray(copied.read_bytes())
    schema = CALIBRATION_STAGE_ARTIFACT_SCHEMA.encode()
    schema_offset = raw.find(schema)
    assert schema_offset >= 0
    raw[schema_offset] ^= 1
    copied.write_bytes(raw)
    monkeypatch.setattr(
        torch.optim,
        "Adam",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("optimizer construction must be unreachable")
        ),
    )
    with pytest.raises((ValueError, RuntimeError, EOFError)):
        run_coefficient_stage_training(
            policy,
            batch,
            copied,
            tmp_path / "rejected.pt",
            _identity(),
            CoefficientStageConfig(steps=1),
        )
    assert not (tmp_path / "rejected.pt").exists()


@pytest.mark.parametrize("stage", ["direction", "coefficient"])
def test_training_stage_rolls_back_injected_borrowed_policy_mutation_without_publication(
    tmp_path,
    monkeypatch,
    stage,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    policy_snapshot = _snapshot(policy)
    monkeypatch.setattr(
        stage1_owner,
        "direction_stage_compensation_ratio",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    if stage == "coefficient":
        predecessor = _make_direction_artifact(tmp_path, monkeypatch, policy, batch)
        monkeypatch.setattr(
            stage2_owner,
            "coefficient_validation_error",
            lambda *args, **kwargs: torch.tensor(0.0),
        )
        policy_snapshot = _snapshot(policy)
    real_adam = torch.optim.Adam

    def mutating_adam(*args, **kwargs):
        optimizer = real_adam(*args, **kwargs)
        real_step = optimizer.step

        def mutating_step(*step_args, **step_kwargs):
            result = real_step(*step_args, **step_kwargs)
            with torch.no_grad():
                next(policy.parameters()).add_(1.0)
            return result

        optimizer.step = mutating_step
        return optimizer

    monkeypatch.setattr(torch.optim, "Adam", mutating_adam)
    target = tmp_path / f"{stage}-rejected.pt"
    with pytest.raises(ValueError, match="frozen owner mutated"):
        if stage == "direction":
            run_direction_stage_training(
                policy,
                batch,
                target,
                _identity(),
                DirectionStageConfig(steps_per_axis=1),
            )
        else:
            run_coefficient_stage_training(
                policy,
                batch,
                predecessor,
                target,
                _identity(),
                CoefficientStageConfig(steps=1),
            )
    _assert_snapshot(policy, policy_snapshot)
    assert not target.exists()


def test_stage3_loads_typed_paths_constructs_no_optimizer_and_binds_exact_digests(
    tmp_path,
    monkeypatch,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    stage2_path = _make_coefficient_artifact(tmp_path, monkeypatch, policy, batch)
    policy_snapshot = _snapshot(policy)
    stage2_bytes = stage2_path.read_bytes()
    stage2_payload = torch.load(stage2_path, map_location="cpu", weights_only=True)
    evidence_path = save_calibration_scale_evidence(tmp_path / "evidence.pt", _scale_evidence())
    monkeypatch.setattr(
        torch.optim,
        "Adam",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Stage 3 must not construct an optimizer")
        ),
    )
    result = run_scale_stage_fitting(
        policy,
        stage2_path,
        evidence_path,
        tmp_path / "calibration.pt",
        _identity(),
    )
    assert result.stage == "complete"
    assert result.parent_stage_sha256 == _sha256(stage2_path)
    assert result.scale_evidence_sha256 == _sha256(evidence_path)
    artifact = torch.load(result.artifact_path, map_location="cpu", weights_only=True)
    assert artifact["metadata"]["parent_stage_sha256"] == result.parent_stage_sha256
    assert artifact["metadata"]["scale_evidence_sha256"] == result.scale_evidence_sha256
    loaded = load_calibration_artifact(result.artifact_path, FaultAxisCatalog.default())
    for owner_name in ("direction_bank", "coefficient_encoder"):
        stage_owner = stage2_payload["owners"][owner_name]["state_dict"]
        deployment_owner = loaded[owner_name]
        for name, value in stage_owner.items():
            torch.testing.assert_close(deployment_owner[name], value, rtol=0.0, atol=0.0)
    _assert_snapshot(policy, policy_snapshot)
    assert stage2_path.read_bytes() == stage2_bytes


def test_stage3_rejects_stage1_without_publication(
    tmp_path,
    monkeypatch,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    stage1_path = _make_direction_artifact(tmp_path, monkeypatch, policy, batch)
    evidence_path = save_calibration_scale_evidence(tmp_path / "evidence.pt", _scale_evidence())
    with pytest.raises(ValueError):
        run_scale_stage_fitting(
            policy,
            stage1_path,
            evidence_path,
            tmp_path / "calibration.pt",
            _identity(),
        )
    assert not (tmp_path / "calibration.pt").exists()


def test_stage3_rejects_wrong_evidence_and_nonfinite_encoder_without_publication(
    tmp_path,
    monkeypatch,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    stage2_path = _make_coefficient_artifact(tmp_path, monkeypatch, policy, batch)
    wrong_evidence = replace(
        _scale_evidence(),
        metadata={**_metadata(), "split_sha256": "other"},
    )
    wrong_path = save_calibration_scale_evidence(tmp_path / "wrong-evidence.pt", wrong_evidence)
    with pytest.raises(ValueError, match="metadata identity"):
        run_scale_stage_fitting(
            policy,
            stage2_path,
            wrong_path,
            tmp_path / "wrong-evidence-output.pt",
            _identity(),
        )
    assert not (tmp_path / "wrong-evidence-output.pt").exists()

    payload = torch.load(stage2_path, map_location="cpu", weights_only=True)
    encoder_state = payload["owners"]["coefficient_encoder"]["state_dict"]
    first_tensor = next(iter(encoder_state.values()))
    first_tensor.flatten()[0] = torch.nan
    torch.save(payload, stage2_path)
    valid_path = save_calibration_scale_evidence(tmp_path / "valid-evidence.pt", _scale_evidence())
    with pytest.raises(ValueError, match="finite"):
        run_scale_stage_fitting(
            policy,
            stage2_path,
            valid_path,
            tmp_path / "nonfinite-output.pt",
            _identity(),
        )
    assert not (tmp_path / "nonfinite-output.pt").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "parent_missing",
        "parent_short",
        "parent_nonhex",
        "owner_set",
        "encoder_config",
        "encoder_shape",
        "gate_threshold",
        "gate_result",
        "gate_nan",
        "direction_shape",
        "direction_norm",
    ],
)
def test_stage3_rejects_invalid_coefficient_artifact_without_optimizer_or_publication(
    tmp_path,
    monkeypatch,
    mutation,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    stage2_path = _make_coefficient_artifact(tmp_path, monkeypatch, policy, batch)
    payload = torch.load(stage2_path, map_location="cpu", weights_only=True)
    if mutation == "parent_missing":
        payload.pop("parent_stage_sha256")
    elif mutation == "parent_short":
        payload["parent_stage_sha256"] = "a" * 63
    elif mutation == "parent_nonhex":
        payload["parent_stage_sha256"] = "g" * 64
    elif mutation == "owner_set":
        payload["owners"]["unexpected"] = {}
    elif mutation == "encoder_config":
        payload["owners"]["coefficient_encoder"]["config"]["hidden_dim"] = 64
    elif mutation == "encoder_shape":
        state = payload["owners"]["coefficient_encoder"]["state_dict"]
        state["state_embedding.weight"] = state["state_embedding.weight"][:1]
    elif mutation == "gate_threshold":
        payload["gate"]["threshold"] = 0.1
    elif mutation == "gate_result":
        payload["gate"]["result"] = 0.051
    elif mutation == "gate_nan":
        payload["gate"]["result"] = float("nan")
    elif mutation == "direction_shape":
        state = payload["owners"]["direction_bank"]["state_dict"]
        state["directions"] = state["directions"][:2]
    else:
        state = payload["owners"]["direction_bank"]["state_dict"]
        state["directions"][0].mul_(2.0)
    torch.save(payload, stage2_path)
    evidence_path = save_calibration_scale_evidence(tmp_path / "evidence.pt", _scale_evidence())
    calls = 0

    def poison(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("Stage 3 optimizer construction must be unreachable")

    monkeypatch.setattr(torch.optim, "Adam", poison)
    output = tmp_path / "calibration.pt"
    with pytest.raises((ValueError, RuntimeError)):
        run_scale_stage_fitting(
            policy,
            stage2_path,
            evidence_path,
            output,
            _identity(),
        )
    assert calls == 0
    assert not output.exists()


def test_stage3_binds_new_digest_after_typed_evidence_bytes_change(
    tmp_path,
    monkeypatch,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    stage2_path = _make_coefficient_artifact(tmp_path, monkeypatch, policy, batch)
    evidence = _scale_evidence()
    evidence_path = save_calibration_scale_evidence(tmp_path / "evidence.pt", evidence)
    first = run_scale_stage_fitting(
        policy,
        stage2_path,
        evidence_path,
        tmp_path / "first.pt",
        _identity(),
    )
    changed = replace(evidence, action_errors=evidence.action_errors + 1.0e-6)
    save_calibration_scale_evidence(evidence_path, changed)
    second = run_scale_stage_fitting(
        policy,
        stage2_path,
        evidence_path,
        tmp_path / "second.pt",
        _identity(),
    )
    assert second.scale_evidence_sha256 == _sha256(evidence_path)
    assert second.scale_evidence_sha256 != first.scale_evidence_sha256


@pytest.mark.parametrize("preexisting", [False, True])
def test_stage3_failed_publication_preserves_old_target_and_cleans_staging(
    tmp_path,
    monkeypatch,
    preexisting,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    stage2_path = _make_coefficient_artifact(tmp_path, monkeypatch, policy, batch)
    evidence_path = save_calibration_scale_evidence(tmp_path / "evidence.pt", _scale_evidence())
    target = tmp_path / "calibration.pt"
    if preexisting:
        target.write_bytes(b"old-deployment-artifact")

    def fail_after_partial_write(path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("injected deployment serialization failure")

    monkeypatch.setattr(
        training_io_owner,
        "save_calibration_artifact",
        fail_after_partial_write,
    )
    with pytest.raises(OSError, match="deployment serialization failure"):
        run_scale_stage_fitting(
            policy,
            stage2_path,
            evidence_path,
            target,
            _identity(),
        )
    if preexisting:
        assert target.read_bytes() == b"old-deployment-artifact"
    else:
        assert not target.exists()
    assert list(tmp_path.glob(".calibration.pt.*.staging*")) == []


@pytest.mark.parametrize("preexisting", [False, True])
def test_stage_artifact_publication_cleans_unique_temp_and_preserves_target(
    tmp_path,
    monkeypatch,
    preexisting,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    target = tmp_path / "stage1.pt"
    if preexisting:
        target.write_bytes(b"old-target")
    monkeypatch.setattr(
        stage1_owner,
        "direction_stage_compensation_ratio",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    real_save = torch.save

    def fail_after_partial_write(value, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("injected serialization failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="injected serialization failure"):
        run_direction_stage_training(
            policy,
            batch,
            target,
            _identity(),
            DirectionStageConfig(steps_per_axis=1),
        )
    monkeypatch.setattr(torch, "save", real_save)
    assert list(tmp_path.glob(".stage1.pt.*.tmp")) == []
    if preexisting:
        assert target.read_bytes() == b"old-target"
    else:
        assert not target.exists()


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
            source_tracker_sha256=_SOURCE_SHA256,
            dataset_sha256=_DATASET_SHA256,
            split_sha256=_SPLIT_SHA256,
            axis_spec=_axis_spec(),
            scale_evidence=_scale_evidence(),
            config=SerialCalibrationConfig(stage1_steps_per_axis=1, stage2_steps=1),
        )
    assert calls == 0


def test_serial_training_does_not_admit_stage3_evidence_before_stage1_optimizer(
    tmp_path,
    monkeypatch,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
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
    with pytest.raises(AssertionError, match="optimizer construction"):
        run_serial_calibration_training(
            policy,
            batch,
            output_dir=tmp_path,
            source_tracker_sha256=_SOURCE_SHA256,
            dataset_sha256=_DATASET_SHA256,
            split_sha256=_SPLIT_SHA256,
            axis_spec=_axis_spec(),
            scale_evidence=wrong_evidence,
            config=SerialCalibrationConfig(stage1_steps_per_axis=1, stage2_steps=1),
        )
    assert calls == 1


@pytest.mark.parametrize("mutation", ["grid", "nonfinite"])
def test_serial_materializes_in_memory_future_evidence_only_after_stage2_publication(
    tmp_path,
    monkeypatch,
    mutation,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    evidence = _scale_evidence()
    if mutation == "grid":
        grid = evidence.coefficient_scan_grid.clone()
        grid[0, 0] = -0.9
        evidence = replace(evidence, coefficient_scan_grid=grid)
    else:
        readings = evidence.readings.clone()
        readings[0, 0, 0] = torch.nan
        evidence = replace(evidence, readings=readings)
    monkeypatch.setattr(
        stage1_owner,
        "direction_stage_compensation_ratio",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    monkeypatch.setattr(
        stage2_owner,
        "coefficient_validation_error",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    real_adam = torch.optim.Adam
    calls = 0

    def spy_adam(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_adam(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", spy_adam)
    with pytest.raises(ValueError):
        run_serial_calibration_training(
            policy,
            batch,
            output_dir=tmp_path,
            source_tracker_sha256=_SOURCE_SHA256,
            dataset_sha256=_DATASET_SHA256,
            split_sha256=_SPLIT_SHA256,
            axis_spec=_axis_spec(),
            scale_evidence=evidence,
            config=SerialCalibrationConfig(stage1_steps_per_axis=1, stage2_steps=1),
        )
    assert calls == 4
    assert (tmp_path / "stage1_direction_frozen.pt").is_file()
    assert (tmp_path / "stage2_coefficient_frozen.pt").is_file()
    assert not (tmp_path / "scale_evidence.pt").exists()
    assert not (tmp_path / "calibration_artifact.pt").exists()


def test_serial_does_not_read_scale_evidence_path_before_stage2_publication(
    tmp_path,
    monkeypatch,
) -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(policy)
    monkeypatch.setattr(
        stage1_owner,
        "direction_stage_compensation_ratio",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    monkeypatch.setattr(
        stage2_owner,
        "coefficient_validation_error",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    with pytest.raises(FileNotFoundError):
        run_serial_calibration_training(
            policy,
            batch,
            output_dir=tmp_path,
            source_tracker_sha256=_SOURCE_SHA256,
            dataset_sha256=_DATASET_SHA256,
            split_sha256=_SPLIT_SHA256,
            axis_spec=_axis_spec(),
            scale_evidence_path=tmp_path / "missing-evidence.pt",
            config=SerialCalibrationConfig(stage1_steps_per_axis=1, stage2_steps=1),
        )
    assert (tmp_path / "stage1_direction_frozen.pt").is_file()
    assert (tmp_path / "stage2_coefficient_frozen.pt").is_file()
    assert not (tmp_path / "calibration_artifact.pt").exists()


def _valid_source_projection(
    policy: FADAPlannerIDMPolicy,
) -> CalibrationRolloutBatch:
    config = _config()
    batch = _batch(config)
    with torch.no_grad():
        intent = policy.planner(batch.observation_history, batch.command)
        latent = policy.idm.encode_latent(
            batch.observation_history,
            batch.action_history,
            intent,
        )
        nominal = policy.idm.decode_latent(latent)
    return replace(batch, planner_intent=intent, nominal_action_chunk=nominal)


def test_source_projection_accepts_observed_cross_device_float32_drift() -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    valid = _valid_source_projection(policy)
    validate_calibration_source_projection(
        policy,
        replace(
            valid,
            planner_intent=valid.planner_intent + 6.5e-4,
            nominal_action_chunk=valid.nominal_action_chunk + 4.0e-4,
        ),
    )


def test_source_projection_rejects_material_planner_drift() -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    valid = _valid_source_projection(policy)
    with pytest.raises(ValueError, match="Planner Intent"):
        validate_calibration_source_projection(
            policy,
            replace(valid, planner_intent=valid.planner_intent + 1.0e-2),
        )


def test_source_projection_rejects_dataset_from_a_different_tracker() -> None:
    policy = FADAPlannerIDMPolicy(_config()).eval()
    valid = _valid_source_projection(policy)
    validate_calibration_source_projection(policy, valid)
    with pytest.raises(ValueError, match="nominal Action"):
        validate_calibration_source_projection(
            policy,
            replace(valid, nominal_action_chunk=valid.nominal_action_chunk + 0.1),
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
        source_tracker_sha256=_SOURCE_SHA256,
        dataset_sha256=_DATASET_SHA256,
        split_sha256=_SPLIT_SHA256,
        axis_spec=_axis_spec(),
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
    complete = torch.load(
        tmp_path / "calibration_artifact.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert len(complete["scale_curves"]) == 3
    assert complete["metadata"]["parent_stage_sha256"] == _sha256(
        tmp_path / "stage2_coefficient_frozen.pt"
    )
    assert complete["metadata"]["scale_evidence_sha256"] == _sha256(tmp_path / "scale_evidence.pt")
    stage1 = torch.load(
        tmp_path / "stage1_direction_frozen.pt",
        map_location="cpu",
        weights_only=True,
    )
    stage2 = torch.load(
        tmp_path / "stage2_coefficient_frozen.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert stage1["stage"] == "direction_frozen"
    assert stage2["stage"] == "coefficient_frozen"


def test_serial_and_independent_transactions_are_fresh_reload_action_equivalent(
    tmp_path,
    monkeypatch,
) -> None:
    """Connectivity compatibility only; this is not formal official-route evidence."""

    torch.manual_seed(311)
    source_policy = FADAPlannerIDMPolicy(_config()).eval()
    batch = _admitted_batch(source_policy)
    source_state = _snapshot(source_policy)
    serial_policy = FADAPlannerIDMPolicy(_config()).eval()
    serial_policy.load_state_dict(source_state, strict=True)
    independent_policy = FADAPlannerIDMPolicy(_config()).eval()
    independent_policy.load_state_dict(source_state, strict=True)
    monkeypatch.setattr(
        stage1_owner,
        "direction_stage_compensation_ratio",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    monkeypatch.setattr(
        stage2_owner,
        "coefficient_validation_error",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    evidence = _scale_evidence()
    serial_dir = tmp_path / "serial"
    independent_dir = tmp_path / "independent"
    independent_dir.mkdir()

    torch.manual_seed(701)
    serial = run_serial_calibration_training(
        serial_policy,
        batch,
        output_dir=serial_dir,
        source_tracker_sha256=_SOURCE_SHA256,
        dataset_sha256=_DATASET_SHA256,
        split_sha256=_SPLIT_SHA256,
        axis_spec=_axis_spec(),
        scale_evidence=evidence,
        config=SerialCalibrationConfig(stage1_steps_per_axis=1, stage2_steps=1),
    )

    torch.manual_seed(701)
    direction = run_direction_stage_training(
        independent_policy,
        batch,
        independent_dir / "stage1.pt",
        _identity(),
        DirectionStageConfig(steps_per_axis=1),
    )
    coefficient = run_coefficient_stage_training(
        independent_policy,
        batch,
        direction.artifact_path,
        independent_dir / "stage2.pt",
        _identity(),
        CoefficientStageConfig(steps=1),
    )
    evidence_path = save_calibration_scale_evidence(
        independent_dir / "evidence.pt",
        evidence,
    )
    complete = run_scale_stage_fitting(
        independent_policy,
        coefficient.artifact_path,
        evidence_path,
        independent_dir / "calibration.pt",
        _identity(),
    )

    serial_stage1 = torch.load(
        serial_dir / "stage1_direction_frozen.pt",
        map_location="cpu",
        weights_only=True,
    )
    independent_stage1 = torch.load(direction.artifact_path, map_location="cpu", weights_only=True)
    serial_stage2 = torch.load(
        serial_dir / "stage2_coefficient_frozen.pt",
        map_location="cpu",
        weights_only=True,
    )
    independent_stage2 = torch.load(
        coefficient.artifact_path,
        map_location="cpu",
        weights_only=True,
    )
    for serial_stage, independent_stage in (
        (serial_stage1, independent_stage1),
        (serial_stage2, independent_stage2),
    ):
        for field in (
            "schema_version",
            "method_contract_id",
            "training_contract_id",
            "stage",
            "architecture",
            "dimensions",
            "identity",
            "gate",
        ):
            assert serial_stage[field] == independent_stage[field]
    for owner_name in ("direction_bank", "coefficient_encoder"):
        serial_owner = serial_stage2["owners"][owner_name]["state_dict"]
        independent_owner = independent_stage2["owners"][owner_name]["state_dict"]
        for name, value in serial_owner.items():
            torch.testing.assert_close(value, independent_owner[name], rtol=0.0, atol=0.0)
    assert serial_stage2["parent_stage_sha256"] == _sha256(
        serial_dir / "stage1_direction_frozen.pt"
    )
    assert independent_stage2["parent_stage_sha256"] == _sha256(direction.artifact_path)
    serial_artifact = load_calibration_artifact(serial["artifact_path"], FaultAxisCatalog.default())
    independent_artifact = load_calibration_artifact(
        complete.artifact_path, FaultAxisCatalog.default()
    )
    assert serial_artifact["metadata"]["parent_stage_sha256"] == _sha256(
        serial_dir / "stage2_coefficient_frozen.pt"
    )
    assert serial_artifact["metadata"]["scale_evidence_sha256"] == _sha256(
        serial_dir / "scale_evidence.pt"
    )
    assert independent_artifact["metadata"]["parent_stage_sha256"] == _sha256(
        coefficient.artifact_path
    )
    assert independent_artifact["metadata"]["scale_evidence_sha256"] == _sha256(evidence_path)

    serial_healthy = FADAPlannerIDMPolicy(_config()).eval()
    serial_healthy.load_state_dict(source_state, strict=True)
    independent_healthy = FADAPlannerIDMPolicy(_config()).eval()
    independent_healthy.load_state_dict(source_state, strict=True)
    serial_loaded = load_calibrated_policy(
        serial_healthy,
        serial["artifact_path"],
        expected_metadata=_metadata(),
        catalog=FaultAxisCatalog.default(),
    )
    independent_loaded = load_calibrated_policy(
        independent_healthy,
        complete.artifact_path,
        expected_metadata=_metadata(),
        catalog=FaultAxisCatalog.default(),
    )
    observation = batch.observation_history[:2].clone()
    action_history = batch.action_history[:2].clone()
    command = batch.command[:2].clone()
    observation[1, -1, 0] += 0.25
    action_history[0, -2, 1] -= 0.15
    command[1, 0] += 0.1
    with torch.no_grad():
        serial_output = serial_loaded(observation, action_history, command)
        independent_output = independent_loaded(observation, action_history, command)
    torch.testing.assert_close(
        serial_output.action_chunk,
        independent_output.action_chunk,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(serial_output.action, serial_output.action_chunk[:, 0])
    torch.testing.assert_close(independent_output.action, independent_output.action_chunk[:, 0])
    torch.testing.assert_close(serial_output.action, independent_output.action, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("names", [("gain",), ("offset", "gain")])
def test_configurable_axis_spec_runs_the_complete_serial_owner_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    names: tuple[str, ...],
) -> None:
    torch.manual_seed(401)
    policy = FADAPlannerIDMPolicy(_config()).eval()
    axis_spec = _axis_spec(names)
    source_batch = _admitted_batch(policy)
    if names == ("offset", "gain"):
        source_batch = _append_held_out_combinations(source_batch)
    batch = project_calibration_rollout_batch(
        source_batch,
        FaultAxisCatalog.default(),
        axis_spec,
        config=policy.config,
    )
    if names == ("offset", "gain"):
        held_out = torch.nonzero(batch.is_held_out_combination, as_tuple=False).flatten()
        assert held_out.numel() == 1
        assert int(batch.axis_id[held_out[0]]) == -1
        torch.testing.assert_close(
            batch.c_true[held_out[0]],
            torch.tensor([0.4, 0.2]),
        )
    monkeypatch.setattr(
        stage1_owner,
        "direction_stage_compensation_ratio",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    monkeypatch.setattr(
        stage2_owner,
        "coefficient_validation_error",
        lambda *args, **kwargs: torch.tensor(0.0),
    )
    result = run_serial_calibration_training(
        policy,
        batch,
        output_dir=tmp_path,
        source_tracker_sha256=_SOURCE_SHA256,
        dataset_sha256=_DATASET_SHA256,
        split_sha256=_SPLIT_SHA256,
        axis_spec=axis_spec,
        scale_evidence=_scale_evidence(names),
        config=SerialCalibrationConfig(stage1_steps_per_axis=1, stage2_steps=1),
    )
    assert result["axis_spec"] == axis_spec.to_payload()
    stage1 = torch.load(
        tmp_path / "stage1_direction_frozen.pt",
        map_location="cpu",
        weights_only=True,
    )
    stage2 = torch.load(
        tmp_path / "stage2_coefficient_frozen.pt",
        map_location="cpu",
        weights_only=True,
    )
    final = load_calibration_artifact(result["artifact_path"], FaultAxisCatalog.default())
    assert stage1["identity"]["axis_spec"] == axis_spec.to_payload()
    assert tuple(stage1["owners"]["direction_bank"]["state_dict"]["directions"].shape) == (
        axis_spec.axis_count,
        policy.config.prediction_horizon,
        policy.config.hidden_dim,
    )
    assert stage2["owners"]["coefficient_encoder"]["config"]["axis_count"] == (axis_spec.axis_count)
    assert final["axis_spec"] == axis_spec.to_payload()
    assert len(final["scale_curves"]) == axis_spec.axis_count
    fresh_healthy = FADAPlannerIDMPolicy(_config()).eval()
    fresh_healthy.load_state_dict(policy.state_dict(), strict=True)
    if names == ("offset", "gain"):
        with pytest.raises(ValueError, match="expected dataset"):
            load_calibrated_policy(
                fresh_healthy,
                result["artifact_path"],
                expected_metadata=_metadata(),
                catalog=FaultAxisCatalog.default(),
                expected_axis_spec=_axis_spec(("gain", "offset")),
            )
    loaded = load_calibrated_policy(
        fresh_healthy,
        result["artifact_path"],
        expected_metadata=_metadata(),
        catalog=FaultAxisCatalog.default(),
        expected_axis_spec=axis_spec,
    )
    observation = batch.observation_history[:2]
    action_history = batch.action_history[:2]
    command = batch.command[:2]
    with torch.no_grad():
        nominal = fresh_healthy(observation, action_history, command)
        reconstructed = loaded.reconstruct_with_coefficients(
            observation,
            action_history,
            command,
            torch.zeros(2, axis_spec.axis_count),
        )
    torch.testing.assert_close(reconstructed.action_chunk, nominal.action_chunk)
    torch.testing.assert_close(reconstructed.action, nominal.action)
