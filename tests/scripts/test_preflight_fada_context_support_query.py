from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from scripts import preflight_fada_context_support_query as preflight

from unilab.algos.torch.distill.fada import FADAArchitectureConfig, FADAPlannerIDMPolicy
from unilab.algos.torch.fada_context.support_query import (
    ContextQueryBatch,
    SupportContextBatch,
    SupportQueryBatch,
    SupportQueryContextConfig,
)
from unilab.algos.torch.fada_context.support_query_data import (
    save_support_query_dataset,
    split_support_query_by_rollout,
    support_query_split_identity_sha256,
)
from unilab.algos.torch.fada_context.support_query_runtime import sha256_file
from unilab.algos.torch.fada_context.support_query_training import (
    prepare_support_query_training,
    save_context_support_query_checkpoint,
)


def _config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=4,
        action_dim=2,
        command_dim=3,
        history_length=3,
        prediction_horizon=6,
        hidden_dim=8,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=16,
    )


def _batch(config: FADAArchitectureConfig) -> SupportQueryBatch:
    pairs, support_length, windows = 2, 4, 2
    command = torch.tensor([[0.4, 0.0, 0.0]]).expand(pairs, -1).clone()
    return SupportQueryBatch(
        support=SupportContextBatch(
            target_future=torch.randn(
                pairs,
                support_length,
                config.prediction_horizon,
                config.obs_dim,
            ),
            realized_state=torch.randn(pairs, support_length, config.obs_dim),
            executed_action=torch.randn(pairs, support_length, config.action_dim),
        ),
        query=ContextQueryBatch(
            observation_history=torch.randn(
                pairs,
                windows,
                config.history_length,
                config.obs_dim,
            ),
            action_history=torch.randn(
                pairs,
                windows,
                config.history_length,
                config.action_dim,
            ),
            command=command[:, None].expand(-1, windows, -1).clone(),
            planner_intent=torch.randn(
                pairs,
                windows,
                config.prediction_horizon,
                config.obs_dim,
            ),
            realized_future=torch.randn(
                pairs,
                windows,
                config.prediction_horizon,
                config.obs_dim,
            ),
            executed_action=torch.randn(pairs, windows, config.action_dim),
            window_anchor=torch.tensor([[2, 3], [2, 3]], dtype=torch.int64),
            valid_window_mask=torch.tensor([[True, True], [True, False]]),
        ),
        support_command=command,
        pair_id=torch.tensor([10, 20], dtype=torch.int64),
        support_rollout_id=torch.tensor([100, 200], dtype=torch.int64),
        query_rollout_id=torch.tensor([101, 201], dtype=torch.int64),
    ).validate(config, support_length=support_length)


def test_preflight_parser_exposes_explicit_artifact_admission_mode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "preflight_fada_context_support_query.py",
            "--artifact-admission",
            "--dataset",
            "dataset.pt",
            "--context-checkpoint",
            "context.pt",
        ],
    )

    args = preflight._parse_args()

    assert args.artifact_admission is True
    assert args.dataset == Path("dataset.pt")
    assert args.context_checkpoint == Path("context.pt")


def test_preflight_reports_v006_query_provenance_and_per_query_delta(
    tmp_path,
    monkeypatch,
) -> None:
    config = _config()
    batch = _batch(config)
    healthy = FADAPlannerIDMPolicy(config)
    checkpoint = tmp_path / "healthy.pt"
    checkpoint.write_bytes(b"healthy")
    artifact = tmp_path / "support-query.pt"
    cfg = SimpleNamespace(
        seed=7,
        device="cpu",
        checkpoint_path=str(checkpoint),
        task_config="fault",
        collection=SimpleNamespace(
            artifact_path=str(artifact),
            support_length=4,
            query_length=6,
        ),
        context=SimpleNamespace(
            hidden_dim=5,
            num_layers=1,
            delta_scale=0.1,
            learning_rate=3.0e-4,
        ),
        training=SimpleNamespace(minimum_zero_context_mse=0.0),
    )
    collected = SimpleNamespace(
        batch=batch,
        accepted_pairs=2,
        rejected_pairs=0,
        reset_pairs=1,
    )
    monkeypatch.setattr(preflight, "load_support_query_config", lambda *_args, **_kwargs: cfg)
    monkeypatch.setattr(preflight, "apply_training_seed", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        preflight,
        "load_fada_policy_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(policy=healthy),
    )
    monkeypatch.setattr(
        preflight, "collect_fixed_fault_support_query", lambda *_args, **_kwargs: collected
    )
    monkeypatch.setattr(preflight, "save_support_query_dataset", lambda *_args, **_kwargs: artifact)
    monkeypatch.setattr(
        preflight,
        "load_support_query_dataset",
        lambda *_args, **_kwargs: (batch, {"fault_strength": 0.7}),
    )

    report = preflight.run_preflight(
        Namespace(config=tmp_path / "config.yaml", output=None, overrides=[])
    )

    assert report["method_contract_id"] == "FADA-CONTEXT-METHOD-v006"
    assert report["query_provenance"] == {
        "pair_ids": [10, 20],
        "support_rollout_ids": [100, 200],
        "query_rollout_ids": [101, 201],
        "current_history_conditioned": True,
    }
    assert report["tensors"]["delta_z"] == [2, 2, healthy.config.hidden_dim]


def _persist_artifact_admission_fixture(
    tmp_path: Path,
) -> tuple[Namespace, FADAPlannerIDMPolicy, Path]:
    config = _config()
    batch = _batch(config)
    healthy = FADAPlannerIDMPolicy(config)
    source_checkpoint = tmp_path / "healthy.pt"
    source_checkpoint.write_bytes(b"healthy-checkpoint")
    source_sha = sha256_file(source_checkpoint)
    dataset_path = save_support_query_dataset(
        tmp_path / "support-query.pt",
        batch,
        config,
        support_length=4,
        query_length=6,
        metadata={
            "source_checkpoint_sha256": source_sha,
            "task_config": "fault-070",
            "fault_joint": "left_knee",
            "fault_strength": 0.7,
            "command": [0.4, 0.0, 0.0],
            "seed": 17,
        },
    )
    train, validation = split_support_query_by_rollout(
        batch,
        validation_fraction=0.34,
        seed=17,
    )
    context_config = SupportQueryContextConfig(
        support_length=4,
        context_hidden_dim=5,
        context_layers=1,
        delta_scale=0.1,
    )
    context_checkpoint = save_context_support_query_checkpoint(
        tmp_path / "context.pt",
        prepare_support_query_training(healthy, context_config, learning_rate=3.0e-4),
        source_checkpoint_sha256=source_sha,
        dataset_sha256=sha256_file(dataset_path),
        train_split_sha256=support_query_split_identity_sha256(train),
        validation_split_sha256=support_query_split_identity_sha256(validation),
        step=3,
        split_seed=17,
        metrics={"validation_mse": 0.25},
        resolved_config={},
    )
    args = Namespace(
        config=preflight.ROOT_DIR / "conf" / "fada_context" / "support_query_left_knee_070.yaml",
        output=None,
        artifact_admission=True,
        dataset=dataset_path,
        context_checkpoint=context_checkpoint,
        overrides=[
            f"checkpoint_path={source_checkpoint}",
            "device=cpu",
            "collection.support_length=4",
            "collection.query_length=6",
            "context.hidden_dim=5",
            "context.num_layers=1",
            "training.validation_fraction=0.34",
        ],
    )
    return args, healthy, context_checkpoint


def test_artifact_admission_entrypoint_uses_real_persistence_and_typed_owner(
    tmp_path,
    monkeypatch,
) -> None:
    args, healthy, _ = _persist_artifact_admission_fixture(tmp_path)
    monkeypatch.setattr(
        preflight,
        "load_fada_policy_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(policy=healthy),
    )

    report = preflight.run_preflight(args)

    assert report["schema"] == "unilab_fada_context_support_query_artifact_admission_v1"
    assert report["mode"] == "artifact_admission"
    assert report["method_contract_id"] == "FADA-CONTEXT-METHOD-v006"
    assert report["checkpoint_schema"] == 4
    assert report["checkpoint_step"] == 3
    assert report["query_provenance"] == {
        "pair_ids": [10, 20],
        "support_rollout_ids": [100, 200],
        "query_rollout_ids": [101, 201],
        "current_history_conditioned": True,
    }
    assert report["tensors"]["delta_z"] == [2, 2, healthy.config.hidden_dim]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 1, "historical fixed-residual checkpoint schema"),
        ("schema_version", 2, "historical fixed-residual checkpoint schema"),
        ("schema_version", 3, "historical fixed-residual checkpoint schema"),
        ("method_contract_id", "FADA-CONTEXT-METHOD-v005", "method Contract mismatch"),
    ],
)
def test_artifact_admission_entrypoint_rejects_historical_or_wrong_method(
    tmp_path,
    monkeypatch,
    field: str,
    value: int | str,
    message: str,
) -> None:
    args, healthy, context_checkpoint = _persist_artifact_admission_fixture(tmp_path)
    payload = torch.load(context_checkpoint, map_location="cpu", weights_only=True)
    payload[field] = value
    torch.save(payload, context_checkpoint)
    monkeypatch.setattr(
        preflight,
        "load_fada_policy_checkpoint",
        lambda *_args, **_kwargs: SimpleNamespace(policy=healthy),
    )

    with pytest.raises(ValueError, match=message):
        preflight.run_preflight(args)
