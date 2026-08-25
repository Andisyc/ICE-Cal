from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill import FADAArchitectureConfig
from unilab.algos.torch.distill.fada_target_data import (
    FADATargetBatch,
    load_fada_target_artifact,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT_DIR / "conf" / "offpolicy"
SCRIPT_PATH = ROOT_DIR / "scripts" / "collect_fada_target.py"


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("collect_fada_target", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _compose_target(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(
            config_name="fada_target",
            overrides=list(overrides),
            return_hydra_config=True,
        )


def _small_config() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=66,
        action_dim=29,
        command_dim=3,
        observation_contract="g1_fada_state_v2",
        history_length=2,
        prediction_horizon=2,
        hidden_dim=8,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=16,
    )


def _target_batch(config: FADAArchitectureConfig) -> FADATargetBatch:
    return FADATargetBatch(
        observation_history=torch.zeros(1, config.history_length, config.obs_dim),
        action_history=torch.zeros(1, config.history_length, config.action_dim),
        command=torch.zeros(1, config.command_dim),
        realized_future=torch.zeros(1, config.prediction_horizon, config.obs_dim),
        executed_action_chunk=torch.zeros(1, config.prediction_horizon, config.action_dim),
        episode_id=torch.zeros(1, dtype=torch.int64),
        start_timestep=torch.zeros(1, dtype=torch.int64),
    ).validate(config)


def test_target_config_reuses_exact_offpolicy_task_owner() -> None:
    cfg = _compose_target()

    assert cfg.hydra.runtime.choices.task == "sac/g1_walk_flat/mujoco_left_knee_090"
    assert cfg.hydra.runtime.choices.collection == "fada_target"
    assert cfg.algo.algo == "sac"
    assert cfg.training.task_name == "G1WalkFlat"
    assert cfg.training.sim_backend == "mujoco"
    assert cfg.env.commands.vel_limit == [[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]]
    assert cfg.env.domain_rand.actuator_strength.multipliers[3] == 0.9
    assert len(cfg.env.domain_rand.actuator_strength.multipliers) == 29
    assert cfg.collection.policy_checkpoint_path == "logs/fada/planner_idm_v006_state66.pt"
    assert cfg.collection.expected_checkpoint_sha256 is None
    assert cfg.collection.output_path.endswith("_v2.pt")


@pytest.mark.parametrize(
    ("selector", "value", "match"),
    [
        ("env.commands.vel_limit", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], "command"),
        ("env.domain_rand.actuator_strength.multipliers.3", 0.8, "actuator"),
        ("training.sim_backend", "motrix", "MuJoCo"),
    ],
)
def test_target_preflight_rejects_identity_drift_before_env_creation(
    tmp_path: Path,
    selector: str,
    value: Any,
    match: str,
) -> None:
    module = _load_script()
    checkpoint = tmp_path / "planner.pt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = _compose_target(
        f"collection.policy_checkpoint_path={checkpoint}",
        f"collection.output_path={tmp_path / 'target.pt'}",
        f"collection.expected_checkpoint_sha256={module.file_sha256(checkpoint)}",
    )
    OmegaConf.update(cfg, selector, value, merge=False)

    with pytest.raises(ValueError, match=match):
        module.preflight_fada_target_collection(cfg, root_dir=ROOT_DIR)


@pytest.mark.parametrize(
    ("selector", "value"),
    [
        ("collection.num_envs", 1.5),
        ("collection.num_windows", 2.75),
        ("collection.max_env_steps", 3.5),
    ],
)
def test_target_preflight_rejects_fractional_positive_integer_fields(
    tmp_path: Path,
    selector: str,
    value: float,
) -> None:
    module = _load_script()
    checkpoint = tmp_path / "planner.pt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = _compose_target(
        f"collection.policy_checkpoint_path={checkpoint}",
        f"collection.output_path={tmp_path / 'target.pt'}",
        f"collection.expected_checkpoint_sha256={module.file_sha256(checkpoint)}",
    )
    OmegaConf.update(cfg, selector, value, merge=False)

    with pytest.raises(ValueError, match=rf"{selector} must be a positive integer"):
        module.preflight_fada_target_collection(cfg, root_dir=ROOT_DIR)


def test_target_preflight_refuses_checkpoint_output_alias_and_existing_output(
    tmp_path: Path,
) -> None:
    module = _load_script()
    checkpoint = tmp_path / "planner.pt"
    checkpoint.write_bytes(b"checkpoint")
    digest = module.file_sha256(checkpoint)
    alias_cfg = _compose_target(
        f"collection.policy_checkpoint_path={checkpoint}",
        f"collection.output_path={checkpoint}",
        f"collection.expected_checkpoint_sha256={digest}",
    )
    with pytest.raises(ValueError, match="must differ"):
        module.preflight_fada_target_collection(alias_cfg, root_dir=ROOT_DIR)

    output = tmp_path / "already-there.pt"
    output.write_bytes(b"owned")
    existing_cfg = _compose_target(
        f"collection.policy_checkpoint_path={checkpoint}",
        f"collection.output_path={output}",
        f"collection.expected_checkpoint_sha256={digest}",
    )
    with pytest.raises(FileExistsError, match="already exists"):
        module.preflight_fada_target_collection(existing_cfg, root_dir=ROOT_DIR)


def test_target_runner_loads_before_env_collects_saves_and_closes(tmp_path: Path) -> None:
    module = _load_script()
    checkpoint = tmp_path / "planner.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "target.pt"
    cfg = _compose_target(
        f"collection.policy_checkpoint_path={checkpoint}",
        f"collection.output_path={output}",
        f"collection.expected_checkpoint_sha256={module.file_sha256(checkpoint)}",
        "collection.num_envs=1",
        "collection.num_windows=1",
        "collection.max_env_steps=7",
    )
    config = _small_config()
    events: list[str] = []

    class _Env:
        num_envs = 1

        def close(self) -> None:
            events.append("close")

    env = _Env()
    policy = SimpleNamespace(config=config)
    collection_result = SimpleNamespace(
        batch=_target_batch(config),
        env_steps=3,
        rejected_done_transitions=0,
        rejected_command_windows=0,
    )

    def load_policy(path: Path, *, device: str) -> SimpleNamespace:
        assert path == checkpoint
        assert device == "cpu"
        events.append("load")
        return SimpleNamespace(policy=policy, checkpoint={"schema_version": 3})

    def ensure_registries() -> None:
        events.append("registry")

    def create_env(
        owner_cfg: DictConfig,
        *,
        num_envs: int,
        env_cfg_override: dict[str, Any],
        sim_backend: str,
    ) -> _Env:
        assert owner_cfg is cfg
        assert num_envs == 1
        assert sim_backend == "mujoco"
        assert env_cfg_override["domain_rand"]["actuator_strength"]["multipliers"][3] == 0.9
        events.append("env")
        return env

    def collect(
        created_env: _Env,
        rollout_policy: Any,
        architecture: FADAArchitectureConfig,
        num_windows: int,
        spec: Any,
    ) -> Any:
        assert created_env is env
        assert rollout_policy is policy
        assert architecture == config
        assert num_windows == 1
        assert spec.max_env_steps == 7
        events.append("collect")
        return collection_result

    def save(
        path: Path,
        batch: FADATargetBatch,
        *,
        config: FADAArchitectureConfig,
        metadata: dict[str, Any],
    ) -> Path:
        assert path == output
        assert batch is collection_result.batch
        assert config == policy.config
        assert metadata["policy_checkpoint_sha256"] == module.file_sha256(checkpoint)
        assert metadata["task"] == "G1WalkFlat"
        assert metadata["fault_profile"] == "left_knee_strength_0.9"
        assert metadata["num_windows"] == 1
        events.append("save")
        return path

    summary = module.run_fada_target_collection(
        cfg,
        root_dir=ROOT_DIR,
        load_policy_fn=load_policy,
        ensure_registries_fn=ensure_registries,
        create_env_fn=create_env,
        collect_fn=collect,
        save_fn=save,
    )

    assert events == ["load", "registry", "env", "collect", "save", "close"]
    assert summary["status"] == "completed"
    assert summary["env_steps"] == 3
    assert summary["artifact_path"] == str(output)


def test_target_runner_rejects_legacy_checkpoint_before_registry_or_env(
    tmp_path: Path,
) -> None:
    module = _load_script()
    checkpoint = tmp_path / "planner.pt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = _compose_target(
        f"collection.policy_checkpoint_path={checkpoint}",
        f"collection.output_path={tmp_path / 'target.pt'}",
        f"collection.expected_checkpoint_sha256={module.file_sha256(checkpoint)}",
    )
    events: list[str] = []

    with pytest.raises(ValueError, match="projection does not match|active FADA route"):
        module.run_fada_target_collection(
            cfg,
            root_dir=ROOT_DIR,
            load_policy_fn=lambda *_args, **_kwargs: SimpleNamespace(
                policy=SimpleNamespace(
                    config=FADAArchitectureConfig(
                        obs_dim=3,
                        action_dim=2,
                        command_dim=2,
                        history_length=2,
                        prediction_horizon=2,
                        hidden_dim=8,
                        num_heads=2,
                        planner_layers=1,
                        idm_encoder_layers=1,
                        idm_decoder_layers=1,
                        feedforward_dim=16,
                    )
                ),
                checkpoint={"schema_version": 3},
            ),
            ensure_registries_fn=lambda: events.append("registry"),
            create_env_fn=lambda *_args, **_kwargs: events.append("env"),
        )

    assert events == []


def test_target_runner_real_persistence_round_trip_on_official_composition(
    tmp_path: Path,
) -> None:
    module = _load_script()
    checkpoint = tmp_path / "planner.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "target.pt"
    cfg = _compose_target(
        f"collection.policy_checkpoint_path={checkpoint}",
        f"collection.output_path={output}",
        f"collection.expected_checkpoint_sha256={module.file_sha256(checkpoint)}",
        "collection.num_envs=1",
        "collection.num_windows=1",
    )
    config = _small_config()
    policy = SimpleNamespace(config=config)
    batch = _target_batch(config)

    class _Env:
        num_envs = 1

        def close(self) -> None:
            return None

    module.run_fada_target_collection(
        cfg,
        root_dir=ROOT_DIR,
        load_policy_fn=lambda *_args, **_kwargs: SimpleNamespace(
            policy=policy, checkpoint={"schema_version": 3}
        ),
        ensure_registries_fn=lambda: None,
        create_env_fn=lambda *_args, **_kwargs: _Env(),
        collect_fn=lambda *_args, **_kwargs: SimpleNamespace(
            batch=batch,
            env_steps=3,
            rejected_done_transitions=0,
            rejected_command_windows=0,
        ),
    )

    loaded = load_fada_target_artifact(output, config=config)
    torch.testing.assert_close(loaded.batch.observation_history, batch.observation_history)
    assert loaded.metadata["policy_checkpoint_sha256"] == module.file_sha256(checkpoint)
    assert loaded.metadata["num_windows"] == 1


def test_target_runner_closes_env_when_collection_fails(tmp_path: Path) -> None:
    module = _load_script()
    checkpoint = tmp_path / "planner.pt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = _compose_target(
        f"collection.policy_checkpoint_path={checkpoint}",
        f"collection.output_path={tmp_path / 'target.pt'}",
        f"collection.expected_checkpoint_sha256={module.file_sha256(checkpoint)}",
    )
    config = _small_config()
    closed: list[bool] = []

    class _Env:
        num_envs = 1

        def close(self) -> None:
            closed.append(True)

    with pytest.raises(RuntimeError, match="collection failed"):
        module.run_fada_target_collection(
            cfg,
            root_dir=ROOT_DIR,
            load_policy_fn=lambda *_args, **_kwargs: SimpleNamespace(
                policy=SimpleNamespace(config=config), checkpoint={"schema_version": 3}
            ),
            ensure_registries_fn=lambda: None,
            create_env_fn=lambda *_args, **_kwargs: _Env(),
            collect_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("collection failed")
            ),
        )

    assert closed == [True]


def test_target_cli_imports_only_target_collection_boundary() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'config_path="../conf/offpolicy"' in source
    assert 'config_name="fada_target"' in source
    assert "collect_fada_target_windows" in source
    assert "save_fada_target_artifact" in source
    assert "collect_fada_source_windows" not in source
