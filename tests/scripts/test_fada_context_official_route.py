from __future__ import annotations

import copy
import hashlib
import importlib
import json
import runpy
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from scripts import evaluate_fada_context_support_query as evaluate_cli
from scripts import play_fada_context_viser as playback_cli
from scripts import preflight_fada_context_support_query as preflight_cli
from torch import nn

from unilab.algos.torch.distill import (
    FADAArchitectureConfig,
    FADAInverseDynamicsModel,
    FADAPlannerIDMPolicy,
    load_fada_deployable_policy_checkpoint,
    load_fada_policy_checkpoint,
)
from unilab.algos.torch.fada_context import support_query_training as context_training
from unilab.algos.torch.fada_context.support_query import (
    ContextActionOutput,
    ContextQueryBatch,
    FADASupportContextEncoder,
    FrozenIDMSupportQueryPolicy,
    SupportBoundContextPolicy,
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
    PreparedContextSupportQueryArtifact,
    prepare_support_query_training,
    save_context_support_query_checkpoint,
)
from unilab.visualization import interactive_playback as playback_owner

DESIGN_ID = "ICE-Cal / ICA-DP-08 / FADA-CONTEXT-METHOD-v006 + FADA-CONTEXT-TRAIN-v005"
CHECKOUT_ID = (
    "codex/in-context-execution-calibration@5949136e43d3"
    "+content-sha256:2ec4a818a4e1d085ba83d0c3e81928d1bbcf756a2006082cc884f1e9fc3c8c6b"
)
EXPECTED_SECOND_CYCLE_DELTA = 0.16519248485565186


def _play_interactive_module() -> ModuleType:
    return importlib.import_module("play_interactive")


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _mapping_digest(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        digest.update(name.encode())
        digest.update(_tensor_digest(values[name]).encode())
    return digest.hexdigest()


def _config_digest(cfg: DictConfig) -> str:
    payload = OmegaConf.to_container(cfg, resolve=True)
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode()).hexdigest()


def _batch_fields(batch: SupportQueryBatch) -> dict[str, torch.Tensor]:
    return {
        "support_target_future": batch.support.target_future,
        "support_realized_state": batch.support.realized_state,
        "support_executed_action": batch.support.executed_action,
        "query_observation_history": batch.query.observation_history,
        "query_action_history": batch.query.action_history,
        "query_command": batch.query.command,
        "query_planner_intent": batch.query.planner_intent,
        "query_realized_future": batch.query.realized_future,
        "query_executed_action": batch.query.executed_action,
        "query_window_anchor": batch.query.window_anchor,
        "query_valid_window_mask": batch.query.valid_window_mask,
        "support_command": batch.support_command,
        "pair_id": batch.pair_id,
        "support_rollout_id": batch.support_rollout_id,
        "query_rollout_id": batch.query_rollout_id,
    }


def _cloned_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _architecture() -> FADAArchitectureConfig:
    return FADAArchitectureConfig(
        obs_dim=4,
        action_dim=2,
        command_dim=3,
        history_length=30,
        prediction_horizon=6,
        hidden_dim=8,
        num_heads=2,
        planner_layers=1,
        idm_encoder_layers=1,
        idm_decoder_layers=1,
        feedforward_dim=16,
        dropout=0.0,
    )


def _configure_deterministic_source(policy: FADAPlannerIDMPolicy) -> None:
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        for module in policy.idm.modules():
            if isinstance(module, nn.LayerNorm):
                module.weight.fill_(1.0)
        positions = policy.idm.future_position.embedding
        positions[0, 0, 0] = 1.0
        positions[0, 0, 1] = -1.0
        for index in range(1, policy.config.prediction_horizon):
            positions[0, index, 0] = -1.0
            positions[0, index, 1] = 1.0
            positions[0, index, 2] = float(index) * 0.1
        policy.idm.action_head.weight[0, 0] = 1.0
        policy.idm.action_head.weight[1, 1] = 1.0


def _support_query_batch(config: FADAArchitectureConfig) -> SupportQueryBatch:
    pairs, support_length, windows = 6, 2, 1

    def ramp(shape: tuple[int, ...], offset: float) -> torch.Tensor:
        return torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape) * 1.0e-3 + offset

    command = torch.tensor([[0.4, 0.0, 0.0]], dtype=torch.float32).repeat(pairs, 1)
    return SupportQueryBatch(
        support=SupportContextBatch(
            target_future=ramp(
                (pairs, support_length, config.prediction_horizon, config.obs_dim),
                0.1,
            ),
            realized_state=ramp((pairs, support_length, config.obs_dim), 0.2),
            executed_action=ramp((pairs, support_length, config.action_dim), 0.3),
        ),
        query=ContextQueryBatch(
            observation_history=ramp(
                (pairs, windows, config.history_length, config.obs_dim),
                0.01,
            ),
            action_history=ramp(
                (pairs, windows, config.history_length, config.action_dim),
                0.02,
            ),
            command=command[:, None, :].clone(),
            planner_intent=ramp(
                (pairs, windows, config.prediction_horizon, config.obs_dim),
                0.4,
            ),
            realized_future=ramp(
                (pairs, windows, config.prediction_horizon, config.obs_dim),
                0.5,
            ),
            executed_action=ramp((pairs, windows, config.action_dim), 0.6),
            window_anchor=torch.full(
                (pairs, windows),
                config.history_length - 1,
                dtype=torch.int64,
            ),
            valid_window_mask=torch.ones((pairs, windows), dtype=torch.bool),
        ),
        support_command=command,
        pair_id=torch.arange(100, 100 + pairs, dtype=torch.int64),
        support_rollout_id=torch.tensor([10, 10, 20, 20, 30, 30], dtype=torch.int64),
        query_rollout_id=torch.tensor([11, 11, 21, 21, 31, 31], dtype=torch.int64),
    ).validate(config, support_length=support_length)


def _configure_deterministic_context(context: FADASupportContextEncoder) -> None:
    hidden = context.context_config.context_hidden_dim
    with torch.no_grad():
        for parameter in context.parameters():
            parameter.zero_()
        context.query_frame_projection.weight[0, 0] = 1.0
        context.query_sequence_encoder.weight_ih_l0[2 * hidden, 0] = 1.0
        context.delta_head.weight[0, hidden] = 1.0


@dataclass(frozen=True)
class _OfficialArtifacts:
    root: Path
    architecture: FADAArchitectureConfig
    context_config: SupportQueryContextConfig
    source_checkpoint: Path
    dataset_path: Path
    context_checkpoint: Path
    support_query_config: Path
    batch: SupportQueryBatch
    train: SupportQueryBatch
    validation: SupportQueryBatch
    dataset_fields: dict[str, torch.Tensor]
    context_state: dict[str, torch.Tensor]
    planner_state: dict[str, torch.Tensor]
    idm_state: dict[str, torch.Tensor]
    source_sha256: str
    dataset_sha256: str
    context_sha256: str
    train_sha256: str
    validation_sha256: str


@pytest.fixture(scope="module")
def official_artifacts(tmp_path_factory: pytest.TempPathFactory) -> _OfficialArtifacts:
    root = tmp_path_factory.mktemp("fada-official-route")
    architecture = _architecture()
    source_policy = FADAPlannerIDMPolicy(architecture).eval()
    _configure_deterministic_source(source_policy)
    planner_state = _cloned_state(source_policy.planner)
    idm_state = _cloned_state(source_policy.idm)
    source_checkpoint = root / "planner_idm_schema2.pt"
    torch.save(
        {
            "schema_version": 2,
            "architecture": asdict(architecture),
            "planner_state_dict": source_policy.planner.state_dict(),
            "idm_state_dict": source_policy.idm.state_dict(),
            "planner_optimizer_state_dict": {},
            "idm_optimizer_state_dict": {},
            "completed_iterations": 0,
            "samples_seen": 0,
            "runtime_config": {},
            "quality_metrics": {},
        },
        source_checkpoint,
    )
    loaded = load_fada_policy_checkpoint(source_checkpoint, device="cpu")
    assert loaded.checkpoint["schema_version"] == 2
    assert loaded.policy.config == architecture
    assert _mapping_digest(_cloned_state(loaded.policy.planner)) == _mapping_digest(planner_state)
    assert _mapping_digest(_cloned_state(loaded.policy.idm)) == _mapping_digest(idm_state)

    batch = _support_query_batch(architecture)
    source_sha256 = sha256_file(source_checkpoint)
    dataset_path = save_support_query_dataset(
        root / "support_query_schema2.pt",
        batch,
        architecture,
        support_length=2,
        query_length=35,
        metadata={
            "source_checkpoint_sha256": source_sha256,
            "task_config": "sac/g1_walk_flat/mujoco_left_knee_070",
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
        support_length=2,
        context_hidden_dim=2,
        context_layers=1,
        delta_scale=0.5,
    )
    setup = prepare_support_query_training(
        loaded.policy,
        context_config,
        learning_rate=3.0e-4,
    )
    _configure_deterministic_context(setup.policy.context_encoder)
    context_state = _cloned_state(setup.policy.context_encoder)
    dataset_sha256 = sha256_file(dataset_path)
    train_sha256 = support_query_split_identity_sha256(train)
    validation_sha256 = support_query_split_identity_sha256(validation)
    context_checkpoint = save_context_support_query_checkpoint(
        root / "context_schema4.pt",
        setup,
        source_checkpoint_sha256=source_sha256,
        dataset_sha256=dataset_sha256,
        train_split_sha256=train_sha256,
        validation_split_sha256=validation_sha256,
        step=7,
        split_seed=17,
        metrics={"validation_mse": 0.125},
        resolved_config={"design": DESIGN_ID},
    )

    support_query_config = root / "support_query.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "checkpoint_path": str(source_checkpoint),
                "task_config": "sac/g1_walk_flat/mujoco_left_knee_070",
                "device": "cpu",
                "seed": 17,
                "collection": {
                    "num_envs": 1,
                    "num_pairs": 6,
                    "support_length": 2,
                    "query_length": 35,
                    "max_reset_pairs": 1,
                    "artifact_path": str(dataset_path),
                },
                "context": {
                    "hidden_dim": 2,
                    "num_layers": 1,
                    "delta_scale": 0.5,
                    "learning_rate": 3.0e-4,
                },
                "training": {
                    "batch_size": 1,
                    "validation_fraction": 0.34,
                    "steps": 1,
                    "log_interval": 1,
                    "checkpoint_interval": 1,
                    "gradient_clip_norm": 1.0,
                    "minimum_zero_context_mse": 0.0,
                    "output_dir": str(root / "unused-training-output"),
                },
                "boundary": {
                    "optimizer_steps_allowed": False,
                    "training_started": False,
                },
            }
        ),
        support_query_config,
    )
    return _OfficialArtifacts(
        root=root,
        architecture=architecture,
        context_config=context_config,
        source_checkpoint=source_checkpoint,
        dataset_path=dataset_path,
        context_checkpoint=context_checkpoint,
        support_query_config=support_query_config,
        batch=batch,
        train=train,
        validation=validation,
        dataset_fields={name: value.clone() for name, value in _batch_fields(batch).items()},
        context_state=context_state,
        planner_state=planner_state,
        idm_state=idm_state,
        source_sha256=source_sha256,
        dataset_sha256=dataset_sha256,
        context_sha256=sha256_file(context_checkpoint),
        train_sha256=train_sha256,
        validation_sha256=validation_sha256,
    )


class _PairedEvalEnv:
    def __init__(self, *, faulted: bool, obs_dim: int, action_dim: int) -> None:
        self.num_envs = 1
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actions: list[np.ndarray] = []
        self.restore_calls = 0
        self.autoreset: list[bool] = []
        self.closed = False
        self._observation = np.zeros((1, obs_dim), dtype=np.float32)
        strength = np.ones((1, 29), dtype=np.float32)
        if faulted:
            strength[:, 3] = 0.7
        self.state = SimpleNamespace(
            obs={"obs": self._observation.copy()},
            info={
                "commands": np.asarray([[0.4, 0.0, 0.0]], dtype=np.float32),
                "privileged_actuator_strength": strength,
            },
        )

    def set_autoreset(self, enabled: bool) -> None:
        self.autoreset.append(bool(enabled))

    def capture_rollout_snapshot(self) -> dict[str, np.ndarray]:
        return {"observation": self._observation.copy()}

    def restore_rollout_snapshot(self, snapshot: dict[str, np.ndarray]) -> None:
        self.restore_calls += 1
        self._observation = snapshot["observation"].copy()
        self.state.obs["obs"] = self._observation.copy()

    def step(self, action: np.ndarray) -> SimpleNamespace:
        action_array = np.asarray(action, dtype=np.float32)
        self.actions.append(action_array.copy())
        self._observation[:, : self.action_dim] += 0.05 * action_array
        self.state.obs["obs"] = self._observation.copy()
        return SimpleNamespace(
            terminated=np.zeros((1,), dtype=np.bool_),
            truncated=np.zeros((1,), dtype=np.bool_),
        )

    def get_base_pos(self) -> np.ndarray:
        return np.pad(self._observation[:, :2], ((0, 0), (0, 1)))

    def get_base_quat(self) -> np.ndarray:
        return np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def get_local_linvel(self) -> np.ndarray:
        return np.pad(self._observation[:, :2], ((0, 0), (0, 1)))

    def get_dof_pos(self) -> np.ndarray:
        return self._observation[:, :2].copy()

    def get_dof_vel(self) -> np.ndarray:
        return self._observation[:, 2:4].copy()

    def close(self) -> None:
        self.closed = True


@dataclass
class _EvaluationHarness:
    calls: list[dict[str, Any]]
    healthy: _PairedEvalEnv | None = None
    fault: _PairedEvalEnv | None = None

    def factory(
        self,
        root_dir: Path,
        cfg: DictConfig,
        *,
        num_envs: int,
        seed: int,
    ) -> tuple[_PairedEvalEnv, _PairedEvalEnv]:
        self.calls.append(
            {
                "root_dir": Path(root_dir),
                "task_config": str(cfg.task_config),
                "num_envs": num_envs,
                "seed": seed,
            }
        )
        self.healthy = _PairedEvalEnv(faulted=False, obs_dim=4, action_dim=2)
        self.fault = _PairedEvalEnv(faulted=True, obs_dim=4, action_dim=2)
        return self.healthy, self.fault


class _PlaybackEnv:
    def __init__(self, *, command: np.ndarray, action_dim: int) -> None:
        self.action_space = SimpleNamespace(
            shape=(action_dim,),
            low=np.full(action_dim, -10.0, dtype=np.float32),
            high=np.full(action_dim, 10.0, dtype=np.float32),
        )
        self.cfg = SimpleNamespace(
            ctrl_dt=0.01,
            asset=SimpleNamespace(base_name="base"),
            scene=None,
        )
        self.state = SimpleNamespace(
            obs={"obs": np.zeros((1, 4), dtype=np.float32)},
            info={"commands": command.copy()},
        )
        self.autoreset: list[bool] = []

    def set_autoreset(self, enabled: bool) -> None:
        self.autoreset.append(bool(enabled))

    def get_physics_state_snapshot(self) -> np.ndarray:
        return np.zeros((1, 8), dtype=np.float64)

    def get_scene_artifacts(self) -> SimpleNamespace:
        return SimpleNamespace(visual_model_file=None)

    def get_playback_model(self) -> SimpleNamespace:
        return SimpleNamespace(nbody=1, stat=SimpleNamespace(extent=1.0))


class _PlaybackWrapper:
    def __init__(
        self,
        env: _PlaybackEnv,
        *,
        device: str,
        policy_obs_mode: str,
        obs_dim: int,
    ) -> None:
        self.env = env
        self.device = device
        self.policy_obs_mode = policy_obs_mode
        self.num_obs = obs_dim
        self.actions: list[torch.Tensor] = []
        self._index = 0
        self._observations = [
            torch.zeros((1, obs_dim), dtype=torch.float32),
            torch.nn.functional.pad(torch.ones((1, 1)), (0, obs_dim - 1)),
            torch.nn.functional.pad(torch.full((1, 1), 2.0), (0, obs_dim - 1)),
        ]

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        self._index = 0
        observation = self._observations[0].clone()
        self.env.state.obs["obs"] = observation.numpy().copy()
        return {"actor": observation}, {}

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
        self.actions.append(actions.detach().cpu().clone())
        self._index += 1
        observation = self._observations[self._index].clone()
        self.env.state.obs["obs"] = observation.numpy().copy()
        return (
            {"actor": observation},
            torch.zeros(1),
            torch.zeros(1, dtype=torch.bool),
            {},
        )

    def get_observations(self) -> dict[str, torch.Tensor]:
        return {"actor": self._observations[self._index].clone()}

    def close(self) -> None:
        return None


class _FakeMjData:
    def __init__(self, _model: Any) -> None:
        self.xpos = np.zeros((1, 3), dtype=np.float64)


class _FakeViewer:
    def __init__(self) -> None:
        self.running_checks = 0
        self.sync_calls = 0
        self.user_scn = SimpleNamespace(ngeom=0)

    def __enter__(self) -> _FakeViewer:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def is_running(self) -> bool:
        self.running_checks += 1
        return self.running_checks <= 2

    def sync(self) -> None:
        self.sync_calls += 1


@dataclass
class _PlaybackHarness:
    command: np.ndarray
    envs: list[_PlaybackEnv]
    wrappers: list[_PlaybackWrapper]
    viewer: _FakeViewer
    viewer_launches: int = 0
    dependency_maps: list[dict[str, Any]] | None = None


def _install_playback_external_seams(
    monkeypatch: pytest.MonkeyPatch,
    artifacts: _OfficialArtifacts,
    *,
    command: np.ndarray,
) -> _PlaybackHarness:
    harness = _PlaybackHarness(
        command=command.copy(),
        envs=[],
        wrappers=[],
        viewer=_FakeViewer(),
        dependency_maps=[],
    )
    original_default = playback_owner._default_fada_playback_deps

    def create_env(
        _cfg: DictConfig,
        *,
        num_envs: int,
        env_cfg_override: dict[str, Any],
        sim_backend: str,
        task_name: str,
    ) -> _PlaybackEnv:
        assert num_envs == 1
        assert isinstance(env_cfg_override, dict)
        assert sim_backend == "mujoco"
        assert task_name == "G1WalkHeight"
        env = _PlaybackEnv(command=harness.command, action_dim=artifacts.architecture.action_dim)
        harness.envs.append(env)
        return env

    def create_wrapper(
        env: _PlaybackEnv,
        *,
        device: str,
        policy_obs_mode: str,
    ) -> _PlaybackWrapper:
        wrapper = _PlaybackWrapper(
            env,
            device=device,
            policy_obs_mode=policy_obs_mode,
            obs_dim=artifacts.architecture.obs_dim,
        )
        harness.wrappers.append(wrapper)
        return wrapper

    def external_deps(root_dir: str | Path) -> dict[str, Any]:
        deps = cast(dict[str, Any], original_default(root_dir))
        assert deps["load_fada_policy"] is load_fada_deployable_policy_checkpoint
        deps["create_env"] = create_env
        deps["wrapper_cls"] = create_wrapper
        assert harness.dependency_maps is not None
        harness.dependency_maps.append(deps)
        return deps

    interactive = _play_interactive_module()

    def launch_passive(
        _model: Any,
        _data: Any,
        *,
        key_callback: Any,
    ) -> _FakeViewer:
        assert callable(key_callback)
        harness.viewer_launches += 1
        return harness.viewer

    monkeypatch.setattr(playback_owner, "_default_fada_playback_deps", external_deps)
    monkeypatch.setattr(interactive.mujoco, "MjData", _FakeMjData)
    monkeypatch.setattr(interactive.mujoco, "mj_setState", lambda *_args: None)
    monkeypatch.setattr(interactive.mujoco, "mj_forward", lambda *_args: None)
    monkeypatch.setattr(interactive.mujoco.viewer, "launch_passive", launch_passive)
    monkeypatch.setattr(interactive.time, "sleep", lambda _seconds: None)
    return harness


@dataclass
class _OwnerReceipts:
    context: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    bound: list[dict[str, Any]]


def _install_owner_spies(
    monkeypatch: pytest.MonkeyPatch,
    phase: dict[str, str],
) -> _OwnerReceipts:
    receipts = _OwnerReceipts(context=[], actions=[], bound=[])
    original_context = FADASupportContextEncoder.forward
    original_act = FrozenIDMSupportQueryPolicy.act_with_context
    original_bound = SupportBoundContextPolicy.forward

    def context_forward(
        owner: FADASupportContextEncoder,
        support: SupportContextBatch,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        delta_z = original_context(owner, support, observation_history, action_history)
        receipts.context.append(
            {
                "phase": phase["name"],
                "owner": owner,
                "support_id": id(support),
                "observation_history": observation_history.detach().cpu().clone(),
                "action_history": action_history.detach().cpu().clone(),
                "delta_z": delta_z.detach().cpu().clone(),
            }
        )
        return delta_z

    def act_with_context(
        owner: FrozenIDMSupportQueryPolicy,
        support: SupportContextBatch,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> ContextActionOutput:
        output = original_act(owner, support, observation_history, action_history, command)
        receipts.actions.append(
            {
                "phase": phase["name"],
                "owner": owner,
                "support_id": id(support),
                "command": command.detach().cpu().clone(),
                "delta_z": output.delta_z.detach().cpu().clone(),
                "action_chunk": output.action_chunk.detach().cpu().clone(),
                "action": output.action.detach().cpu().clone(),
            }
        )
        return output

    def bound_forward(
        owner: SupportBoundContextPolicy,
        observation_history: torch.Tensor,
        action_history: torch.Tensor,
        command: torch.Tensor,
    ) -> ContextActionOutput:
        output = cast(
            ContextActionOutput,
            original_bound(owner, observation_history, action_history, command),
        )
        receipts.bound.append(
            {
                "phase": phase["name"],
                "owner": owner,
                "support_id": id(owner.support),
                "support": tuple(value.detach().cpu().clone() for value in owner.support.tensors()),
                "support_command": owner.support_command.detach().cpu().clone(),
            }
        )
        return output

    monkeypatch.setattr(FADASupportContextEncoder, "forward", context_forward)
    monkeypatch.setattr(FrozenIDMSupportQueryPolicy, "act_with_context", act_with_context)
    monkeypatch.setattr(SupportBoundContextPolicy, "forward", bound_forward)
    return receipts


def _invoke_preflight_main(
    monkeypatch: pytest.MonkeyPatch,
    artifacts: _OfficialArtifacts,
    *,
    context_checkpoint: Path | None = None,
    label: str,
) -> dict[str, Any]:
    output = artifacts.root / f"preflight-{label}.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preflight_fada_context_support_query.py",
            "--config",
            str(artifacts.support_query_config),
            "--artifact-admission",
            "--dataset",
            str(artifacts.dataset_path),
            "--context-checkpoint",
            str(context_checkpoint or artifacts.context_checkpoint),
            "--output",
            str(output),
        ],
    )
    assert preflight_cli.main() == 0
    return cast(dict[str, Any], json.loads(output.read_text(encoding="utf-8")))


def _invoke_evaluator_main(
    monkeypatch: pytest.MonkeyPatch,
    artifacts: _OfficialArtifacts,
    harness: _EvaluationHarness,
) -> tuple[Path, dict[str, Any]]:
    output = artifacts.root / "evaluation.json"
    monkeypatch.setattr(evaluate_cli, "create_fixed_fault_paired_environments", harness.factory)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_fada_context_support_query.py",
            "--config",
            str(artifacts.support_query_config),
            "--context-checkpoint",
            str(artifacts.context_checkpoint),
            "--dataset",
            str(artifacts.dataset_path),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--num-envs",
            "1",
            "--steps",
            "2",
            "--seed",
            "101",
        ],
    )
    assert evaluate_cli.main() == 0
    return output, json.loads(output.read_text(encoding="utf-8"))


def _playback_overrides(
    artifacts: _OfficialArtifacts,
    *,
    action_mode: str,
    label: str,
) -> list[str]:
    return [
        "+context_playback=left_knee_070",
        f"context_playback.healthy_checkpoint={artifacts.source_checkpoint}",
        f"context_playback.context_checkpoint={artifacts.context_checkpoint}",
        f"context_playback.dataset={artifacts.dataset_path}",
        "context_playback.support_length=2",
        "+context_playback.query_length=35",
        "context_playback.validation_fraction=0.34",
        "context_playback.split_seed=17",
        "context_playback.support_index=0",
        "context_playback.hidden_dim=2",
        "context_playback.num_layers=1",
        "context_playback.delta_scale=0.5",
        f"training.play_checkpoint_path={artifacts.source_checkpoint}",
        "training.device=cpu",
        f"interactive.action_mode={action_mode}",
        "interactive.policy_obs_mode=actor",
        "interactive.use_env_visual_model=false",
        "interactive.show_target_bodies=false",
        "interactive.show_reward_debug=false",
        "interactive.keyboard=false",
        f"hydra.run.dir={artifacts.root / f'hydra-{label}'}",
        "hydra.output_subdir=null",
        "hydra.job.chdir=false",
    ]


def _invoke_playback_main(
    monkeypatch: pytest.MonkeyPatch,
    artifacts: _OfficialArtifacts,
    *,
    action_mode: str,
    label: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "play_fada_context_viser.py",
            *_playback_overrides(artifacts, action_mode=action_mode, label=label),
        ],
    )
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
    GlobalHydra.instance().clear()
    try:
        assert playback_cli.__file__ is not None
        runpy.run_path(playback_cli.__file__, run_name="__main__")
    finally:
        GlobalHydra.instance().clear()


def test_official_offline_transaction_carries_v006_artifact_to_two_first_actions(
    official_artifacts: _OfficialArtifacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = official_artifacts
    phase = {"name": "preflight"}
    owner_receipts = _install_owner_spies(monkeypatch, phase)
    prepared: list[PreparedContextSupportQueryArtifact] = []
    real_evaluator_prepare = evaluate_cli.prepare_context_support_query_artifact

    def record_evaluator_prepare(*args: Any, **kwargs: Any) -> PreparedContextSupportQueryArtifact:
        artifact = real_evaluator_prepare(*args, **kwargs)
        prepared.append(artifact)
        return artifact

    monkeypatch.setattr(
        evaluate_cli,
        "prepare_context_support_query_artifact",
        record_evaluator_prepare,
    )

    preflight_report = _invoke_preflight_main(
        monkeypatch,
        artifacts,
        label="official",
    )
    assert preflight_report["mode"] == "artifact_admission"
    assert preflight_report["method_contract_id"] == "FADA-CONTEXT-METHOD-v006"
    assert preflight_report["checkpoint_schema"] == 4
    assert preflight_report["checkpoint_step"] == 7
    assert preflight_report["tensors"]["delta_z"] == [6, 1, 8]

    phase["name"] = "evaluator"
    evaluation_harness = _EvaluationHarness(calls=[])
    evaluation_path, evaluation_report = _invoke_evaluator_main(
        monkeypatch,
        artifacts,
        evaluation_harness,
    )
    assert evaluation_report["schema"] == (
        "unilab_fada_context_support_query_closed_loop_artifact_v2"
    )
    assert not evaluation_path.with_suffix(f"{evaluation_path.suffix}.tmp").exists()
    assert len(prepared) == 1
    prepared_artifact = prepared[0]
    for name, expected in artifacts.dataset_fields.items():
        torch.testing.assert_close(
            _batch_fields(prepared_artifact.dataset)[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )
    assert _mapping_digest(_batch_fields(prepared_artifact.dataset)) == _mapping_digest(
        artifacts.dataset_fields
    )
    assert _mapping_digest(_cloned_state(prepared_artifact.policy.context_encoder)) == (
        _mapping_digest(artifacts.context_state)
    )

    phase["name"] = "playback"
    playback_harness = _install_playback_external_seams(
        monkeypatch,
        artifacts,
        command=np.asarray([[0.4, 0.0, 0.0]], dtype=np.float32),
    )
    interactive = _play_interactive_module()
    real_build_playback_config = interactive._build_playback_config
    real_create_fada_session = playback_owner.create_fada_playback_session
    real_advance = playback_owner.FADAPlaybackSession.advance
    real_playback_prepare = context_training.prepare_context_support_query_artifact
    playback_configs: list[Any] = []
    playback_sessions: list[Any] = []
    playback_prepared: list[PreparedContextSupportQueryArtifact] = []
    effective_config_digests: list[str] = []
    advance_calls: list[int] = []

    def record_build_playback_config(*args: Any, **kwargs: Any) -> Any:
        config = real_build_playback_config(*args, **kwargs)
        playback_configs.append(config)
        return config

    def record_playback_prepare(*args: Any, **kwargs: Any) -> PreparedContextSupportQueryArtifact:
        artifact = real_playback_prepare(*args, **kwargs)
        playback_prepared.append(artifact)
        return artifact

    def record_create_fada_session(**kwargs: Any) -> Any:
        cfg = kwargs["cfg"]
        assert isinstance(cfg, DictConfig)
        assert bool(cfg.context_playback.enabled)
        assert str(cfg.interactive.action_mode) == "policy"
        effective_config_digests.append(_config_digest(cfg))
        result = real_create_fada_session(**kwargs)
        playback_sessions.append(result[0])
        return result

    def record_advance(owner: playback_owner.FADAPlaybackSession, controls: Any) -> bool:
        advanced = real_advance(owner, controls)
        advance_calls.append(owner.step_count)
        return cast(bool, advanced)

    monkeypatch.setattr(interactive, "_build_playback_config", record_build_playback_config)
    monkeypatch.setattr(
        context_training,
        "prepare_context_support_query_artifact",
        record_playback_prepare,
    )
    monkeypatch.setattr(
        playback_owner,
        "create_fada_playback_session",
        record_create_fada_session,
    )
    monkeypatch.setattr(playback_owner.FADAPlaybackSession, "advance", record_advance)

    _invoke_playback_main(
        monkeypatch,
        artifacts,
        action_mode="policy",
        label="official",
    )

    playback_context = [row for row in owner_receipts.context if row["phase"] == "playback"]
    playback_actions = [row for row in owner_receipts.actions if row["phase"] == "playback"]
    playback_bound = [row for row in owner_receipts.bound if row["phase"] == "playback"]
    assert len(playback_configs) == 1
    assert playback_configs[0].action_mode == "policy"
    assert len(playback_sessions) == 1
    assert len(effective_config_digests) == 1
    assert advance_calls == [1, 2]
    assert playback_harness.viewer_launches == 1
    assert playback_harness.viewer.sync_calls == 2
    assert len(playback_context) == len(playback_actions) == len(playback_bound) == 2
    assert len(playback_harness.wrappers) == 1
    consumed_actions = playback_harness.wrappers[0].actions
    assert len(consumed_actions) == 2

    torch.testing.assert_close(
        playback_context[0]["observation_history"][:, -1],
        torch.zeros((1, artifacts.architecture.obs_dim)),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        playback_context[1]["observation_history"][:, -1],
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        playback_actions[0]["delta_z"],
        torch.zeros((1, artifacts.architecture.hidden_dim)),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        playback_actions[1]["delta_z"][0, 0],
        torch.tensor(EXPECTED_SECOND_CYCLE_DELTA),
        rtol=0.0,
        atol=1.0e-7,
    )
    for consumed, action_receipt in zip(consumed_actions, playback_actions, strict=True):
        assert consumed.ndim == 2
        torch.testing.assert_close(
            consumed,
            action_receipt["action_chunk"][:, 0],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            consumed,
            action_receipt["action"],
            rtol=0.0,
            atol=0.0,
        )
        assert not torch.equal(consumed, action_receipt["action_chunk"][:, 1])
    torch.testing.assert_close(
        consumed_actions[1][0, 0] - consumed_actions[0][0, 0],
        torch.tensor(EXPECTED_SECOND_CYCLE_DELTA),
        rtol=0.0,
        atol=2.0e-7,
    )
    assert len({row["support_id"] for row in playback_actions}) == 1
    assert len({row["support_id"] for row in playback_bound}) == 1
    for left, right in zip(playback_bound[0]["support"], playback_bound[1]["support"], strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        playback_bound[0]["support_command"],
        playback_bound[1]["support_command"],
        rtol=0.0,
        atol=0.0,
    )

    assert len(playback_prepared) == 1
    playback_artifact = playback_prepared[0]
    for name, expected in artifacts.dataset_fields.items():
        torch.testing.assert_close(
            _batch_fields(playback_artifact.dataset)[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )
    bound_owner = playback_bound[0]["owner"]
    for observed, expected in zip(
        bound_owner.support.tensors(),
        playback_artifact.validation.support.index_select(torch.tensor([0])).tensors(),
        strict=True,
    ):
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
    bound_context_state = _cloned_state(bound_owner.policy.context_encoder)
    assert set(bound_context_state) == set(artifacts.context_state)
    for name, expected in artifacts.context_state.items():
        torch.testing.assert_close(
            bound_context_state[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )
    assert _mapping_digest(bound_context_state) == _mapping_digest(artifacts.context_state)
    assert _mapping_digest(_cloned_state(bound_owner.policy.planner)) == _mapping_digest(
        artifacts.planner_state
    )
    assert _mapping_digest(_cloned_state(bound_owner.policy.idm)) == _mapping_digest(
        artifacts.idm_state
    )
    assert all(not parameter.requires_grad for parameter in bound_owner.policy.planner.parameters())
    assert all(not parameter.requires_grad for parameter in bound_owner.policy.idm.parameters())
    assert bound_owner.policy.planner.training is False
    assert bound_owner.policy.idm.training is False

    assert playback_sessions[0].step_count == 2
    assert evaluation_harness.calls == [
        {
            "root_dir": evaluate_cli.ROOT_DIR,
            "task_config": "sac/g1_walk_flat/mujoco_left_knee_070",
            "num_envs": 1,
            "seed": 101,
        }
    ]
    assert evaluation_harness.healthy is not None
    assert evaluation_harness.fault is not None
    assert len(evaluation_harness.healthy.actions) == 2
    assert len(evaluation_harness.fault.actions) == 4
    assert evaluation_harness.healthy.restore_calls == 1
    assert evaluation_harness.fault.restore_calls == 2
    assert evaluation_harness.healthy.autoreset == [False]
    assert evaluation_harness.fault.autoreset == [False]
    assert evaluation_harness.healthy.closed is True
    assert evaluation_harness.fault.closed is True
    assert len([row for row in owner_receipts.context if row["phase"] == "preflight"]) == 1
    assert len([row for row in owner_receipts.context if row["phase"] == "evaluator"]) == 2
    assert len([row for row in owner_receipts.actions if row["phase"] == "evaluator"]) == 2
    assert len([row for row in owner_receipts.bound if row["phase"] == "evaluator"]) == 2
    assert evaluation_report["healthy_checkpoint_sha256"] == artifacts.source_sha256
    assert evaluation_report["dataset_sha256"] == artifacts.dataset_sha256
    assert evaluation_report["context_checkpoint_sha256"] == artifacts.context_sha256
    assert evaluation_report["train_split_sha256"] == artifacts.train_sha256
    assert evaluation_report["validation_split_sha256"] == artifacts.validation_sha256
    edge_receipts = {
        "EDGE-01": {
            "design": DESIGN_ID,
            "checkout": CHECKOUT_ID,
            "method": preflight_report["method_contract_id"],
            "schema": preflight_report["checkpoint_schema"],
            "step": preflight_report["checkpoint_step"],
            "history_length": artifacts.architecture.history_length,
            "prediction_horizon": artifacts.architecture.prediction_horizon,
            "support_length": artifacts.context_config.support_length,
            "source": artifacts.source_sha256,
            "dataset": artifacts.dataset_sha256,
            "context": artifacts.context_sha256,
            "train": artifacts.train_sha256,
            "validation": artifacts.validation_sha256,
            "pair_ids": preflight_report["query_provenance"]["pair_ids"],
            "support_rollout_ids": preflight_report["query_provenance"]["support_rollout_ids"],
            "query_rollout_ids": preflight_report["query_provenance"]["query_rollout_ids"],
        },
        "EDGE-02": {
            "support_id": playback_bound[0]["support_id"],
            "command": playback_bound[0]["support_command"].tolist(),
            "effective_config_sha256": effective_config_digests[0],
        },
        "EDGE-03": {"context_calls": len(playback_context), "act_calls": len(playback_actions)},
        "EDGE-04": {
            "advance_calls": advance_calls,
            "consumed_actions": len(consumed_actions),
            "margin": float(consumed_actions[1][0, 0] - consumed_actions[0][0, 0]),
        },
        "EDGE-05": {
            "dataset_value_digest": _mapping_digest(artifacts.dataset_fields),
            "context_value_digest": _mapping_digest(artifacts.context_state),
            "dataset_file_sha256": artifacts.dataset_sha256,
            "context_file_sha256": artifacts.context_sha256,
            "first_consumer_step": 1,
        },
        "EDGE-06": {
            "env_factory_calls": len(evaluation_harness.calls),
            "report": str(evaluation_path),
            "delta_shape": evaluation_report["reports"][0]["context"]["delta_z_trace_shape"],
        },
    }
    assert set(edge_receipts) == {f"EDGE-0{index}" for index in range(1, 7)}
    assert edge_receipts["EDGE-03"] == {"context_calls": 2, "act_calls": 2}
    assert edge_receipts["EDGE-04"]["consumed_actions"] == 2
    assert edge_receipts["EDGE-06"]["delta_shape"] == [2, 1, 8]


def test_official_preflight_rejects_schema3_before_context_construction(
    official_artifacts: _OfficialArtifacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = official_artifacts.root / "context_schema3.pt"
    payload = torch.load(
        official_artifacts.context_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    payload["schema_version"] = 3
    torch.save(payload, historical)
    original_init = FADASupportContextEncoder.__init__
    context_constructions = 0

    def record_init(owner: FADASupportContextEncoder, *args: Any, **kwargs: Any) -> None:
        nonlocal context_constructions
        context_constructions += 1
        original_init(owner, *args, **kwargs)

    monkeypatch.setattr(FADASupportContextEncoder, "__init__", record_init)
    with pytest.raises(ValueError, match="historical fixed-residual checkpoint schema"):
        _invoke_preflight_main(
            monkeypatch,
            official_artifacts,
            context_checkpoint=historical,
            label="schema3",
        )
    assert context_constructions == 0


def test_official_playback_rejects_nonpolicy_mode_before_action_consumption(
    official_artifacts: _OfficialArtifacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase = {"name": "nonpolicy"}
    receipts = _install_owner_spies(monkeypatch, phase)
    harness = _install_playback_external_seams(
        monkeypatch,
        official_artifacts,
        command=np.asarray([[0.4, 0.0, 0.0]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="action_mode=policy"):
        _invoke_playback_main(
            monkeypatch,
            official_artifacts,
            action_mode="zero",
            label="nonpolicy",
        )
    assert receipts.context == []
    assert receipts.actions == []
    assert harness.viewer_launches == 0
    assert len(harness.wrappers) == 1
    assert harness.wrappers[0].actions == []


def test_official_playback_rejects_command_mismatch_before_context_action(
    official_artifacts: _OfficialArtifacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase = {"name": "command-mismatch"}
    receipts = _install_owner_spies(monkeypatch, phase)
    harness = _install_playback_external_seams(
        monkeypatch,
        official_artifacts,
        command=np.asarray([[0.5, 0.0, 0.0]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="does not match Support command provenance"):
        _invoke_playback_main(
            monkeypatch,
            official_artifacts,
            action_mode="policy",
            label="command-mismatch",
        )
    assert receipts.context == []
    assert receipts.actions == []
    assert harness.viewer_launches == 1
    assert len(harness.wrappers) == 1
    assert harness.wrappers[0].actions == []
