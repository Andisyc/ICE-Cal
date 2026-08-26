from __future__ import annotations

import json
import multiprocessing as mp
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import unilab.algos.torch.distill.fada_async_runtime as fada_async_runtime
from tests.algos._fada_training_test_support import (
    ROOT,
    _load_train_distill,
)
from unilab.algos.torch.distill import (
    FADA_CHECKPOINT_SCHEMA_VERSION,
    FADAPlannerIDMPolicy,
    MLPStudentPolicy,
    load_fada_checkpoint,
    load_fada_source_batch,
    save_distillation_checkpoint,
)
from unilab.algos.torch.distill.fada import (
    FADA_IDM_SOURCE_ROLE_IDS,
    FADA_SCENARIO_IDS,
)
from unilab.algos.torch.distill.fada_async_runtime import PersistentFADACollectorWorker
from unilab.algos.torch.distill.fada_workflow_setup import (
    build_fada_architecture_config,
)


@dataclass
class _FormalEnvState:
    obs: dict[str, np.ndarray]
    info: dict[str, np.ndarray]
    terminated: np.ndarray
    truncated: np.ndarray


class _FormalG1Env:
    """Deterministic external simulator adapter; owns no FADA semantics."""

    def __init__(self) -> None:
        self.num_envs = 1
        self.action_space = type("ActionSpace", (), {"shape": (29,)})()
        self.current_obs = np.zeros((1, 98), dtype=np.float32)
        self.step_count = 0
        self.closed = False
        self.physics_guard_max_abs: float | None = None
        self.state = self._state(np.asarray([[0.4, 0.0, 0.0]], dtype=np.float32))

    def _state(self, commands: np.ndarray) -> _FormalEnvState:
        return _FormalEnvState(
            obs={"obs": self.current_obs.copy()},
            info={"commands": commands.copy()},
            terminated=np.zeros((1,), dtype=np.bool_),
            truncated=np.zeros((1,), dtype=np.bool_),
        )

    def reset_all(self) -> _FormalEnvState:
        self.current_obs.fill(0.0)
        self.step_count = 0
        self.state = self._state(np.asarray([[0.4, 0.0, 0.0]], dtype=np.float32))
        return self.state

    def refresh_state(self) -> _FormalEnvState:
        commands = np.asarray(self.state.info["commands"], dtype=np.float32)
        self.state = self._state(commands)
        return self.state

    def step(self, actions: np.ndarray) -> _FormalEnvState:
        action_rows = np.asarray(actions, dtype=np.float32)
        if action_rows.shape != (1, 29):
            raise ValueError(f"formal environment action shape mismatch: {action_rows.shape}")
        self.current_obs[:, :29] += 0.01 * np.tanh(action_rows)
        self.current_obs[:, 96:98] = float(self.step_count % 7) / 10.0
        self.step_count += 1
        commands = np.asarray(self.state.info["commands"], dtype=np.float32)
        self.state = self._state(commands)
        return self.state

    @contextmanager
    def preserve_rollout_state(self):
        observation = self.current_obs.copy()
        commands = np.asarray(self.state.info["commands"], dtype=np.float32).copy()
        step_count = self.step_count
        try:
            yield
        finally:
            self.current_obs = observation
            self.step_count = step_count
            self.state = self._state(commands)

    def set_physics_envelope_guard(self, max_abs_state: float | None) -> None:
        self.physics_guard_max_abs = max_abs_state

    def close(self) -> None:
        self.closed = True


class _FormalIntermediateOracle(torch.nn.Module):
    obs_dim = 98
    action_dim = 29

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return 0.05 * torch.tanh(obs[:, :29])


def _formal_env_factory(*_args, **_kwargs) -> _FormalG1Env:
    return _FormalG1Env()


def _formal_intermediate_loader(*_args, **_kwargs) -> torch.nn.Module:
    return _FormalIntermediateOracle()


def _formal_intermediate_reloader(*_args, **_kwargs) -> None:
    return None


def _formal_worker_factory(**kwargs) -> PersistentFADACollectorWorker:
    return PersistentFADACollectorWorker(
        **kwargs,
        env_factory=_formal_env_factory,
        intermediate_teacher_loader=_formal_intermediate_loader,
        intermediate_teacher_reloader=_formal_intermediate_reloader,
    )


def _save_formal_unified_oracle(path: Path) -> None:
    policy = MLPStudentPolicy(
        obs_dim=98,
        action_dim=29,
        hidden_dims=(8,),
        activation="elu",
        squash_action=True,
    )
    with torch.no_grad():
        for index, parameter in enumerate(policy.parameters(), start=1):
            parameter.fill_(index / 100.0)
    save_distillation_checkpoint(
        path,
        student=policy,
        agent_steps=8,
        teacher_metadata={"source": "formal-unified-oracle"},
        distill_runtime_cfg={
            "student_model_type": "mlp",
            "student_obs_dim": 98,
            "student_action_dim": 29,
            "student_hidden_dims": [8],
            "student_activation": "elu",
            "student_squash_action": True,
        },
    )


def _bounded_formal_config(tmp_path: Path):
    cfg = OmegaConf.load(ROOT / "note" / "fada" / "evidence" / "fada_v007r1_final_config.yaml")
    unified_oracle = tmp_path / "oracle" / "dagger_iteration_8.pt"
    unified_oracle.parent.mkdir(parents=True)
    _save_formal_unified_oracle(unified_oracle)
    intermediates = [tmp_path / "oracle" / f"model_{step}.pt" for step in range(240, 4801, 240)]
    for path in intermediates:
        path.touch()

    cfg.training.device = "cpu"
    cfg.teacher.checkpoint_path = str(unified_oracle)
    fada = cfg.training.fada
    if "standing_teacher_checkpoint_path" in fada.stand_transition_curriculum:
        del fada.stand_transition_curriculum.standing_teacher_checkpoint_path
    fada.async_request_timeout_seconds = 30.0
    fada.async_artifact_dir = str(tmp_path / "source_batches")
    fada.intermediate_oracle_checkpoint_paths = [str(path) for path in intermediates]
    fada.quality_eval_max_windows = 12
    fada.iterations = 3
    fada.windows_per_iteration = 12
    fada.num_envs = 1
    fada.replay_capacity = 96
    fada.batch_size = 12
    fada.idm_updates = 1
    fada.planner_updates = 1
    fada.max_env_steps = 72
    fada.checkpoint_path = str(tmp_path / "planner_idm_v011_formal.pt")
    return cfg, unified_oracle, tuple(intermediates)


def test_refactored_official_route_closes_updates_persistence_and_first_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch.manual_seed(20260823)
    module = _load_train_distill()
    cfg, unified_oracle, intermediate_paths = _bounded_formal_config(tmp_path)
    architecture = build_fada_architecture_config(cfg)
    assert (
        architecture.obs_dim,
        architecture.action_dim,
        architecture.command_dim,
        architecture.history_length,
        architecture.prediction_horizon,
    ) == (66, 29, 3, 30, 6)

    loaded_intermediates: list[Path] = []

    def _load_external_checkpoint(path, _spec, *, device):
        assert device == "cpu"
        loaded_intermediates.append(Path(path))
        return _FormalIntermediateOracle()

    monkeypatch.setattr(module, "load_sac_teacher_policy", _load_external_checkpoint)
    monkeypatch.setattr(
        fada_async_runtime,
        "_build_persistent_fada_worker",
        _formal_worker_factory,
    )

    baseline_children = {child.pid for child in mp.active_children()}
    module.main.__wrapped__(cfg)
    result = json.loads(capsys.readouterr().out)

    assert loaded_intermediates == list(intermediate_paths)
    assert len(set(loaded_intermediates)) == 20
    assert unified_oracle.is_file()
    assert result["execution_mode"] == "persistent_async"
    assert result["training_schedule"] == "alternating_idm_then_planner"
    assert result["completed_iterations"] == 3
    assert result["samples_seen"] == 108
    assert result["replay_size"] == 96
    assert result["replay_effective_capacity"] == 96
    assert result["replay_role_counts"] == {
        "planner_eligible": 32,
        "planner_ineligible": 64,
    }
    assert result["last_idm_loss"] is not None
    assert result["last_planner_loss"] is not None
    assert not ({child.pid for child in mp.active_children()} - baseline_children)

    artifacts = [
        load_fada_source_batch(
            Path(cfg.training.fada.async_artifact_dir) / f"iteration_{iteration:04d}.pt",
            config=architecture,
        )
        for iteration in range(3)
    ]
    for iteration, loaded in enumerate(artifacts):
        assert loaded.metadata["iteration"] == iteration
        assert loaded.metadata["training_schedule"] == "alternating_idm_then_planner"
        assert loaded.metadata["scenario_allocations"] == {
            "walk": 6,
            "static_stand": 3,
            "walk_to_stand": 3,
        }
        assert loaded.batch.command.shape[0] == 36
        assert int(loaded.batch.planner_eligible.sum()) == 12
        assert int((loaded.batch.command_scenario == FADA_SCENARIO_IDS["walk"]).sum()) == 30
        assert int(
            (loaded.batch.command_scenario == FADA_SCENARIO_IDS["static_stand"]).sum()
        ) == 3
        assert int(
            (loaded.batch.command_scenario == FADA_SCENARIO_IDS["walk_to_stand"]).sum()
        ) == 3
        main_roles = loaded.batch.idm_source_role[:12]
        oracle_shadow_count = int(
            (main_roles == FADA_IDM_SOURCE_ROLE_IDS["oracle_shadow"]).sum()
        )
        assert oracle_shadow_count == (12 if iteration == 0 else 3)
        summaries = loaded.metadata["collections"]
        main = [item for item in summaries if item["source"] == "optimal_or_current_policy"]
        intermediate = [item for item in summaries if item["source"] == "intermediate_oracle"]
        assert {item["oracle_role"] for item in main} == {"unified"}
        assert {item["rollout_mode"] for item in main} == {
            "oracle" if iteration == 0 else "planner_idm"
        }
        assert len(intermediate) == 20

    restored_policy = FADAPlannerIDMPolicy(architecture)
    checkpoint = Path(cfg.training.fada.checkpoint_path)
    payload = load_fada_checkpoint(checkpoint, restored_policy)
    assert payload["schema_version"] == FADA_CHECKPOINT_SCHEMA_VERSION == 5
    assert payload["training_schedule"] == "alternating_idm_then_planner"
    assert "optimizer_state_dict" not in payload
    assert "idm_optimizer_state_dict" in payload
    assert "planner_optimizer_state_dict" in payload
    assert payload["idm_optimizer_state_dict"]["state"]
    assert payload["planner_optimizer_state_dict"]["state"]
    assert payload["completed_iterations"] == 3
    assert payload["samples_seen"] == 108
    action = restored_policy.explore(
        torch.zeros(1, 30, 66),
        torch.zeros(1, 30, 29),
        torch.zeros(1, 3),
    )
    assert action.shape == (1, 29)
    assert torch.isfinite(action).all()
    assert not checkpoint.with_suffix(checkpoint.suffix + ".tmp").exists()
