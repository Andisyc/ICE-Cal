from __future__ import annotations

import os
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from unilab.algos.torch.distill.async_runtime import DaggerCollectRequest
from unilab.algos.torch.distill.data import build_distillation_dataset
from unilab.algos.torch.distill.g1_persistent_worker import (
    PersistentG1DistillationWorker,
)
from unilab.algos.torch.distill.performance import (
    DISTILLATION_METRICS_SCHEMA_VERSION,
    DistillationStageObservation,
)
from unilab.ipc import SharedWeightSync


class _Policy(torch.nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((obs.shape[0], 3), dtype=obs.dtype)


class _FakeClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class _Env:
    def __init__(self, num_envs: int) -> None:
        self.num_envs = num_envs
        self.reset_calls = 0
        self.close_calls = 0

    def reset(self, _indices):
        self.reset_calls += 1
        return {"obs": np.zeros((self.num_envs, 8), dtype=np.float32)}, {}

    def close(self) -> None:
        self.close_calls += 1


def _role_cfg(
    *,
    task_name: str,
    checkpoint: Path,
    command_filter: str,
    target_height_info_key: str | None = None,
) -> dict:
    return {
        "training": {
            "task_name": task_name,
            "sim_backend": "mujoco",
            "collect_teacher_obs_key": "obs",
            "collect_teacher_projection": "identity",
            "collect_student_projection": "identity",
            "collect_student_drop_index": None,
            "collect_command_sample_filter": command_filter,
            "collect_command_info_key": "commands",
            "collect_target_height_info_key": target_height_info_key,
            "collect_command_xy_threshold": 0.05,
            "collect_command_yaw_threshold": 0.05,
            "collect_max_env_steps": None,
        },
        "teacher": {
            "checkpoint_path": str(checkpoint),
            "obs_dim": 8,
            "action_dim": 3,
            "algo_type": "sac",
            "actor_hidden_dim": 16,
            "use_layer_norm": False,
            "obs_normalization": False,
        },
        "student": {"obs_dim": 8, "action_dim": 3},
        "env": {},
    }


def _request(tmp_path: Path, scenario: str, index: int) -> DaggerCollectRequest:
    return DaggerCollectRequest(
        request_id=f"req-{index}-{scenario}",
        scenario=scenario,
        iteration=1,
        checkpoint_path=str((tmp_path / "student.pt").resolve()),
        output_path=str((tmp_path / f"{index}-{scenario}.pt").resolve()),
        expected_weight_version=1,
    )


def _collector_performance_metadata(
    metadata: dict,
    *,
    num_samples: int,
    env_steps: int,
) -> dict:
    observations = (
        DistillationStageObservation(
            stage="teacher_inference",
            duration_seconds=0.4,
            row_count=num_samples,
            env_step_count=0,
            success=True,
            error=None,
            cleanup_state="not_applicable",
        ),
        DistillationStageObservation(
            stage="student_inference",
            duration_seconds=0.3,
            row_count=num_samples,
            env_step_count=0,
            success=True,
            error=None,
            cleanup_state="not_applicable",
        ),
        DistillationStageObservation(
            stage="env_step",
            duration_seconds=0.2,
            row_count=0,
            env_step_count=env_steps,
            success=True,
            error=None,
            cleanup_state="not_applicable",
        ),
        DistillationStageObservation(
            stage="tensor_pack",
            duration_seconds=0.1,
            row_count=num_samples,
            env_step_count=0,
            success=True,
            error=None,
            cleanup_state="not_applicable",
        ),
    )
    return {
        **metadata,
        "env_steps": env_steps,
        "performance_metrics_schema_version": DISTILLATION_METRICS_SCHEMA_VERSION,
        "performance_stage_observations": [item.as_dict() for item in observations],
    }


def test_persistent_runtime_builder_forwards_transition_grid(monkeypatch) -> None:
    import unilab.algos.torch.distill.g1_persistent_worker as worker_module

    captured: dict = {}

    class FakeRuntime:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(worker_module, "PersistentDistillationRuntime", FakeRuntime)
    monkeypatch.setattr(
        worker_module.mp,
        "get_context",
        lambda _method: SimpleNamespace(Queue=lambda **_kwargs: object()),
    )
    cfg = OmegaConf.create(
        {
            "training": {
                "device": "cpu",
                "workflow": {
                    "collect_num_envs": 64,
                    "dagger_samples_per_role": 65536,
                    "transition_pre_switch_steps": 8,
                    "transition_min_post_switch_steps": 20,
                    "transition_walk_command": [0.4, 0.0, 0.0],
                    "transition_walk_commands": [
                        [0.4, 0.0, 0.0],
                        [0.0, 0.4, 0.0],
                        [0.0, 0.0, 0.4],
                    ],
                    "transition_walk_target_height": 0.754,
                    "transition_post_switch_target_heights": [0.650, 0.702, 0.754],
                    "transition_max_env_steps": None,
                },
            }
        }
    )

    runtime = worker_module.build_persistent_g1_distillation_runtime(
        cfg=cfg,
        role_cfgs={"walk": OmegaConf.create({})},
        role_specs=[SimpleNamespace(role="walk", task="g1_walk_height_nominal/mujoco")],
        scenario_specs=[
            SimpleNamespace(
                as_dict=lambda: {
                    "name": "walk_to_stop",
                    "kind": "transition",
                    "source_roles": ["walk", "stand_height"],
                }
            )
        ],
    )

    assert isinstance(runtime, FakeRuntime)
    workflow_cfg = captured["worker_kwargs"]["workflow_cfg"]
    assert workflow_cfg["transition_walk_commands"] == [
        [0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, 0.0, 0.4],
    ]
    assert workflow_cfg["transition_walk_target_height"] == pytest.approx(0.754)
    assert workflow_cfg["transition_post_switch_target_heights"] == pytest.approx(
        [0.650, 0.702, 0.754]
    )


def test_g1_persistent_worker_reuses_exact_resources_across_scenario_sequence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import unilab.algos.torch.distill.g1_persistent_worker as worker_module

    walk_teacher = tmp_path / "walk_teacher.pt"
    stand_teacher = tmp_path / "stand_teacher.pt"
    student_checkpoint = tmp_path / "student.pt"
    walk_teacher.write_bytes(b"walk")
    stand_teacher.write_bytes(b"stand")
    student_checkpoint.write_bytes(b"student")
    policy = _Policy()
    monkeypatch.setattr(
        worker_module,
        "load_distillation_student_policy",
        lambda *_args, **_kwargs: SimpleNamespace(
            policy=policy,
            distill_runtime_cfg={},
        ),
    )
    monkeypatch.setattr(
        worker_module,
        "load_sac_teacher_policy",
        lambda *_args, **_kwargs: _Policy(),
    )
    monkeypatch.setattr(worker_module, "ensure_registries", lambda: None)

    class _Adapter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def build_task_env_cfg_override(self):
            return {"scene": {"model_file": "fake.xml"}}

    monkeypatch.setattr(worker_module, "BackendAdapter", _Adapter)
    created_envs: list[_Env] = []

    def create_env(_cfg, *, num_envs, **_kwargs):
        created_envs.append(_Env(num_envs))
        return created_envs[-1]

    monkeypatch.setattr(worker_module, "create_env", create_env)

    def role_collect(_env, *, num_samples, role_label, metadata, **kwargs):
        assert kwargs["initial_reset"] is not None
        intent = "active" if role_label == "walk_flat" else "inactive"
        commands = (
            torch.tensor([[0.4, 0.0, 0.0]] * num_samples)
            if intent == "active"
            else torch.zeros((num_samples, 3))
        )
        return build_distillation_dataset(
            torch.zeros((num_samples, 8)),
            torch.zeros((num_samples, 8)),
            expected_student_obs_dim=8,
            expected_teacher_obs_dim=8,
            expected_teacher_action_dim=3,
            role_labels=(role_label,) * num_samples,
            teacher_actions=torch.zeros((num_samples, 3)),
            commands=commands,
            command_intents=(intent,) * num_samples,
            metadata=_collector_performance_metadata(
                metadata,
                num_samples=num_samples,
                env_steps=2,
            ),
        )

    transition_inputs: dict = {}

    def transition_collect(_env, *, num_samples, metadata, **kwargs):
        assert kwargs["initial_reset"] is not None
        transition_inputs.update(kwargs)
        half = num_samples // 2
        return build_distillation_dataset(
            torch.zeros((num_samples, 8)),
            torch.zeros((num_samples, 8)),
            expected_student_obs_dim=8,
            expected_teacher_obs_dim=8,
            expected_teacher_action_dim=3,
            role_labels=("walk_flat",) * half + ("stand",) * half,
            teacher_actions=torch.zeros((num_samples, 3)),
            commands=torch.zeros((num_samples, 3)),
            command_intents=("active",) * half + ("inactive",) * half,
            scenario_labels=("walk_to_stop",) * num_samples,
            transition_ages=torch.tensor([-1] * half + list(range(half))),
            command_before=torch.tensor([[0.4, 0.0, 0.0]] * num_samples),
            command_after=torch.zeros((num_samples, 3)),
            metadata=_collector_performance_metadata(
                metadata,
                num_samples=num_samples,
                env_steps=3,
            ),
        )

    monkeypatch.setattr(
        worker_module,
        "collect_distillation_dataset_from_env",
        role_collect,
    )
    monkeypatch.setattr(
        worker_module,
        "collect_transition_distillation_dataset_from_env",
        transition_collect,
    )

    weight_sync = SharedWeightSync({})
    weight_sync.write_weights({})
    lifecycle_report_queue: queue.Queue = queue.Queue(maxsize=1)
    clock_values = [
        base + offset
        for base in (0.0, 10.0, 20.0, 30.0)
        for offset in (0.0, 1.0, 1.1, 2.0, 2.5, 3.0, 3.25, 4.0)
    ]
    worker = PersistentG1DistillationWorker(
        root_dir=str(tmp_path),
        role_cfgs={
            "walk_flat": _role_cfg(
                task_name="G1WalkFlat",
                checkpoint=walk_teacher,
                command_filter="active",
                target_height_info_key="height_commands",
            ),
            "stand": _role_cfg(
                task_name="G1StandStill",
                checkpoint=stand_teacher,
                command_filter="inactive",
                target_height_info_key="height_commands",
            ),
        },
        role_specs=[
            {"role": "walk_flat", "task": "g1_walk_flat/mujoco"},
            {"role": "stand", "task": "g1_stand_still/mujoco"},
        ],
        scenario_specs=[
            {"name": "walk_flat", "kind": "role", "source_roles": ["walk_flat"]},
            {"name": "static_stand", "kind": "role", "source_roles": ["stand"]},
            {
                "name": "walk_to_stop",
                "kind": "transition",
                "source_roles": ["walk_flat", "stand"],
            },
        ],
        workflow_cfg={
            "collect_num_envs": 2,
            "dagger_samples_per_role": 4,
            "transition_pre_switch_steps": 2,
            "transition_min_post_switch_steps": 0,
            "transition_walk_command": [0.4, 0.0, 0.0],
            "transition_walk_commands": [
                [0.4, 0.0, 0.0],
                [0.0, 0.4, 0.0],
                [0.0, 0.0, 0.4],
            ],
            "transition_walk_target_height": 0.754,
            "transition_post_switch_target_heights": [0.650, 0.702, 0.754],
            "transition_max_env_steps": None,
        },
        initial_checkpoint_path=str(student_checkpoint),
        device="cpu",
        weight_sync_name=weight_sync.name,
        weight_sync_lock=weight_sync._lock,
        weight_param_shapes={},
        lifecycle_report_queue=lifecycle_report_queue,
        clock=_FakeClock(clock_values),
    )
    try:
        results = [
            worker.collect(_request(tmp_path, scenario, index))
            for index, scenario in enumerate(
                ("walk_flat", "static_stand", "walk_to_stop", "walk_flat")
            )
        ]
        assert all(result.worker_pid == os.getpid() for result in results)
        assert [result.metrics["student_init_count"] for result in results] == [1.0] * 4
        assert [result.metrics["env_init_count"] for result in results] == [1.0, 2.0, 2.0, 2.0]
        assert [result.metrics["teacher_init_count"] for result in results] == [
            1.0,
            2.0,
            2.0,
            2.0,
        ]
        assert [result.metrics["request_reset_count"] for result in results] == [1.0] * 4
        assert [result.metadata["performance_metrics_schema_version"] for result in results] == [
            DISTILLATION_METRICS_SCHEMA_VERSION
        ] * 4
        observations = [
            tuple(
                DistillationStageObservation.from_dict(payload)
                for payload in result.metadata["performance_stage_observations"]
            )
            for result in results
        ]
        assert [tuple(item.stage for item in request) for request in observations] == [
            (
                "weight_sync",
                "teacher_inference",
                "student_inference",
                "env_step",
                "tensor_pack",
                "artifact_write",
                "total_elapsed",
            )
        ] * 4
        assert [tuple(item.duration_seconds for item in request) for request in observations] == [
            pytest.approx((0.1, 0.4, 0.3, 0.2, 0.1, 0.25, 4.0))
        ] * 4
        assert [tuple(item.row_count for item in request) for request in observations] == [
            (0, 4, 4, 0, 4, 4, 4)
        ] * 4
        assert [observations[index][-1].env_step_count for index in range(4)] == [
            2,
            2,
            3,
            2,
        ]
        assert [observations[index][-1].cleanup_state for index in range(4)] == ["pending"] * 4
        assert [result.metrics["weight_sync_seconds"] for result in results] == [
            pytest.approx(0.1)
        ] * 4
        assert [result.metrics["artifact_write_seconds"] for result in results] == [
            pytest.approx(0.25)
        ] * 4
        assert transition_inputs["walk_commands"] == [
            [0.4, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 0.0, 0.4],
        ]
        assert transition_inputs["nominal_walk_target_height"] == pytest.approx(0.754)
        assert transition_inputs["post_switch_target_heights"] == pytest.approx(
            [0.650, 0.702, 0.754]
        )
        assert transition_inputs["target_height_info_key"] == "height_commands"
        assert len(worker.resources.cache_keys) == 2
    finally:
        worker.close()
        weight_sync.cleanup()

    assert len(created_envs) == 2
    assert [env.reset_calls for env in created_envs] == [3, 1]
    assert [env.close_calls for env in created_envs] == [1, 1]
    close_report = lifecycle_report_queue.get_nowait()
    assert close_report["student_init_count"] == 1
    assert close_report["resource_counters"]["teacher_close_count"] == 2
    assert close_report["resource_counters"]["env_close_count"] == 2
