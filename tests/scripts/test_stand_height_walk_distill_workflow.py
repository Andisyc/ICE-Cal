from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import scripts.train_distill as train_distill
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[2]
_CONF_DIR = _ROOT / "conf" / "distill"


def _compose(overrides: list[str]):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONF_DIR), version_base="1.3"):
        return compose("config", overrides=overrides)


def _set_new_workflow_env(monkeypatch: pytest.MonkeyPatch, root: Path) -> tuple[Path, Path]:
    walk_teacher = root / "walk_height_teacher.pt"
    stand_teacher = root / "stand_height_teacher.pt"
    walk_teacher.write_bytes(b"synthetic-walk-teacher")
    stand_teacher.write_bytes(b"synthetic-stand-teacher")
    monkeypatch.setenv("UNILAB_G1_WALK_HEIGHT_TEACHER", str(walk_teacher))
    monkeypatch.setenv("UNILAB_G1_STAND_HEIGHT_TEACHER", str(stand_teacher))
    monkeypatch.setenv("UNILAB_G1_WALK_HEIGHT_DATASET", "")
    monkeypatch.setenv("UNILAB_G1_STAND_HEIGHT_DATASET", "")
    return walk_teacher, stand_teacher


def test_new_two_expert_workflow_composes_without_changing_legacy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_new_workflow_env(monkeypatch, tmp_path)
    monkeypatch.setenv("UNILAB_G1_WALK_TEACHER", "/legacy/walk.pt")
    monkeypatch.setenv("UNILAB_G1_STAND_TEACHER", "/legacy/stand.pt")
    monkeypatch.setenv("UNILAB_G1_WALK_DATASET", "")
    monkeypatch.setenv("UNILAB_G1_STAND_DATASET", "")

    new_cfg = _compose(["workflow=g1_stand_height_walk"])
    legacy_cfg = _compose(["workflow=g1_walk_stand"])

    new_roles = OmegaConf.to_container(new_cfg.training.workflow.roles, resolve=True)
    assert new_cfg.training.task_name == "G1StandHeightWalk"
    assert new_cfg.teacher.obs_dim == 99
    assert new_cfg.student.obs_dim == 99
    assert new_cfg.student.action_dim == 29
    assert new_cfg.student.model_type == "moe"
    assert new_cfg.student.num_experts == 2
    assert dict(new_cfg.algo.role_expert_targets) == {"walk": 0, "stand_height": 1}
    assert dict(new_cfg.algo.command_intent_expert_targets) == {
        "active": 0,
        "inactive": 1,
    }
    assert [entry["role"] for entry in new_roles] == ["walk", "stand_height"]
    assert new_cfg.training.workflow.schema_version == 2
    assert OmegaConf.to_container(
        new_cfg.training.workflow.transition_walk_commands,
        resolve=True,
    ) == [
        [0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, 0.0, 0.4],
    ]
    assert new_cfg.training.workflow.transition_walk_target_height == pytest.approx(0.754)
    assert new_cfg.training.workflow.transition_nominal_settle_steps == 100
    assert list(new_cfg.training.workflow.transition_post_switch_target_heights) == pytest.approx(
        [0.650, 0.702, 0.754]
    )

    assert legacy_cfg.teacher.obs_dim == 98
    assert legacy_cfg.student.obs_dim == 98
    assert legacy_cfg.student.num_experts == 3
    assert dict(legacy_cfg.algo.role_expert_targets) == {"walk_flat": 0, "stand": 1}
    assert [entry.role for entry in legacy_cfg.training.workflow.roles] == [
        "walk_flat",
        "stand",
    ]
    assert list(legacy_cfg.training.workflow.transition_walk_commands) == []
    assert legacy_cfg.training.workflow.transition_walk_target_height is None
    assert legacy_cfg.training.workflow.transition_nominal_settle_steps == 0
    assert list(legacy_cfg.training.workflow.transition_post_switch_target_heights) == []


def test_height_role_owner_profiles_enforce_99d_and_nominal_walk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_new_workflow_env(monkeypatch, tmp_path)
    cfg = _compose(["workflow=g1_stand_height_walk"])
    entries = train_distill._workflow_role_entries(cfg)
    role_cfgs = {entry["role"]: train_distill._workflow_role_cfg(cfg, entry) for entry in entries}

    walk = role_cfgs["walk"]
    stand_height = role_cfgs["stand_height"]
    assert walk.training.task_name == "G1WalkHeight"
    assert walk.teacher.obs_dim == walk.student.obs_dim == 99
    assert walk.training.collect_command_sample_filter == "active"
    assert walk.training.collect_target_height_info_key == "height_commands"
    assert walk.env.commands.height_range == [0.754, 0.754]
    assert walk.env.commands.default_height == pytest.approx(0.754)
    assert walk.env.commands.random_height_during_walking is False
    assert stand_height.training.task_name == "G1StandHeight"
    assert stand_height.teacher.obs_dim == stand_height.student.obs_dim == 99
    assert stand_height.training.collect_command_sample_filter == "inactive"
    assert stand_height.training.collect_target_height_info_key == "height_commands"
    assert stand_height.env.commands.vel_limit == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    train_distill._require_teacher_policy_collection_route(walk)
    train_distill._require_teacher_policy_collection_route(stand_height)
    ordinary_walk = _compose(["task=g1_walk_height/mujoco"])
    with pytest.raises(ValueError, match="collect_target_height_info_key"):
        train_distill._require_teacher_policy_collection_route(ordinary_walk)


def test_single_entry_connector_builds_two_height_aware_specs_without_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_new_workflow_env(monkeypatch, tmp_path)
    cfg = _compose(
        [
            "workflow=g1_stand_height_walk",
            "training.workflow.enabled=true",
            "training.workflow.dagger_iterations=1",
        ]
    )
    cfg.training.workflow.run_dir = str(tmp_path / "run")
    cfg.training.workflow.artifact_dir = str(tmp_path / "artifacts")
    captured: dict[str, Any] = {}

    def fake_bootstrap(**kwargs):
        captured["bootstrap"] = kwargs
        run_dir = Path(kwargs["run_dir"])
        return SimpleNamespace(
            run_dir=run_dir,
            manifest_path=run_dir / "run_manifest.json",
            role_decisions={"walk": "COLLECT", "stand_height": "COLLECT"},
            bootstrap_dataset_path=run_dir / "datasets" / "bootstrap_merged.pt",
            bootstrap_num_samples=4,
            checkpoint_path=run_dir / "checkpoints" / "bootstrap_student.pt",
            bootstrap_updates=1,
        )

    def fake_dagger(**kwargs):
        captured["dagger"] = kwargs
        run_dir = Path(kwargs["run_dir"])
        return SimpleNamespace(
            run_dir=run_dir,
            manifest_path=run_dir / "run_manifest.json",
            completed_iterations=1,
            checkpoint_path=run_dir / "checkpoints" / "dagger_iteration_1.pt",
            cumulative_num_samples=8,
        )

    monkeypatch.setattr(train_distill, "run_bootstrap_workflow", fake_bootstrap)
    monkeypatch.setattr(train_distill, "run_multirole_dagger_workflow", fake_dagger)
    monkeypatch.setattr(train_distill, "finalize_workflow_performance", lambda **_kwargs: None)

    result = train_distill.run_single_entry_workflow(cfg)

    specs = captured["bootstrap"]["role_specs"]
    assert [(spec.role, spec.student_obs_dim, spec.teacher_obs_dim) for spec in specs] == [
        ("walk", 99, 99),
        ("stand_height", 99, 99),
    ]
    assert [spec.target_height_info_key for spec in specs] == [
        "height_commands",
        "height_commands",
    ]
    scenarios = captured["bootstrap"]["scenario_specs"]
    assert [(scenario.name, scenario.source_roles) for scenario in scenarios] == [
        ("walk_flat", ("walk",)),
        ("static_stand", ("stand_height",)),
        ("walk_to_stop", ("walk", "stand_height")),
    ]
    assert callable(captured["dagger"]["collect_scenario"])
    assert result["checkpoint_path"].endswith("dagger_iteration_1.pt")


def test_legacy_transition_connector_forwards_non_nominal_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_new_workflow_env(monkeypatch, tmp_path)
    cfg = _compose(
        [
            "workflow=g1_stand_height_walk",
            "training.workflow.enabled=true",
            "training.workflow.dagger_iterations=1",
        ]
    )
    cfg.training.workflow.run_dir = str(tmp_path / "run")
    cfg.training.workflow.artifact_dir = str(tmp_path / "artifacts")
    captured: dict[str, Any] = {}

    def fake_bootstrap(**kwargs):
        run_dir = Path(kwargs["run_dir"])
        return SimpleNamespace(
            run_dir=run_dir,
            manifest_path=run_dir / "run_manifest.json",
            role_decisions={"walk": "COLLECT", "stand_height": "COLLECT"},
            bootstrap_dataset_path=run_dir / "datasets" / "bootstrap_merged.pt",
            bootstrap_num_samples=4,
            checkpoint_path=run_dir / "checkpoints" / "bootstrap_student.pt",
            bootstrap_updates=1,
        )

    def fake_dagger(**kwargs):
        scenario = next(item for item in kwargs["scenario_specs"] if item.name == "walk_to_stop")
        captured["collection_result"] = kwargs["collect_scenario"](
            scenario,
            tmp_path / "student.pt",
            1,
            tmp_path / "walk_to_stop.pt",
        )
        run_dir = Path(kwargs["run_dir"])
        return SimpleNamespace(
            run_dir=run_dir,
            manifest_path=run_dir / "run_manifest.json",
            completed_iterations=1,
            checkpoint_path=run_dir / "checkpoints" / "dagger_iteration_1.pt",
            cumulative_num_samples=27,
        )

    class FakeEnv:
        closed = False

        def close(self) -> None:
            self.closed = True

    fake_env = FakeEnv()

    class FakeAdapter:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def build_task_env_cfg_override(self) -> dict:
            return {"scene": {"model_file": "fake.xml"}}

    def fake_transition_collect(_env, **kwargs):
        captured["transition"] = kwargs
        observations = [
            train_distill.DistillationStageObservation(
                stage=stage,
                duration_seconds=0.0,
                row_count=27 if stage != "env_step" else 0,
                env_step_count=2 if stage == "env_step" else 0,
                success=True,
                error=None,
                cleanup_state="not_applicable",
            ).as_dict()
            for stage in train_distill.COLLECTOR_REQUEST_STAGE_NAMES
        ]
        return SimpleNamespace(
            num_samples=27,
            metadata={
                "env_steps": 2,
                "performance_stage_observations": observations,
            },
        )

    monkeypatch.setattr(train_distill, "run_bootstrap_workflow", fake_bootstrap)
    monkeypatch.setattr(train_distill, "run_multirole_dagger_workflow", fake_dagger)
    monkeypatch.setattr(
        train_distill,
        "load_distillation_student_policy",
        lambda *_args, **_kwargs: SimpleNamespace(policy=object(), distill_runtime_cfg={}),
    )
    monkeypatch.setattr(
        train_distill,
        "load_sac_teacher_policy",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(train_distill, "ensure_registries", lambda: None)
    monkeypatch.setattr(train_distill, "BackendAdapter", FakeAdapter)
    monkeypatch.setattr(train_distill, "create_env", lambda *_args, **_kwargs: fake_env)
    monkeypatch.setattr(
        train_distill,
        "collect_transition_distillation_dataset_from_env",
        fake_transition_collect,
    )
    monkeypatch.setattr(train_distill, "save_distillation_dataset", lambda *_args: None)
    monkeypatch.setattr(train_distill, "finalize_workflow_performance", lambda **_kwargs: None)

    train_distill.run_single_entry_workflow(cfg)

    transition = captured["transition"]
    assert transition["walk_commands"] == [
        [0.4, 0.0, 0.0],
        [0.0, 0.4, 0.0],
        [0.0, 0.0, 0.4],
    ]
    assert transition["nominal_walk_target_height"] == pytest.approx(0.754)
    assert transition["nominal_settle_steps"] == 100
    assert transition["post_switch_target_heights"] == pytest.approx([0.650, 0.702, 0.754])
    assert transition["target_height_info_key"] == "height_commands"
    assert transition["expected_student_obs_dim"] == 99
    assert transition["expected_teacher_obs_dim"] == 99
    assert captured["collection_result"].num_samples == 27
    assert fake_env.closed is True


def test_persistent_connector_preserves_two_height_aware_roles_and_scenarios(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_new_workflow_env(monkeypatch, tmp_path)
    cfg = _compose(
        [
            "workflow=g1_stand_height_walk",
            "training.workflow.enabled=true",
            "training.workflow.execution_mode=persistent_async",
            "training.workflow.dagger_iterations=1",
        ]
    )
    cfg.training.workflow.run_dir = str(tmp_path / "persistent_run")
    cfg.training.workflow.artifact_dir = str(tmp_path / "artifacts")
    captured: dict[str, Any] = {}

    def fake_bootstrap(**kwargs):
        run_dir = Path(kwargs["run_dir"])
        return SimpleNamespace(
            run_dir=run_dir,
            manifest_path=run_dir / "run_manifest.json",
            role_decisions={"walk": "COLLECT", "stand_height": "COLLECT"},
            bootstrap_dataset_path=run_dir / "datasets" / "bootstrap_merged.pt",
            bootstrap_num_samples=4,
            checkpoint_path=run_dir / "checkpoints" / "bootstrap_student.pt",
            bootstrap_updates=1,
        )

    class FakePersistentService:
        close_report = {
            "worker_pid": 1234,
            "resource_counters": {"env_builds": 2},
        }

        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    service = FakePersistentService()

    def fake_factory(**kwargs):
        captured["factory"] = kwargs
        return service

    def fake_dagger(**kwargs):
        captured["dagger"] = kwargs
        run_dir = Path(kwargs["run_dir"])
        return SimpleNamespace(
            run_dir=run_dir,
            manifest_path=run_dir / "run_manifest.json",
            completed_iterations=1,
            checkpoint_path=run_dir / "checkpoints" / "dagger_iteration_1.pt",
            cumulative_num_samples=8,
        )

    sentinel_events: list[str] = []
    monkeypatch.setattr(train_distill, "run_bootstrap_workflow", fake_bootstrap)
    monkeypatch.setattr(train_distill, "run_multirole_dagger_workflow", fake_dagger)
    monkeypatch.setattr(
        train_distill,
        "_probe_torch_serialization_runtime",
        sentinel_events.append,
    )
    monkeypatch.setattr(train_distill, "finalize_workflow_performance", lambda **_kwargs: None)

    result = train_distill.run_single_entry_workflow(
        cfg,
        persistent_scenario_collector_factory=fake_factory,
    )

    factory_inputs = captured["factory"]
    specs = factory_inputs["role_specs"]
    assert [(spec.role, spec.student_obs_dim, spec.teacher_obs_dim) for spec in specs] == [
        ("walk", 99, 99),
        ("stand_height", 99, 99),
    ]
    assert [spec.target_height_info_key for spec in specs] == [
        "height_commands",
        "height_commands",
    ]
    assert set(factory_inputs["role_cfgs"]) == {"walk", "stand_height"}
    scenarios = factory_inputs["scenario_specs"]
    assert [(scenario.name, scenario.source_roles) for scenario in scenarios] == [
        ("walk_flat", ("walk",)),
        ("static_stand", ("stand_height",)),
        ("walk_to_stop", ("walk", "stand_height")),
    ]
    dagger_inputs = captured["dagger"]
    assert dagger_inputs["execution_mode"] == "persistent_async"
    assert dagger_inputs["collect_scenario"] is None
    assert dagger_inputs["scenario_collector"] is service
    assert dagger_inputs["performance_context"].execution_mode == "persistent_async"
    assert sentinel_events
    assert service.close_calls == 1
    assert result["execution_mode"] == "persistent_async"
