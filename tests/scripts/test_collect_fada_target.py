from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from unilab.algos.torch.distill import FADAArchitectureConfig
from unilab.algos.torch.distill.fada.target_data import FADATargetBatch, load_fada_target_artifact

ROOT_DIR = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT_DIR / "conf" / "offpolicy"
SCRIPT_PATH = ROOT_DIR / "scripts" / "collect_fada_target.py"


def _owner() -> Any:
    return importlib.import_module("unilab.algos.torch.distill.fada.target_workflow")


def _compose(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(
            config_name="fada_target", overrides=list(overrides), return_hydra_config=True
        )


def _compose_slope(*overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(
            config_name="fada_slope_target",
            overrides=list(overrides),
            return_hydra_config=True,
        )


def _config() -> FADAArchitectureConfig:
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


def _batch(config: FADAArchitectureConfig, value: float) -> FADATargetBatch:
    rows = 8
    return FADATargetBatch(
        observation_history=torch.full((rows, 2, 66), value),
        action_history=torch.full((rows, 2, 29), value),
        command=torch.tensor([[0.4, 0.0, 0.0]] * rows),
        realized_future=torch.full((rows, 2, 66), value),
        executed_action_chunk=torch.full((rows, 2, 29), value),
        episode_id=torch.zeros(rows, dtype=torch.int64),
        start_timestep=torch.arange(rows, dtype=torch.int64),
    ).validate(config)


def test_target_config_has_one_paired_bundle_mode() -> None:
    cfg = _compose()
    assert cfg.hydra.runtime.choices.collection == "fada_target"
    assert cfg.collection.policy_checkpoint_path.endswith("planner_idm_v022_cpu_limited.pt")
    assert cfg.collection.output_dir.endswith("g1_walk_flat_mujoco_left_knee_090")
    assert cfg.collection.single_trajectory is True
    assert cfg.collection.record_video is True
    assert list(cfg.collection.command_target) == [0.4, 0.0, 0.0]
    assert cfg.collection.ramp_steps == 25
    assert cfg.collection.settle_steps == 50
    assert cfg.collection.control_steps == 6000
    assert cfg.env.gait_phase_enabled is False
    assert cfg.env.mode_observation is False
    assert cfg.reward.scales.feet_phase == 0.0
    assert not (CONF_DIR / "collection" / "fada_target_paired.yaml").exists()


def test_slope_config_selects_nominal_target_only_collection() -> None:
    cfg = _compose_slope()

    assert cfg.hydra.runtime.choices.task == "sac/g1_walk_flat/mujoco_fada_slope_15"
    assert cfg.target_domain.target_domain_id == "g1_slope_15_mujoco"
    assert cfg.collection.output_dir.endswith("g1_slope_15_mujoco")
    assert cfg.env.scene.model_file.endswith("scene_slope_15.xml")
    assert cfg.env.noise_config.level == 0.0
    assert cfg.env.domain_rand.actuator_strength.enabled is False
    assert list(cfg.env.domain_rand.actuator_strength.multipliers) == []


def test_slope_env_overrides_merge_into_the_structured_g1_owner() -> None:
    from unilab.envs.locomotion.g1.joystick import G1WalkFlatCfg

    cfg = _compose_slope()
    merged = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(G1WalkFlatCfg()), cfg.env))

    assert isinstance(merged, G1WalkFlatCfg)
    assert merged.scene.model_file.endswith("scene_slope_15.xml")
    assert merged.noise_config.level == 0.0
    assert merged.domain_rand.actuator_strength.enabled is False
    assert merged.domain_rand.actuator_strength.multipliers == []


def test_stage_c_composition_root_uses_the_deployable_checkpoint_reader() -> None:
    owner = _owner()
    default_loader = (
        inspect.signature(owner.run_fada_target_collection).parameters["load_policy_fn"].default
    )

    assert default_loader is owner.load_fada_deployable_policy_checkpoint


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_second: bool = False,
    checkpoint_schema: int | str = 5,
    control_steps: int = 10,
) -> tuple[dict[str, Any], list[Any]]:
    owner = _owner()
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "bundle"
    cfg = _compose(
        f"collection.policy_checkpoint_path={checkpoint}",
        f"collection.expected_checkpoint_sha256={owner.file_sha256(checkpoint)}",
        f"collection.output_dir={output}",
        f"collection.control_steps={control_steps}",
        "collection.ramp_steps=0",
        "collection.settle_steps=0",
    )
    config = _config()
    policy = SimpleNamespace(config=config)
    envs: list[Any] = []

    class Env:
        def __init__(self, nominal: bool) -> None:
            self.nominal = nominal
            self.cfg = SimpleNamespace(ctrl_dt=0.02)
            self.closed = False
            self.position = np.zeros((1, 3), dtype=np.float32)
            self.yaw = 0.0
            self.state = SimpleNamespace(
                info={
                    "episode_start_base_pos": np.zeros((1, 3), dtype=np.float32),
                    "episode_start_base_yaw": np.zeros((1,), dtype=np.float32),
                }
            )
            self.play_capabilities = SimpleNamespace(supports_physics_state_playback=True)

        def get_physics_state_snapshot(self) -> np.ndarray:
            return np.zeros((1, 4), dtype=np.float32)

        def get_base_pos(self) -> np.ndarray:
            return self.position.copy()

        def get_base_quat(self) -> np.ndarray:
            return np.asarray(
                [[np.cos(self.yaw / 2.0), 0.0, 0.0, np.sin(self.yaw / 2.0)]],
                dtype=np.float32,
            )

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        importlib.import_module("unilab.algos.torch.distill.fada.path_capture"),
        "render_mujoco_states_video",
        lambda *, output_video, **_kwargs: Path(output_video).write_bytes(b"video"),
    )

    def create_env(_cfg: Any, *, env_cfg_override: dict[str, Any], **_kwargs: Any) -> Env:
        multipliers = env_cfg_override["domain_rand"]["actuator_strength"]["multipliers"]
        env = Env(all(float(v) == 1.0 for v in multipliers))
        envs.append(env)
        return env

    calls = 0

    def collect(env: Env, *_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if fail_second and calls == 2:
            raise RuntimeError("fault branch failed")
        lateral = [0.0, 0.1, -0.1] if env.nominal else [0.0, 0.4, 0.6]
        spec = _args[-1]
        spec.capture_initial_frame()
        for step, value in enumerate(lateral):
            env.position[0] = [float(step), value, 0.8]
            env.yaw = 0.01 * step if env.nominal else 0.05 * step
            spec.capture_frame()
        return SimpleNamespace(
            batch=_batch(config, 0.0 if env.nominal else 1.0),
            env_steps=10,
            rejected_done_transitions=0,
            rejected_command_windows=0,
        )

    result = owner.run_fada_target_collection(
        cfg,
        root_dir=ROOT_DIR,
        load_policy_fn=lambda *_a, **_k: SimpleNamespace(
            policy=policy, checkpoint={"schema_version": checkpoint_schema}
        ),
        ensure_registries_fn=lambda: None,
        create_env_fn=create_env,
        collect_fn=collect,
    )
    return result, envs


def test_stage_c_accepts_an_adapted_policy_for_post_lora_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _envs = _run(
        tmp_path,
        monkeypatch,
        checkpoint_schema="fada-adapted/v3",
    )

    assert result["status"] == "completed"


def test_slope_stage_c_publishes_only_target_bundle_outputs(tmp_path: Path) -> None:
    owner = importlib.import_module("unilab.algos.torch.distill.fada.target_slope_workflow")
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "slope_bundle"
    cfg = _compose_slope(
        f"collection.policy_checkpoint_path={checkpoint}",
        f"collection.output_dir={output}",
        "collection.control_steps=10",
        "collection.max_env_steps=20",
        "collection.ramp_steps=0",
        "collection.settle_steps=0",
    )
    config = _config()
    policy = SimpleNamespace(config=config)

    class Env:
        cfg = SimpleNamespace(ctrl_dt=0.02)
        play_capabilities = SimpleNamespace(supports_physics_state_playback=True)

        def close(self) -> None:
            pass

    def collect(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            batch=_batch(config, 1.0),
            env_steps=12,
            accepted_steps=10,
            episode_count=2,
            rejected_pre_entry_steps=2,
            rejected_command_windows=0,
            termination_counts={
                "fall": 0,
                "environment_termination": 1,
                "truncated": 0,
                "foot_exit": 0,
                "finish": 0,
            },
            representative_physics_states=(np.zeros((1, 4), dtype=np.float32),),
        )

    rendered: dict[str, Any] = {}

    def render(**kwargs: Any) -> str:
        rendered.update(kwargs)
        return str(Path(kwargs["output_video"]).write_bytes(b"video"))

    result = owner.run_fada_slope_collection(
        cfg,
        root_dir=ROOT_DIR,
        load_policy_fn=lambda *_a, **_k: SimpleNamespace(
            policy=policy, checkpoint={"schema_version": 5}
        ),
        ensure_registries_fn=lambda: None,
        create_env_fn=lambda *_a, **_k: Env(),
        collect_fn=collect,
        render_fn=render,
    )

    assert result["status"] == "completed"
    assert {path.name for path in output.iterdir()} == {
        "target.pt",
        "collection.mp4",
        "collection_summary.json",
        "manifest.json",
    }
    assert (
        load_fada_target_artifact(output / "target.pt", config=config).metadata["target_domain_id"]
        == "g1_slope_15_mujoco"
    )
    assert rendered["camera_kwargs"] == {
        "cam_tracking": True,
        "cam_tracking_env_idx": 0,
    }


def test_stage_c_rejects_legacy_adapter_and_short_budget_before_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="schema-5 source or fada-adapted/v3"):
        _run(
            tmp_path,
            monkeypatch,
            checkpoint_schema="fada-adapted/v2",
        )
    with pytest.raises(ValueError, match="control_steps.*usable window"):
        _run(
            tmp_path,
            monkeypatch,
            control_steps=2,
        )


def test_stage_c_env_override_disables_all_non_fault_randomization() -> None:
    owner = _owner()
    cfg = _compose()

    nominal = owner._env_override(cfg, nominal=True, root=ROOT_DIR)
    faulty = owner._env_override(cfg, nominal=False, root=ROOT_DIR)

    disabled_flags = {
        "randomize_reset_pose",
        "randomize_kp",
        "randomize_kd",
        "randomize_ground_friction",
        "randomize_base_mass",
        "randomize_body_mass",
        "random_com",
        "randomize_gravity",
        "randomize_dof_armature",
        "randomize_dof_position_bias",
        "randomize_control_delay",
        "push_robots",
    }
    for override in (nominal, faulty):
        assert override["noise_config"]["level"] == 0.0
        assert all(override["domain_rand"][name] is False for name in disabled_flags)
        assert override["domain_rand"]["torque_rfi_fraction"] == 0.0
    assert nominal["domain_rand"]["actuator_strength"]["multipliers"] == [1.0] * 29
    assert faulty["domain_rand"]["actuator_strength"]["multipliers"][3] == 0.9


def test_right_knee_fault_is_a_config_owned_mirror() -> None:
    owner = _owner()
    left = _compose()
    right = _compose("fault=right_knee_090")

    assert left.fault.task == right.fault.task == "sac/g1_walk_flat/mujoco_fada_target"
    assert left.fault.actuator_index == 3
    assert right.fault.actuator_index == 9
    assert right.collection.output_dir.endswith("g1_walk_flat_mujoco_right_knee_090")
    assert list(right.collection.command_target) == [0.8, 0.0, 0.0]
    assert [list(limit) for limit in right.fault.command_limit] == [
        [0.8, 0.0, 0.0],
        [0.8, 0.0, 0.0],
    ]
    assert OmegaConf.to_container(right.env.commands.vel_limit, resolve=True) == [
        [0.8, 0.0, 0.0],
        [0.8, 0.0, 0.0],
    ]
    owner._assert_identity(right)

    left_multipliers = owner._env_override(left, nominal=False, root=ROOT_DIR)["domain_rand"][
        "actuator_strength"
    ]["multipliers"]
    right_multipliers = owner._env_override(right, nominal=False, root=ROOT_DIR)["domain_rand"][
        "actuator_strength"
    ]["multipliers"]
    assert [index for index, value in enumerate(left_multipliers) if value != 1.0] == [3]
    assert [index for index, value in enumerate(right_multipliers) if value != 1.0] == [9]
    assert left_multipliers[3] == right_multipliers[9] == 0.9


def test_one_call_atomically_publishes_complete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, envs = _run(tmp_path, monkeypatch)
    bundle = Path(result["bundle_dir"])
    assert {p.name for p in bundle.iterdir()} == {
        "nominal.pt",
        "faulty.pt",
        "delta.pt",
        "nominal.mp4",
        "faulty.mp4",
        "path_deviation.json",
        "manifest.json",
    }
    assert all(env.closed for env in envs) and len(envs) == 2
    loaded = load_fada_target_artifact(bundle / "faulty.pt", config=_config())
    assert loaded.metadata["fault_profile"] == "left_knee_strength_0.9"
    delta = torch.load(bundle / "delta.pt", weights_only=True)
    torch.testing.assert_close(delta["delta"]["observation_history"], torch.ones(8, 2, 66))
    deviation = json.loads((bundle / "path_deviation.json").read_text())
    assert deviation["reference_line"]["measurement_start_step"] == 0
    assert deviation["nominal"]["max_abs_lateral_m"] == pytest.approx(0.1)
    assert deviation["faulty"]["max_abs_lateral_m"] == pytest.approx(0.6)
    assert deviation["excess"]["max_abs_lateral_m"] == pytest.approx(0.5)
    assert deviation["nominal"]["yaw_rad"] == pytest.approx([0.0, 0.0, 0.01, 0.02])
    assert deviation["faulty"]["yaw_rad"] == pytest.approx([0.0, 0.0, 0.05, 0.1])
    assert deviation["faulty"]["yaw_drift_rad"] == pytest.approx([0.0, 0.0, 0.05, 0.1])
    assert result["path_deviation_path"] == str(bundle / "path_deviation.json")


def test_branch_failure_closes_envs_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match="fault branch failed"):
        _run(tmp_path, monkeypatch, fail_second=True)
    assert not (tmp_path / "bundle").exists()


def test_paired_batches_align_to_common_time_prefix() -> None:
    owner = _owner()
    config = _config()
    nominal = _batch(config, 0.0)
    faulty = owner._slice_target_batch(_batch(config, 1.0), 5, config)

    aligned_nominal, aligned_faulty = owner._align_paired_batches(nominal, faulty, config)

    assert aligned_nominal.observation_history.shape[0] == 5
    assert aligned_faulty.observation_history.shape[0] == 5
    torch.testing.assert_close(aligned_nominal.start_timestep, aligned_faulty.start_timestep)


def test_delta_rejects_unpaired_rows_before_publication(tmp_path: Path) -> None:
    owner = _owner()
    config = _config()
    nominal = _batch(config, 0.0)
    faulty = _batch(config, 1.0)
    faulty.start_timestep[0] = 99
    with pytest.raises(ValueError, match="row identity mismatch: start_timestep"):
        owner._save_delta(tmp_path / "delta.pt", nominal, faulty, {})
    assert not (tmp_path / "delta.pt").exists()


def test_preflight_accepts_missing_expected_hash_and_records_observed_hash(tmp_path: Path) -> None:
    owner = _owner()
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = _compose(
        f"collection.policy_checkpoint_path={checkpoint}",
        "collection.expected_checkpoint_sha256=null",
        f"collection.output_dir={tmp_path / 'bundle'}",
    )

    preflight = owner.preflight_fada_target_collection(cfg, root_dir=ROOT_DIR)

    assert preflight.checkpoint_sha256 == owner.file_sha256(checkpoint)


def test_preflight_rejects_wrong_explicit_hash(tmp_path: Path) -> None:
    owner = _owner()
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = _compose(
        f"collection.policy_checkpoint_path={checkpoint}",
        f"collection.expected_checkpoint_sha256={'0' * 64}",
        f"collection.output_dir={tmp_path / 'bundle'}",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        owner.preflight_fada_target_collection(cfg, root_dir=ROOT_DIR)


def test_preflight_rejects_existing_bundle(tmp_path: Path) -> None:
    owner = _owner()
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"x")
    output = tmp_path / "bundle"
    output.mkdir()
    cfg = _compose(
        f"collection.policy_checkpoint_path={checkpoint}",
        f"collection.expected_checkpoint_sha256={owner.file_sha256(checkpoint)}",
        f"collection.output_dir={output}",
    )
    with pytest.raises(FileExistsError, match="bundle already exists"):
        owner.preflight_fada_target_collection(cfg, root_dir=ROOT_DIR)


def test_cli_is_thin_composition_root() -> None:
    source = SCRIPT_PATH.read_text()
    assert "run_fada_target_collection" in source
    assert "collect_fada_target_windows" not in source
    assert len(source.splitlines()) < 30
