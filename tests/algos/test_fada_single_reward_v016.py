from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unilab.algos.torch.distill import fada_privileged_oracle
from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
    resolve_privileged_locomotion_sac_runtime,
)

ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = ROOT / "conf/offpolicy"


def _compose(task: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "v016-unit-test-lineage")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose("config", overrides=["algo=sac", f"task={task}"])


def _assert_no_stand_authority(value: object, *, path: str = "reward") -> None:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        for key, child in value.items():
            assert not str(key).startswith("stand_"), f"forbidden {path}.{key}"
            _assert_no_stand_authority(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_stand_authority(child, path=f"{path}[{index}]")
    elif isinstance(value, str):
        assert not value.startswith("stand_"), f"forbidden term {value!r} at {path}"


def test_v016_nominal_profile_is_single_reward_phase_neutral_and_unprivileged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = CONF_DIR / "task/sac/g1_walk_flat/mujoco_no_gait_single_reward.yaml"
    assert profile.is_file(), "v016 nominal single-Reward profile is missing"
    cfg = _compose("sac/g1_walk_flat/mujoco_no_gait_single_reward", monkeypatch)

    assert cfg.algo.get("runtime_impl") is None
    assert cfg.env.gait_phase_enabled is False
    assert cfg.env.fada_privileged_observation.enabled is False
    assert cfg.env.domain_rand.actuator_strength.enabled is False
    assert not bool(cfg.reward.get("mode", {}).get("enabled", False))
    _assert_no_stand_authority(cfg.reward)
    assert cfg.reward.gait_constraint.enabled is False
    assert cfg.reward.gait_constraint.penalty_scale == pytest.approx(0.0)
    assert cfg.reward.scales.feet_phase == pytest.approx(0.0)
    assert cfg.reward.scales.feet_phase_contrast == pytest.approx(0.0)
    assert cfg.reward.scales.feet_phase_contact == pytest.approx(0.0)


def test_v016_privileged_profile_inherits_exact_nominal_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nominal = _compose("sac/g1_walk_flat/mujoco_no_gait_single_reward", monkeypatch)
    privileged = _compose("sac/g1_walk_flat/mujoco_fada_privileged_oracle", monkeypatch)

    assert OmegaConf.to_container(privileged.reward, resolve=True) == OmegaConf.to_container(
        nominal.reward, resolve=True
    )
    assert privileged.env.fada_privileged_observation.enabled is True
    assert privileged.env.domain_rand.actuator_strength.enabled is True
    assert privileged.env.domain_rand.actuator_strength.candidate_actuator_indices == [3]


def test_v016_single_reward_validator_rejects_retired_authority() -> None:
    validator = getattr(fada_privileged_oracle, "validate_fada_single_reward", None)
    assert callable(validator), "v016 single-Reward validator is missing"

    validator(
        reward_scales={"tracking_lin_vel": 2.0, "alive": 10.0, "feet_phase": 0.0},
        reward_config={
            "scales": {"tracking_lin_vel": 2.0, "alive": 10.0, "feet_phase": 0.0},
            "gait_constraint": {"enabled": False, "penalty_scale": 0.0},
        },
    )

    with pytest.raises(ValueError, match="reward.mode"):
        validator(
            reward_scales={"alive": 10.0},
            reward_config={"scales": {"alive": 10.0}, "mode": {"enabled": True}},
        )
    with pytest.raises(ValueError, match="stand_recovery_terms"):
        validator(
            reward_scales={"alive": 10.0},
            reward_config={
                "scales": {"alive": 10.0},
                "stand_recovery_terms": ["stand_fall_l2"],
            },
        )


def test_v016_runtime_preflight_rejects_mode_and_nested_stand_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _compose("sac/g1_walk_flat/mujoco_fada_privileged_oracle", monkeypatch)
    runtime = resolve_privileged_locomotion_sac_runtime(
        OmegaConf.to_container(cfg.algo, resolve=True)
    )
    assert runtime is not None
    runtime.validate_training_config(cfg)

    OmegaConf.update(cfg, "reward.mode", {"enabled": True}, merge=False, force_add=True)
    with pytest.raises(ValueError, match="reward.mode"):
        runtime.validate_training_config(cfg)

    del cfg.reward.mode
    OmegaConf.update(
        cfg,
        "reward.stand_recovery_terms",
        ["stand_fall_l2"],
        merge=False,
        force_add=True,
    )
    with pytest.raises(ValueError, match="stand_recovery_terms"):
        runtime.validate_training_config(cfg)
