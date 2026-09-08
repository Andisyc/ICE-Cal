"""Removing random knee weakening must preserve the independent DR schedule."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada.privileged_oracle_sac import (
    resolve_privileged_locomotion_sac_runtime,
)
from unilab.base.registry import apply_cfg_overrides
from unilab.envs.locomotion.g1.walk_actuator_randomization import (
    sample_actuator_strength_multipliers,
    validate_actuator_strength_config,
)
from unilab.envs.locomotion.g1.walk_config import G1ActuatorStrengthConfig, G1WalkFlatCfg
from unilab.envs.locomotion.g1.walk_domain_randomization import (
    G1WalkDomainRandomizationProvider,
)
from unilab.training.offpolicy.factory import build_offpolicy_env_cfg_override


def test_random_weakening_removed_without_disabling_physical_dr(monkeypatch):
    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "removed-strength-test")
    root = Path(__file__).resolve().parents[4]
    with initialize_config_dir(config_dir=str(root / "conf/offpolicy"), version_base="1.3"):
        cfg = compose("config", overrides=[
            "algo=sac", "task=sac/g1_walk_flat/mujoco_fada_fixed_contact",
        ])
    runtime = resolve_privileged_locomotion_sac_runtime(
        OmegaConf.to_container(cfg.algo, resolve=True)
    )
    runtime.validate_training_config(cfg)
    typed = G1WalkFlatCfg()
    apply_cfg_overrides(typed, build_offpolicy_env_cfg_override("sac", cfg))
    typed.validate()
    env = SimpleNamespace(cfg=typed, _num_action=29)
    provider = G1WalkDomainRandomizationProvider()
    assert provider._sample_actuator_strength_multipliers(env, 2) is None
    assert provider.grouped_domain_rand_curriculum_profile(env) == (0, 0.0)
    assert not provider.effective_grouped_domain_rand_config(env).randomize_kp
    assert provider.update_iteration_curriculum(env, iteration=500, terminated_fraction=0.0)
    effective = provider.effective_grouped_domain_rand_config(env)
    assert provider.grouped_domain_rand_curriculum_profile(env) == (1, 0.2)
    assert effective.randomize_kp and effective.randomize_kd
    np.testing.assert_allclose(effective.kp_multiplier_range, [0.98, 1.02])
    restored = G1WalkDomainRandomizationProvider()
    restored.restore_actuator_strength_curriculum_state(
        provider.capture_actuator_strength_curriculum_state()
    )
    assert restored.grouped_domain_rand_curriculum_profile(env) == (1, 0.2)

    legacy = G1ActuatorStrengthConfig(
        enabled=False, sampling_mode="single_candidate", candidate_actuator_indices=[3],
        multiplier_range=[0.8, 1.0], nominal_probability=0.3,
    )
    assert validate_actuator_strength_config(legacy, expected_actions=29) is None
    legacy.enabled = True
    with pytest.raises(ValueError, match="removed"):
        sample_actuator_strength_multipliers(legacy, num_reset=2, expected_actions=29)
    fixed = G1ActuatorStrengthConfig(enabled=True, multipliers=[1.0, 0.9])
    np.testing.assert_array_equal(
        sample_actuator_strength_multipliers(fixed, num_reset=2, expected_actions=2),
        [[1.0, 0.9], [1.0, 0.9]],
    )
    fixed.curriculum_enabled = True
    with pytest.raises(ValueError, match="removed"):
        validate_actuator_strength_config(fixed, expected_actions=2)
