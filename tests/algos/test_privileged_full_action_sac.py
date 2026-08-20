from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from unilab.algos.torch.fast_sac.learner import SACActor

ROOT_DIR = Path(__file__).resolve().parents[2]


def _write_nominal(path: Path) -> SACActor:
    torch.manual_seed(41)
    actor = SACActor(
        obs_dim=5,
        action_dim=2,
        hidden_dim=16,
        use_layer_norm=False,
    )
    with torch.no_grad():
        actor.fc_mu.weight.normal_(mean=0.0, std=0.03)
        actor.fc_mu.bias.copy_(torch.tensor([-0.2, 0.2]))
    torch.save({"actor": actor.state_dict()}, path)
    return actor


def test_full_action_actor_warm_start_matches_nominal_without_nominal_branch(
    tmp_path: Path,
) -> None:
    from unilab.algos.torch.fada_context.privileged_full_action_sac import (
        PrivilegedFullActionSACActor,
    )

    checkpoint = tmp_path / "nominal.pt"
    nominal = _write_nominal(checkpoint)
    teacher = PrivilegedFullActionSACActor(
        obs_dim=5,
        priv_info_dim=29,
        action_dim=2,
        hidden_dim=16,
        priv_info_embed_dim=4,
        priv_mlp_hidden_dims=(8, 4),
        use_layer_norm=False,
        nominal_initialization_checkpoint=checkpoint,
    )
    obs = torch.randn(7, 5)
    strength_a = torch.ones(7, 29)
    strength_b = torch.rand(7, 29) * 0.2 + 0.8

    expected = nominal.explore(obs, deterministic=True)
    torch.testing.assert_close(
        teacher.explore(obs, strength_a, deterministic=True), expected, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        teacher.explore(obs, strength_b, deterministic=True), expected, rtol=1e-6, atol=1e-7
    )
    assert not hasattr(teacher, "nominal_actor")
    assert all(parameter.requires_grad for parameter in teacher.parameters())


def test_full_action_learner_uses_final_29d_tail_and_updates_complete_actor(tmp_path: Path) -> None:
    from unilab.algos.torch.fada_context.privileged_full_action_sac import (
        PrivilegedFullActionSACLearner,
        derive_motor_strength_from_critic_obs,
    )

    checkpoint = tmp_path / "nominal.pt"
    _write_nominal(checkpoint)
    obs = torch.randn(8, 5)
    critic_only = torch.randn(8, 3)
    strength = torch.rand(8, 29) * 0.2 + 0.8
    critic = torch.cat([obs, critic_only, strength], dim=-1)
    torch.testing.assert_close(
        derive_motor_strength_from_critic_obs(obs, critic, priv_info_dim=29, context="test"),
        strength,
    )
    learner = PrivilegedFullActionSACLearner(
        obs_dim=5,
        critic_obs_dim=37,
        priv_info_dim=29,
        action_dim=2,
        nominal_initialization_checkpoint=checkpoint,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        priv_info_embed_dim=4,
        priv_mlp_hidden_dims=(8, 4),
        num_atoms=11,
        use_layer_norm=False,
    )
    optimized = {
        id(parameter)
        for group in learner.actor_optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized == {id(parameter) for parameter in learner.actor.parameters()}
    assert all(
        not parameter.requires_grad for parameter in learner.nominal_anchor_actor.parameters()
    )
    assert learner._nominal_action_anchor_loss(obs, critic).item() == pytest.approx(0.0, abs=1e-12)
    with torch.no_grad():
        learner.actor.action_mean_head.bias.add_(0.1)
    assert learner._nominal_action_anchor_loss(obs, critic).item() > 0.0
    before = {
        name: parameter.detach().clone() for name, parameter in learner.actor.named_parameters()
    }
    next_critic = torch.randn(8, 37)
    next_critic[:, -29:] = torch.rand(8, 29) * 0.2 + 0.8
    metrics = learner.update_actor(
        {
            "obs": obs,
            "critic": critic,
            "actions": torch.zeros(8, 2),
            "rewards": torch.randn(8),
            "next_obs": torch.randn(8, 5),
            "next_critic": next_critic,
            "dones": torch.zeros(8),
            "truncated": torch.zeros(8),
        }
    )
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert metrics["nominal_action_anchor_mse"] >= 0.0
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in learner.actor.named_parameters()
    )


def test_full_action_checkpoint_binds_initialization_identity(tmp_path: Path) -> None:
    from unilab.algos.torch.fada_context.privileged_full_action_sac import (
        PrivilegedFullActionSACLearner,
    )

    checkpoint_a = tmp_path / "a.pt"
    checkpoint_b = tmp_path / "b.pt"
    _write_nominal(checkpoint_a)
    _write_nominal(checkpoint_b)
    payload = torch.load(checkpoint_b, weights_only=True)
    payload["actor"]["fc_mu.bias"] += 0.1
    torch.save(payload, checkpoint_b)
    kwargs = dict(
        obs_dim=5,
        critic_obs_dim=37,
        priv_info_dim=29,
        action_dim=2,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        priv_info_embed_dim=4,
        priv_mlp_hidden_dims=(8, 4),
        num_atoms=11,
        use_layer_norm=False,
    )
    source = PrivilegedFullActionSACLearner(
        **kwargs, nominal_initialization_checkpoint=checkpoint_a
    )
    state = source.get_state_dict()
    same = PrivilegedFullActionSACLearner(**kwargs, nominal_initialization_checkpoint=checkpoint_a)
    same.load_state_dict(state)
    mismatch = PrivilegedFullActionSACLearner(
        **kwargs, nominal_initialization_checkpoint=checkpoint_b
    )
    with pytest.raises(ValueError, match="nominal_initialization_sha256 mismatch"):
        mismatch.load_state_dict(state)


def test_full_action_one_env_mujoco_step_and_update_are_finite() -> None:
    pytest.importorskip("mujoco", reason="MuJoCo is required for the live sentinel")
    try:
        from mujoco.batch_env import BatchEnvPool as _  # noqa: F401
    except Exception:
        pytest.skip("mujoco.batch_env is unavailable")

    from unilab.algos.torch.fada_context.full_action_formal_protocol import FORMAL_TASK_CONFIG
    from unilab.algos.torch.fada_context.privileged_full_action_sac import (
        PrivilegedFullActionSACLearner,
    )
    from unilab.base.observations import split_obs_dict
    from unilab.training import BackendAdapter, create_env, ensure_registries

    nominal_checkpoint = ROOT_DIR / "checkpoints/oracles/G1WalkFlat/model_5000.pt"
    if not nominal_checkpoint.is_file():
        pytest.skip("formal nominal SAC checkpoint is a local, untracked artifact")

    with initialize_config_dir(config_dir=str(ROOT_DIR / "conf" / "offpolicy"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"task={FORMAL_TASK_CONFIG}"])
    ensure_registries()
    env_override = BackendAdapter(
        cfg, root_dir=ROOT_DIR, algo_name="sac"
    ).build_task_env_cfg_override()
    env = create_env(
        cfg,
        num_envs=1,
        env_cfg_override=env_override,
        sim_backend="mujoco",
    )
    try:
        state = env.init_state()
        obs_np, critic_np = split_obs_dict(state.obs)
        strength = np.asarray(state.info["privileged_actuator_strength"], dtype=np.float32)
        assert strength.shape == (1, 29)
        assert strength[0, 3] == pytest.approx(0.9)
        assert np.count_nonzero(strength != 1.0) == 1
        learner = PrivilegedFullActionSACLearner(
            obs_dim=98,
            critic_obs_dim=130,
            priv_info_dim=29,
            action_dim=29,
            nominal_initialization_checkpoint=nominal_checkpoint,
            actor_hidden_dim=512,
            critic_hidden_dim=64,
            priv_info_embed_dim=16,
            priv_mlp_hidden_dims=(128, 64, 16),
            num_atoms=11,
            device="cpu",
        )
        obs = torch.from_numpy(np.asarray(obs_np, dtype=np.float32))
        critic = torch.from_numpy(np.asarray(critic_np, dtype=np.float32))
        with torch.inference_mode():
            action = learner.actor.explore(obs, torch.from_numpy(strength), deterministic=False)
        assert action.shape == (1, 29)
        assert torch.isfinite(action).all()
        next_state = env.step(action.numpy())
        next_obs_np, next_critic_np = split_obs_dict(next_state.obs)
        batch = {
            "obs": obs,
            "critic": critic,
            "actions": action,
            "rewards": torch.from_numpy(np.asarray(next_state.reward, dtype=np.float32)),
            "next_obs": torch.from_numpy(np.asarray(next_obs_np, dtype=np.float32)),
            "next_critic": torch.from_numpy(np.asarray(next_critic_np, dtype=np.float32)),
            "dones": torch.from_numpy(np.asarray(next_state.terminated, dtype=np.float32)),
            "truncated": torch.from_numpy(np.asarray(next_state.truncated, dtype=np.float32)),
        }
        metrics = {**learner.update_critic(batch), **learner.update_actor(batch)}
        assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    finally:
        env.close()
