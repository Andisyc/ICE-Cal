from __future__ import annotations

from pathlib import Path

import pytest
import torch

from unilab.algos.torch.fast_sac.learner import SACActor


def _write_nominal_checkpoint(
    path: Path,
    *,
    obs_dim: int = 5,
    action_dim: int = 2,
    hidden_dim: int = 16,
) -> SACActor:
    torch.manual_seed(17)
    actor = SACActor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        use_layer_norm=False,
    )
    with torch.no_grad():
        actor.fc_mu.weight.normal_(mean=0.0, std=0.03)
        actor.fc_mu.bias.copy_(torch.linspace(-0.2, 0.2, action_dim))
    torch.save({"actor": actor.state_dict()}, path)
    return actor


def test_privileged_residual_actor_zero_residual_matches_frozen_nominal(tmp_path: Path) -> None:
    from unilab.algos.torch.fada_context.privileged_residual_sac import (
        PrivilegedResidualSACActor,
    )

    checkpoint = tmp_path / "nominal.pt"
    nominal = _write_nominal_checkpoint(checkpoint)
    actor = PrivilegedResidualSACActor(
        obs_dim=5,
        priv_info_dim=3,
        action_dim=2,
        hidden_dim=16,
        priv_info_embed_dim=4,
        priv_mlp_hidden_dims=(8, 4),
        use_layer_norm=False,
        nominal_checkpoint_path=checkpoint,
        residual_scale=0.2,
    )
    obs = torch.randn(6, 5)
    strength = torch.ones(6, 3)

    expected = nominal.explore(obs, deterministic=True)
    actual = actor.explore(obs, strength, deterministic=True)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert all(not parameter.requires_grad for parameter in actor.nominal_actor.parameters())
    actor.train()
    assert actor.nominal_actor.training is False


def test_privileged_residual_actor_bounds_delta_before_final_clip(tmp_path: Path) -> None:
    from unilab.algos.torch.fada_context.privileged_residual_sac import (
        PrivilegedResidualSACActor,
    )

    checkpoint = tmp_path / "nominal.pt"
    _write_nominal_checkpoint(checkpoint)
    actor = PrivilegedResidualSACActor(
        obs_dim=5,
        priv_info_dim=3,
        action_dim=2,
        hidden_dim=16,
        priv_info_embed_dim=4,
        priv_mlp_hidden_dims=(8, 4),
        use_layer_norm=False,
        nominal_checkpoint_path=checkpoint,
        residual_scale=0.15,
    )
    with torch.no_grad():
        actor.residual_actor.action_mean_head.bias.fill_(10.0)
    obs = torch.randn(4, 5)
    strength = torch.ones(4, 3)

    delta = actor.residual_action(obs, strength, deterministic=True)
    action = actor.explore(obs, strength, deterministic=True)

    assert torch.max(torch.abs(delta)).item() <= 0.150001
    assert torch.max(action).item() <= 1.0
    assert torch.min(action).item() >= -1.0


def test_privileged_residual_learner_uses_final_privileged_tail_and_freezes_nominal(
    tmp_path: Path,
) -> None:
    from unilab.algos.torch.fada_context.privileged_residual_sac import (
        PrivilegedResidualSACLearner,
        derive_motor_strength_from_critic_obs,
    )

    checkpoint = tmp_path / "nominal.pt"
    _write_nominal_checkpoint(checkpoint)
    actor_obs = torch.randn(7, 5)
    critic_only = torch.randn(7, 3)
    strength = torch.rand(7, 3) * 0.2 + 0.8
    critic_obs = torch.cat([actor_obs, critic_only, strength], dim=-1)
    torch.testing.assert_close(
        derive_motor_strength_from_critic_obs(
            actor_obs,
            critic_obs,
            priv_info_dim=3,
            context="test",
        ),
        strength,
    )

    learner = PrivilegedResidualSACLearner(
        obs_dim=5,
        critic_obs_dim=11,
        priv_info_dim=3,
        action_dim=2,
        device="cpu",
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        priv_info_embed_dim=4,
        priv_mlp_hidden_dims=(8, 4),
        num_atoms=11,
        use_layer_norm=False,
        nominal_checkpoint_path=checkpoint,
        residual_scale=0.2,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in learner.actor.residual_actor.named_parameters()
    }
    batch = {
        "obs": actor_obs,
        "critic": critic_obs,
        "actions": torch.randn(7, 2).clamp(-0.5, 0.5),
        "rewards": torch.randn(7),
        "next_obs": torch.randn(7, 5),
        "next_critic": torch.randn(7, 11),
        "dones": torch.zeros(7),
        "truncated": torch.zeros(7),
    }
    batch["next_critic"][:, -3:] = torch.rand(7, 3) * 0.2 + 0.8

    metrics = learner.update_actor(batch)

    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert all(parameter.grad is None for parameter in learner.actor.nominal_actor.parameters())
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in learner.actor.residual_actor.named_parameters()
    )


def test_privileged_residual_checkpoint_binds_nominal_identity(tmp_path: Path) -> None:
    from unilab.algos.torch.fada_context.privileged_residual_sac import (
        PrivilegedResidualSACLearner,
    )

    nominal_a = tmp_path / "nominal_a.pt"
    nominal_b = tmp_path / "nominal_b.pt"
    _write_nominal_checkpoint(nominal_a)
    _write_nominal_checkpoint(nominal_b)
    with torch.no_grad():
        payload = torch.load(nominal_b, weights_only=True)
        payload["actor"]["fc_mu.bias"].add_(0.1)
        torch.save(payload, nominal_b)

    kwargs = dict(
        obs_dim=5,
        critic_obs_dim=11,
        priv_info_dim=3,
        action_dim=2,
        device="cpu",
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        priv_info_embed_dim=4,
        priv_mlp_hidden_dims=(8, 4),
        num_atoms=11,
        use_layer_norm=False,
        residual_scale=0.2,
    )
    source = PrivilegedResidualSACLearner(**kwargs, nominal_checkpoint_path=nominal_a)
    state = source.get_state_dict()
    assert state["privileged_residual_teacher"]["nominal_checkpoint_sha256"]

    same = PrivilegedResidualSACLearner(**kwargs, nominal_checkpoint_path=nominal_a)
    same.load_state_dict(state)

    mismatch = PrivilegedResidualSACLearner(**kwargs, nominal_checkpoint_path=nominal_b)
    with pytest.raises(ValueError, match="nominal checkpoint identity"):
        mismatch.load_state_dict(state)

    tampered = dict(state)
    tampered["actor"] = dict(state["actor"])
    tampered["actor"]["nominal_actor.fc_mu.bias"] = state["actor"]["nominal_actor.fc_mu.bias"] + 0.1
    with pytest.raises(ValueError, match="does not match the configured nominal checkpoint"):
        same.load_state_dict(tampered)


def test_privileged_residual_runtime_requires_explicit_nominal_checkpoint(tmp_path: Path) -> None:
    from unilab.algos.torch.fada_context.privileged_residual_sac import (
        PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE,
        resolve_privileged_residual_sac_runtime,
    )

    checkpoint = tmp_path / "nominal.pt"
    _write_nominal_checkpoint(checkpoint)
    runtime = resolve_privileged_residual_sac_runtime(
        {
            "runtime_impl": PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE,
            "actor": {
                "nominal_checkpoint_path": str(checkpoint),
                "priv_info_dim": 29,
                "residual_scale": 0.2,
            },
        }
    )
    assert runtime is not None
    kwargs = runtime.build_model_kwargs(obs_dim=5, critic_obs_dim=37)
    assert kwargs["nominal_checkpoint_path"] == str(checkpoint)
    assert kwargs["priv_info_dim"] == 29

    with pytest.raises(ValueError, match="nominal_checkpoint_path"):
        resolve_privileged_residual_sac_runtime(
            {"runtime_impl": PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE, "actor": {}}
        )
