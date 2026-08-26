from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from scripts.train_offpolicy import build_runner

from unilab.algos.torch.distill.fada_privileged_oracle import (
    FADA_ORACLE_FINAL_ITERATION,
    FADA_ORACLE_INTERMEDIATE_ITERATIONS,
    FADAOracleCheckpointGateway,
)
from unilab.algos.torch.distill.fada_privileged_oracle_sac import (
    FADAPrivilegedSACLearner,
)
from unilab.algos.torch.offpolicy.double_buffer_runner import DoubleBufferOffPolicyRunner
from unilab.base.registry import ensure_registries
from unilab.ipc.replay_buffer import ReplayBuffer
from unilab.training import BackendAdapter, create_env

ROOT = Path(__file__).resolve().parents[2]


def _compose_offline_oracle_config():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(ROOT / "conf/offpolicy"), version_base="1.3"):
        return compose(
            "config",
            overrides=[
                "algo=sac",
                "task=sac/g1_walk_flat/mujoco_fada_privileged_oracle",
                "training.device=cpu",
                "training.use_amp=false",
                "algo.num_envs=1",
                "algo.batch_size=4",
                "algo.algo_params.use_compile=false",
            ],
        )


@pytest.mark.filterwarnings("ignore:overflow encountered in cast:RuntimeWarning")
def test_v012_privileged_oracle_official_offline_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "formal-offline-v012")
    cfg = _compose_offline_oracle_config()

    runner = build_runner("sac", cfg)
    assert isinstance(runner, DoubleBufferOffPolicyRunner)
    assert isinstance(runner.learner, FADAPrivilegedSACLearner)
    learner = runner.learner
    contract = learner.checkpoint_contract
    assert contract is not None
    assert (contract.obs_dim, contract.critic_obs_dim, contract.action_dim) == (98, 303, 29)
    assert len(contract.body_names) == 31
    assert len(contract.actuated_joint_names) == 29
    assert set(dict(contract.config_hashes)) == {"algo", "env", "reward", "training"}
    assert runner.checkpoint_saver is not None

    torch.manual_seed(20260826)
    rows = 4
    replay = ReplayBuffer(
        capacity=rows,
        obs_dim=contract.obs_dim,
        critic_dim=contract.critic_obs_dim,
        action_dim=contract.action_dim,
        device="cpu",
    )
    obs = torch.randn(rows, contract.obs_dim)
    next_obs = torch.randn(rows, contract.obs_dim)
    critic = torch.randn(rows, contract.critic_obs_dim)
    next_critic = torch.randn(rows, contract.critic_obs_dim)
    actions = torch.tanh(torch.randn(rows, contract.action_dim))
    rewards = torch.tensor([0.5, -0.25, 1.0, 0.0])
    dones = torch.tensor([False, False, True, False])
    truncated = torch.tensor([False, True, False, False])
    replay.add(
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        truncated,
        critic=critic,
        next_critic=next_critic,
    )
    batch = replay.sample(rows)

    actor_before = {
        name: value.detach().clone() for name, value in learner.actor.state_dict().items()
    }
    critic_metrics = learner.update_critic(batch)
    actor_metrics = learner.update_actor(batch)
    actor_after = learner.actor.state_dict()
    assert all(torch.isfinite(torch.tensor(value)) for value in critic_metrics.values())
    assert all(torch.isfinite(torch.tensor(value)) for value in actor_metrics.values())
    assert any(
        not torch.equal(actor_before[name], actor_after[name]) for name in actor_before
    )

    full_checkpoint = tmp_path / "full" / "model_240.pt"
    runner._save_checkpoint(full_checkpoint, iteration=240)
    payload = torch.load(full_checkpoint, map_location="cpu", weights_only=True)
    restored_actor = {
        name: value.detach().clone() for name, value in learner.actor.state_dict().items()
    }
    first_parameter = next(learner.actor.parameters())
    with torch.no_grad():
        first_parameter.add_(37.0)
    learner.load_state_dict(payload)
    assert all(
        torch.equal(learner.actor.state_dict()[name], value)
        for name, value in restored_actor.items()
    )

    gateway = runner.checkpoint_saver.__self__
    assert isinstance(gateway, FADAOracleCheckpointGateway)
    tiny_learner = SimpleNamespace(
        get_state_dict=lambda: {
            "actor": {"marker": torch.tensor([11.0])},
            "optimizer_marker": 13,
        }
    )
    lineage_root = tmp_path / "lineage"
    for iteration in (*FADA_ORACLE_INTERMEDIATE_ITERATIONS, FADA_ORACLE_FINAL_ITERATION):
        gateway.save(tiny_learner, lineage_root / f"model_{iteration}.pt", iteration)
    lineage = json.loads((lineage_root / "fada_oracle_lineage.json").read_text())
    assert lineage["oracle_lineage_id"] == "formal-offline-v012"
    assert len(lineage["checkpoint_sha256"]) == 21
    assert lineage["final_iteration"] == 5000

    print(
        json.dumps(
            {
                "identity": {
                    "obs_dim": contract.obs_dim,
                    "critic_obs_dim": contract.critic_obs_dim,
                    "action_dim": contract.action_dim,
                    "body_count": len(contract.body_names),
                    "joint_count": len(contract.actuated_joint_names),
                },
                "updates": {
                    "critic_metric_count": len(critic_metrics),
                    "actor_metric_count": len(actor_metrics),
                    "actor_parameter_changed": True,
                },
                "persistence": {
                    "full_checkpoint_restored": True,
                    "lineage_checkpoint_count": len(lineage["checkpoint_sha256"]),
                },
            },
            sort_keys=True,
        )
    )


@pytest.mark.filterwarnings("ignore:overflow encountered in cast:RuntimeWarning")
def test_v012_official_env_steps_through_first_velocity_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ICE_CAL_ORACLE_LINEAGE_ID", "formal-push-v012")
    cfg = _compose_offline_oracle_config()
    ensure_registries()
    override = BackendAdapter(cfg, root_dir=ROOT, algo_name="sac").build_task_env_cfg_override()
    env = create_env(
        cfg,
        num_envs=1,
        env_cfg_override=override,
        sim_backend="mujoco",
    )
    try:
        state = env.init_state()
        actions = np.zeros((1, env.action_space.shape[0]), dtype=np.float32)
        interval_steps = round(
            float(cfg.env.domain_rand.fada_push_interval_seconds) / env.cfg.ctrl_dt
        )
        for _ in range(interval_steps + 1):
            state = env.step(actions)

        assert env.step_counter == interval_steps + 1
        assert np.isfinite(state.obs["obs"]).all()
        assert np.isfinite(state.obs["critic"]).all()
        assert np.isfinite(state.reward).all()
    finally:
        env.close()
