from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.algos._fada_training_test_support import _CommandControlledEnv, _curriculum_config
from unilab.algos.torch.distill import (
    DistillationTeacherSpec,
    FADACollectionSpec,
    MLPStudentPolicy,
    collect_fada_source_windows,
    save_distillation_checkpoint,
)


def _save_distilled_oracle(path: Path, *, obs_dim: int = 98, action_dim: int = 29) -> None:
    policy = MLPStudentPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
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
        agent_steps=17,
        teacher_metadata={"source": "walk-stand-dagger"},
        distill_runtime_cfg={
            "student_model_type": "mlp",
            "student_obs_dim": obs_dim,
            "student_action_dim": action_dim,
            "student_hidden_dims": [8],
            "student_activation": "elu",
            "student_squash_action": True,
        },
    )


def test_fada_oracle_loader_accepts_frozen_distillation_student(tmp_path: Path) -> None:
    from unilab.algos.torch.distill.fada_oracle import load_fada_oracle_policy

    checkpoint = tmp_path / "dagger_iteration_8.pt"
    _save_distilled_oracle(checkpoint)

    oracle = load_fada_oracle_policy(
        checkpoint,
        DistillationTeacherSpec(obs_dim=98, action_dim=29),
        device="cpu",
    )

    action = oracle(torch.arange(98, dtype=torch.float32).reshape(1, 98) / 100.0)
    assert action.shape == (1, 29)
    assert oracle.obs_dim == 98
    assert oracle.action_dim == 29
    assert not oracle.training
    assert all(not parameter.requires_grad for parameter in oracle.parameters())


def test_fada_oracle_loader_rejects_distillation_dimension_mismatch(tmp_path: Path) -> None:
    from unilab.algos.torch.distill.fada_oracle import load_fada_oracle_policy

    checkpoint = tmp_path / "wrong-dim.pt"
    _save_distilled_oracle(checkpoint, obs_dim=97)

    with pytest.raises(ValueError, match="obs dim mismatch"):
        load_fada_oracle_policy(
            checkpoint,
            DistillationTeacherSpec(obs_dim=98, action_dim=29),
            device="cpu",
        )


def test_walk_to_stand_uses_one_oracle_without_second_policy() -> None:
    class CountingOracle(torch.nn.Module):
        obs_dim = 3
        action_dim = 2

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return torch.full((obs.shape[0], 2), 0.125, device=obs.device)

    oracle = CountingOracle()
    result = collect_fada_source_windows(
        _CommandControlledEnv(),
        teacher_policy=oracle,
        config=_curriculum_config(),
        num_windows=1,
        spec=FADACollectionSpec(
            command_info_keys=("commands",),
            command_scenario="walk_to_stand",
            transition_walk_command=(0.4, 0.0, 0.0),
            transition_pre_switch_steps=2,
            transition_post_switch_steps=4,
            max_env_steps=16,
            collect_oracle_shadow=True,
        ),
    )

    assert result.batch.command.shape == (1, 3)
    assert oracle.calls > 0

