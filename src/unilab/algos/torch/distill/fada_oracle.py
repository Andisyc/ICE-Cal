"""Cold-path checkpoint Gateway for the single frozen FADA Oracle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn

from unilab.algos.torch.common.actor_factory import build_actor
from unilab.algos.torch.common.normalization import EmpiricalNormalization
from unilab.algos.torch.hora.observations import split_hora_obs_with_priv_info

from .fada_privileged_oracle import validate_fada_oracle_lineage
from .playback import load_distillation_student_policy
from .teacher import DistillationTeacherSpec, LoadedTeacherPolicy, load_sac_teacher_policy


def _is_distillation_student_checkpoint(payload: object) -> bool:
    return isinstance(payload, dict) and {
        "student_state_dict",
        "distill_runtime_cfg",
    }.issubset(payload)


class LoadedFADAPrivilegedOraclePolicy(nn.Module):
    """Frozen privileged Oracle with an env-observation inference boundary."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        obs_normalizer: nn.Module | None,
        checkpoint_identity: Mapping[str, object],
    ) -> None:
        super().__init__()
        self.actor = actor
        self.obs_dim = int(obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.priv_info_dim = self.critic_obs_dim - self.obs_dim
        self.action_dim = int(action_dim)
        self.obs_normalizer = obs_normalizer
        self.checkpoint_identity = dict(checkpoint_identity)
        self.oracle_lineage_id = str(checkpoint_identity.get("oracle_lineage_id", ""))
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor, priv_info: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] != self.obs_dim or priv_info.shape[-1] != self.priv_info_dim:
            raise ValueError(
                "privileged Oracle input shape mismatch: "
                f"obs={tuple(obs.shape)} priv_info={tuple(priv_info.shape)}"
            )
        actor_obs = obs
        if self.obs_normalizer is not None:
            actor_obs = self.obs_normalizer(actor_obs, update=False)
        action = self.actor.explore(actor_obs, priv_info, deterministic=True)
        if action.shape != (obs.shape[0], self.action_dim):
            raise ValueError(f"privileged Oracle action shape mismatch: {tuple(action.shape)}")
        return action

    def actions_from_env_observation(
        self,
        obs: Mapping[str, np.ndarray],
        info: Mapping[str, object] | None,
    ) -> np.ndarray:
        actor_obs, critic_obs, priv_info = split_hora_obs_with_priv_info(
            dict(obs), None if info is None else dict(info)
        )
        if critic_obs is None or critic_obs.shape != (actor_obs.shape[0], self.critic_obs_dim):
            raise ValueError(
                "privileged Oracle requires the exact critic observation shape "
                f"(*, {self.critic_obs_dim})"
            )
        if priv_info is None or priv_info.shape != (actor_obs.shape[0], self.priv_info_dim):
            raise ValueError(
                "privileged Oracle requires the exact critic-tail privileged shape "
                f"(*, {self.priv_info_dim})"
            )
        device = next(self.actor.parameters()).device
        action = self(
            torch.as_tensor(actor_obs, dtype=torch.float32, device=device),
            torch.as_tensor(priv_info, dtype=torch.float32, device=device),
        )
        result = action.detach().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("privileged Oracle produced non-finite actions")
        return result


def _privileged_checkpoint_metadata(payload: Mapping[str, object]) -> Mapping[str, object]:
    metadata = payload.get("fada_privileged_oracle")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint missing fada_privileged_oracle identity")
    dimensions = metadata.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("privileged Oracle checkpoint identity missing dimensions")
    if metadata.get("actor_directly_privileged") is not True:
        raise ValueError("privileged Oracle checkpoint must bind actor_directly_privileged=true")
    return metadata


def _load_privileged_oracle_policy(
    payload: Mapping[str, object],
    spec: DistillationTeacherSpec,
    *,
    device: str | torch.device,
) -> LoadedFADAPrivilegedOraclePolicy:
    metadata = _privileged_checkpoint_metadata(payload)
    dimensions = metadata["dimensions"]
    assert isinstance(dimensions, Mapping)
    critic_obs_dim = int(dimensions["critic"])
    expected_critic_obs_dim = (
        critic_obs_dim if spec.critic_obs_dim is None else int(spec.critic_obs_dim)
    )
    expected = (int(spec.obs_dim), expected_critic_obs_dim, int(spec.action_dim))
    observed = (int(dimensions["obs"]), critic_obs_dim, int(dimensions["action"]))
    if observed != expected:
        raise ValueError(
            f"privileged Oracle checkpoint dimension mismatch: expected={expected} observed={observed}"
        )
    actor_state = payload.get("actor")
    if not isinstance(actor_state, Mapping):
        raise ValueError("privileged Oracle checkpoint is missing actor state")
    actor = build_actor(
        "privileged_locomotion_sac",
        int(spec.obs_dim),
        int(spec.action_dim),
        int(spec.actor_hidden_dim),
        bool(spec.use_layer_norm),
        device,
        priv_info_dim=critic_obs_dim - int(spec.obs_dim),
        priv_info_embed_dim=int(spec.priv_info_embed_dim),
        priv_mlp_hidden_dims=tuple(spec.priv_mlp_hidden_dims),
        priv_info_normalization=bool(spec.priv_info_normalization),
    )
    actor.load_state_dict(dict(actor_state), strict=True)
    obs_normalizer: EmpiricalNormalization | None = None
    if spec.obs_normalization:
        normalizer_state = payload.get("obs_normalizer")
        if not isinstance(normalizer_state, Mapping):
            raise ValueError("privileged Oracle checkpoint is missing obs_normalizer state")
        obs_normalizer = EmpiricalNormalization(shape=int(spec.obs_dim), device=device)
        obs_normalizer.load_state_dict(dict(normalizer_state), strict=True)
        obs_normalizer.eval()
    return LoadedFADAPrivilegedOraclePolicy(
        actor=actor,
        obs_dim=int(spec.obs_dim),
        critic_obs_dim=critic_obs_dim,
        action_dim=int(spec.action_dim),
        obs_normalizer=obs_normalizer,
        checkpoint_identity=metadata,
    )


def validate_loaded_fada_oracle_lineage(
    final_policy: nn.Module,
    intermediate_policies: list[nn.Module],
) -> None:
    """Validate the complete privileged 20+1 lineage before env construction."""

    if not isinstance(final_policy, LoadedFADAPrivilegedOraclePolicy):
        if any(
            isinstance(policy, LoadedFADAPrivilegedOraclePolicy) for policy in intermediate_policies
        ):
            raise ValueError("FADA Oracle lineage mixes privileged and non-privileged policies")
        return
    if not all(
        isinstance(policy, LoadedFADAPrivilegedOraclePolicy) for policy in intermediate_policies
    ):
        raise ValueError("privileged final Oracle requires privileged intermediate Oracles")
    records = [
        policy.checkpoint_identity
        for policy in intermediate_policies
        if isinstance(policy, LoadedFADAPrivilegedOraclePolicy)
    ]
    records.append(final_policy.checkpoint_identity)
    validate_fada_oracle_lineage(records)


def load_fada_oracle_policy(
    checkpoint_path: str | Path,
    spec: DistillationTeacherSpec,
    *,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load one SAC or distillation policy behind the FADA tensor-policy contract."""

    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if _is_distillation_student_checkpoint(payload):
        loaded = load_distillation_student_policy(path, device=device)
        if loaded.obs_dim != int(spec.obs_dim):
            raise ValueError(
                "FADA Oracle obs dim mismatch: "
                f"checkpoint={loaded.obs_dim}, configured={int(spec.obs_dim)}"
            )
        if loaded.action_dim != int(spec.action_dim):
            raise ValueError(
                "FADA Oracle action dim mismatch: "
                f"checkpoint={loaded.action_dim}, configured={int(spec.action_dim)}"
            )
        policy = loaded.policy
    elif isinstance(payload, Mapping) and "fada_privileged_oracle" in payload:
        if spec.algo_type != "privileged_locomotion_sac":
            raise ValueError(
                "privileged Oracle checkpoint requires teacher.algo_type="
                "'privileged_locomotion_sac'"
            )
        policy = _load_privileged_oracle_policy(payload, spec, device=device)
    else:
        policy = load_sac_teacher_policy(path, spec, device=device)

    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def reload_fada_oracle_policy_(
    policy: nn.Module,
    checkpoint_path: str | Path,
    spec: DistillationTeacherSpec,
) -> None:
    """Reload a same-lineage Oracle without changing its runtime object identity."""

    replacement = load_fada_oracle_policy(
        checkpoint_path,
        spec,
        device=next(policy.parameters()).device,
    )
    if type(replacement) is not type(policy):
        raise ValueError("Oracle checkpoint runtime type changed during resident reload")
    if isinstance(policy, LoadedFADAPrivilegedOraclePolicy):
        assert isinstance(replacement, LoadedFADAPrivilegedOraclePolicy)
        if replacement.oracle_lineage_id != policy.oracle_lineage_id:
            raise ValueError("intermediate Oracle checkpoint lineage mismatch")
        policy.actor.load_state_dict(replacement.actor.state_dict(), strict=True)
        if (policy.obs_normalizer is None) != (replacement.obs_normalizer is None):
            raise ValueError("Oracle observation normalizer ownership changed during reload")
        if policy.obs_normalizer is not None and replacement.obs_normalizer is not None:
            policy.obs_normalizer.load_state_dict(
                replacement.obs_normalizer.state_dict(), strict=True
            )
        policy.eval()
        return
    if isinstance(policy, LoadedTeacherPolicy):
        assert isinstance(replacement, LoadedTeacherPolicy)
        policy.actor.load_state_dict(replacement.actor.state_dict(), strict=True)
        if policy.obs_normalizer is not None and replacement.obs_normalizer is not None:
            policy.obs_normalizer.load_state_dict(
                replacement.obs_normalizer.state_dict(), strict=True
            )
        policy.eval()
        return
    raise TypeError(f"unsupported resident Oracle policy type: {type(policy).__name__}")


__all__ = [
    "LoadedFADAPrivilegedOraclePolicy",
    "load_fada_oracle_policy",
    "reload_fada_oracle_policy_",
    "validate_loaded_fada_oracle_lineage",
]
