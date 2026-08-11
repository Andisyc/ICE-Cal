"""Privileged full-action SAC teacher for fixed actuator-strength training."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from unilab.algos.torch.fast_sac.learner import FastSACLearner, SACActor
from unilab.algos.torch.hora.sac_models import HoraSACActor
from unilab.algos.torch.offpolicy.runtime import OffPolicyRuntime

PRIVILEGED_FULL_ACTION_SAC_ALGO_TYPE = "privileged_full_action_sac"
PRIVILEGED_ACTUATOR_STRENGTH_DIM = 29
_CHECKPOINT_SCHEMA = "unilab_privileged_full_action_teacher_v1"


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_motor_strength_from_critic_obs(
    actor_obs: torch.Tensor,
    critic_obs: torch.Tensor,
    *,
    priv_info_dim: int,
    context: str,
) -> torch.Tensor:
    """Return only the declared motor-strength tail from critic observations."""
    actor_dim = int(actor_obs.shape[-1])
    critic_dim = int(critic_obs.shape[-1])
    if int(priv_info_dim) <= 0:
        raise ValueError(f"{context} requires positive priv_info_dim, got {priv_info_dim}.")
    if critic_dim < actor_dim + int(priv_info_dim):
        raise ValueError(
            f"Privileged full-action SAC {context} requires a final {priv_info_dim}D "
            f"motor-strength tail; got actor_dim={actor_dim}, critic_dim={critic_dim}."
        )
    return critic_obs[..., -int(priv_info_dim) :]


class PrivilegedFullActionSACActor(HoraSACActor):
    """Trainable full-action policy conditioned on actor observations and motor strength."""

    def __init__(
        self,
        obs_dim: int,
        priv_info_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 512,
        priv_info_embed_dim: int = 16,
        priv_mlp_hidden_dims: Sequence[int] = (128, 64, 16),
        log_std_max: float = 0.0,
        log_std_min: float = -5.0,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        device: str | torch.device = "cpu",
        nominal_initialization_checkpoint: str | Path | None = None,
    ) -> None:
        if int(priv_info_dim) != PRIVILEGED_ACTUATOR_STRENGTH_DIM:
            raise ValueError(
                f"Privileged full-action SAC requires priv_info_dim=29, got {priv_info_dim}."
            )
        super().__init__(
            obs_dim=obs_dim,
            priv_info_dim=priv_info_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            priv_info_embed_dim=priv_info_embed_dim,
            priv_mlp_hidden_dims=tuple(priv_mlp_hidden_dims),
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            use_tanh=use_tanh,
            use_layer_norm=use_layer_norm,
            device=device,
        )
        self.nominal_initialization_path: str | None = None
        self.nominal_initialization_sha256: str | None = None
        if nominal_initialization_checkpoint:
            self.load_nominal_initialization(
                nominal_initialization_checkpoint,
                hidden_dim=hidden_dim,
                use_layer_norm=use_layer_norm,
                device=device,
            )

    def load_nominal_initialization(
        self,
        checkpoint_path: str | Path,
        *,
        hidden_dim: int,
        use_layer_norm: bool,
        device: str | torch.device,
    ) -> None:
        """Warm-start the full-action network without retaining a nominal policy branch."""
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Nominal SAC checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("actor"), dict):
            raise ValueError(f"Nominal SAC checkpoint is missing actor state: {path}")

        nominal = SACActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            log_std_max=self.log_std_max,
            log_std_min=self.log_std_min,
            use_tanh=self.use_tanh,
            use_layer_norm=use_layer_norm,
            device=device,
        )
        try:
            nominal.load_state_dict(checkpoint["actor"], strict=True)
        except RuntimeError as error:
            raise ValueError(
                "Nominal SAC checkpoint is incompatible with the full-action teacher dimensions: "
                f"{path}"
            ) from error

        with torch.no_grad():
            nominal_linears = [module for module in nominal.net if isinstance(module, nn.Linear)]
            teacher_linears = [
                module for module in self.actor_trunk if isinstance(module, nn.Linear)
            ]
            if len(nominal_linears) != len(teacher_linears):
                raise ValueError("Nominal and teacher actor trunks have different linear depth")
            for index, (source, target) in enumerate(zip(nominal_linears, teacher_linears)):
                if index == 0:
                    target.weight.zero_()
                    target.weight[:, : self.obs_dim].copy_(source.weight)
                else:
                    target.weight.copy_(source.weight)
                target.bias.copy_(source.bias)
            nominal_norms = [module for module in nominal.net if isinstance(module, nn.LayerNorm)]
            teacher_norms = [
                module for module in self.actor_trunk if isinstance(module, nn.LayerNorm)
            ]
            if len(nominal_norms) != len(teacher_norms):
                raise ValueError(
                    "Nominal and teacher actor trunks have different normalization depth"
                )
            for source, target in zip(nominal_norms, teacher_norms):
                target.load_state_dict(source.state_dict(), strict=True)
            self.action_mean_head.load_state_dict(nominal.fc_mu.state_dict(), strict=True)
            self.action_logstd_head.load_state_dict(nominal.fc_logstd.state_dict(), strict=True)

        self.nominal_initialization_path = str(path)
        self.nominal_initialization_sha256 = _checkpoint_sha256(path)


class PrivilegedFullActionSACLearner(FastSACLearner):
    """FastSAC learner whose complete actor action is conditioned on motor strength."""

    def __init__(
        self,
        *,
        obs_dim: int,
        critic_obs_dim: int,
        priv_info_dim: int,
        action_dim: int,
        nominal_initialization_checkpoint: str | Path,
        device: str = "cpu",
        actor_hidden_dim: int = 512,
        priv_info_embed_dim: int = 16,
        priv_mlp_hidden_dims: Sequence[int] = (128, 64, 16),
        log_std_max: float = 0.0,
        log_std_min: float = -5.0,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        actor_lr: float = 3e-4,
        nominal_action_anchor_coef: float = 1.0,
        weight_decay: float = 0.001,
        use_symmetry: bool = False,
        symmetry_augmentation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if use_symmetry or symmetry_augmentation is not None:
            raise ValueError("Privileged full-action SAC does not support symmetry augmentation.")
        if int(critic_obs_dim) < int(obs_dim) + int(priv_info_dim):
            raise ValueError(
                "Privileged full-action SAC critic observation is missing the declared "
                f"motor-strength tail: obs_dim={obs_dim}, critic_obs_dim={critic_obs_dim}, "
                f"priv_info_dim={priv_info_dim}."
            )
        if not float(nominal_action_anchor_coef) > 0.0:
            raise ValueError(
                "Privileged full-action SAC requires nominal_action_anchor_coef > 0, "
                f"got {nominal_action_anchor_coef}."
            )
        self.nominal_action_anchor_coef = float(nominal_action_anchor_coef)
        super().__init__(
            obs_dim=obs_dim,
            critic_obs_dim=critic_obs_dim,
            action_dim=action_dim,
            device=device,
            actor_hidden_dim=actor_hidden_dim,
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            use_tanh=use_tanh,
            use_layer_norm=use_layer_norm,
            actor_lr=actor_lr,
            weight_decay=weight_decay,
            use_symmetry=False,
            symmetry_augmentation=None,
            **kwargs,
        )
        self.priv_info_dim = int(priv_info_dim)
        self.actor = PrivilegedFullActionSACActor(
            obs_dim=obs_dim,
            priv_info_dim=self.priv_info_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            priv_info_embed_dim=priv_info_embed_dim,
            priv_mlp_hidden_dims=tuple(priv_mlp_hidden_dims),
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            use_tanh=use_tanh,
            use_layer_norm=use_layer_norm,
            device=device,
            nominal_initialization_checkpoint=nominal_initialization_checkpoint,
        )
        self.nominal_anchor_actor = SACActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=actor_hidden_dim,
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            use_tanh=use_tanh,
            use_layer_norm=use_layer_norm,
            device=device,
        )
        nominal_checkpoint = torch.load(
            Path(nominal_initialization_checkpoint).expanduser().resolve(),
            map_location=device,
            weights_only=True,
        )
        self.nominal_anchor_actor.load_state_dict(nominal_checkpoint["actor"], strict=True)
        self.nominal_anchor_actor.eval()
        self.nominal_anchor_actor.requires_grad_(False)
        fused = isinstance(device, str) and device.startswith("cuda")
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=actor_lr,
            weight_decay=weight_decay,
            fused=fused,
            betas=(0.9, 0.95),
        )

    def _strength(
        self, actor_obs: torch.Tensor, critic_obs: torch.Tensor, context: str
    ) -> torch.Tensor:
        return derive_motor_strength_from_critic_obs(
            actor_obs,
            critic_obs,
            priv_info_dim=self.priv_info_dim,
            context=context,
        )

    def _get_actions_and_log_probs_for_critic(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor = cast(PrivilegedFullActionSACActor, self.actor)
        return actor.get_actions_and_log_probs(
            actor_obs,
            self._strength(actor_obs, critic_obs, "critic update"),
        )

    def _get_actions_and_log_probs_for_actor(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actor = cast(PrivilegedFullActionSACActor, self.actor)
        return actor.get_actions_and_log_probs(
            actor_obs,
            self._strength(actor_obs, critic_obs, "actor update"),
        )

    def _nominal_action_anchor_loss(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> torch.Tensor:
        actor = cast(PrivilegedFullActionSACActor, self.actor)
        teacher_action, _mean, _log_std = actor(
            actor_obs,
            self._strength(actor_obs, critic_obs, "nominal action anchor"),
        )
        with torch.no_grad():
            nominal_action = self.nominal_anchor_actor.explore(
                actor_obs,
                deterministic=True,
            )
        return F.mse_loss(teacher_action, nominal_action)

    def _actor_loss_tensors(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sac_loss, policy_entropy, action_std = super()._actor_loss_tensors(obs, critic_obs)
        anchor_loss = self._nominal_action_anchor_loss(obs, critic_obs)
        return (
            sac_loss + self.nominal_action_anchor_coef * anchor_loss,
            policy_entropy,
            action_std,
        )

    def update_actor(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        metrics = super().update_actor(batch)
        with torch.no_grad():
            anchor_loss = self._nominal_action_anchor_loss(batch["obs"], batch["critic"])
        metrics["nominal_action_anchor_mse"] = float(anchor_loss.detach().cpu())
        return metrics

    def get_state_dict(self) -> dict[str, Any]:
        state = super().get_state_dict()
        actor = cast(PrivilegedFullActionSACActor, self.actor)
        state["privileged_full_action_teacher"] = {
            "schema": _CHECKPOINT_SCHEMA,
            "nominal_initialization_path": actor.nominal_initialization_path,
            "nominal_initialization_sha256": actor.nominal_initialization_sha256,
            "obs_dim": actor.obs_dim,
            "priv_info_dim": actor.priv_info_dim,
            "action_dim": actor.action_dim,
            "nominal_action_anchor_coef": self.nominal_action_anchor_coef,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        metadata = state_dict.get("privileged_full_action_teacher")
        if not isinstance(metadata, dict) or metadata.get("schema") != _CHECKPOINT_SCHEMA:
            raise ValueError("Privileged full-action teacher checkpoint metadata is invalid.")
        actor = cast(PrivilegedFullActionSACActor, self.actor)
        expected = {
            "nominal_initialization_sha256": actor.nominal_initialization_sha256,
            "obs_dim": actor.obs_dim,
            "priv_info_dim": actor.priv_info_dim,
            "action_dim": actor.action_dim,
            "nominal_action_anchor_coef": self.nominal_action_anchor_coef,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"Privileged full-action teacher checkpoint {key} mismatch: "
                    f"expected {value}, got {metadata.get(key)}"
                )
        super().load_state_dict(state_dict)


@dataclass(frozen=True)
class PrivilegedFullActionSACRuntime(OffPolicyRuntime):
    learner_cls: type[Any] | None = PrivilegedFullActionSACLearner
    algo_type: str | None = PRIVILEGED_FULL_ACTION_SAC_ALGO_TYPE
    supports_symmetry: bool = False
    actor_cfg: dict[str, Any] = field(default_factory=dict)

    def validate_training_config(self, cfg: Any) -> None:
        from unilab.algos.torch.fada_context.full_action_formal_protocol import (
            validate_full_action_formal_training_config,
        )

        validate_full_action_formal_training_config(cfg)

    def build_model_kwargs(self, *, obs_dim: int, critic_obs_dim: int) -> dict[str, Any]:
        priv_info_dim = int(self.actor_cfg.get("priv_info_dim", PRIVILEGED_ACTUATOR_STRENGTH_DIM))
        if priv_info_dim != PRIVILEGED_ACTUATOR_STRENGTH_DIM:
            raise ValueError(f"Full-action teacher requires priv_info_dim=29, got {priv_info_dim}.")
        if int(critic_obs_dim) < int(obs_dim) + priv_info_dim:
            raise ValueError(
                "Full-action teacher requires a final 29D actuator-strength critic tail; "
                f"got obs_dim={obs_dim}, critic_obs_dim={critic_obs_dim}."
            )
        return {
            "priv_info_dim": priv_info_dim,
            "priv_info_embed_dim": int(self.actor_cfg.get("priv_info_embed_dim", 16)),
            "priv_mlp_hidden_dims": tuple(
                self.actor_cfg.get("priv_mlp_hidden_dims", (128, 64, 16))
            ),
            "nominal_initialization_checkpoint": str(
                self.actor_cfg["nominal_initialization_checkpoint"]
            ),
            "nominal_action_anchor_coef": float(
                self.actor_cfg.get("nominal_action_anchor_coef", 1.0)
            ),
        }


def resolve_privileged_full_action_sac_runtime(
    rl_cfg: dict[str, Any],
) -> PrivilegedFullActionSACRuntime | None:
    if rl_cfg.get("runtime_impl") != PRIVILEGED_FULL_ACTION_SAC_ALGO_TYPE:
        return None
    actor_cfg_raw = rl_cfg.get("actor", {})
    actor_cfg = actor_cfg_raw if isinstance(actor_cfg_raw, dict) else {}
    if actor_cfg.get("nominal_initialization_checkpoint") in (None, ""):
        raise ValueError(
            "Privileged full-action SAC requires algo.actor.nominal_initialization_checkpoint."
        )
    return PrivilegedFullActionSACRuntime(actor_cfg=dict(actor_cfg))


def load_privileged_full_action_actor_checkpoint(
    actor: PrivilegedFullActionSACActor,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Privileged full-action checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("actor"), dict):
        raise ValueError(f"Privileged full-action checkpoint is missing actor state: {path}")
    metadata = checkpoint.get("privileged_full_action_teacher")
    if not isinstance(metadata, dict) or metadata.get("schema") != _CHECKPOINT_SCHEMA:
        raise ValueError(f"Privileged full-action checkpoint metadata is invalid: {path}")
    expected = {
        "nominal_initialization_sha256": actor.nominal_initialization_sha256,
        "obs_dim": actor.obs_dim,
        "priv_info_dim": actor.priv_info_dim,
        "action_dim": actor.action_dim,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Privileged full-action checkpoint {key} mismatch: "
                f"expected {value}, got {metadata.get(key)}"
            )
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.eval()
    return {
        "path": str(path),
        "sha256": _checkpoint_sha256(path),
        "update_count": int(checkpoint.get("update_count", 0)),
        **dict(metadata),
    }


__all__ = [
    "PRIVILEGED_ACTUATOR_STRENGTH_DIM",
    "PRIVILEGED_FULL_ACTION_SAC_ALGO_TYPE",
    "PrivilegedFullActionSACActor",
    "PrivilegedFullActionSACLearner",
    "PrivilegedFullActionSACRuntime",
    "derive_motor_strength_from_critic_obs",
    "load_privileged_full_action_actor_checkpoint",
    "resolve_privileged_full_action_sac_runtime",
]
