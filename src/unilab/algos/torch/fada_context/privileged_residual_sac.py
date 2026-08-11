"""Privileged residual SAC teacher with a frozen nominal walking policy."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.optim as optim

from unilab.algos.torch.fast_sac.learner import FastSACLearner, SACActor
from unilab.algos.torch.hora.sac_models import HoraSACActor
from unilab.algos.torch.offpolicy.runtime import OffPolicyRuntime

PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE = "privileged_residual_sac"
PRIVILEGED_ACTUATOR_STRENGTH_DIM = 29
_CHECKPOINT_SCHEMA = "unilab_privileged_residual_teacher_v1"


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
    """Read the explicit motor-strength tail without consuming other critic-only fields."""
    actor_dim = int(actor_obs.shape[-1])
    critic_dim = int(critic_obs.shape[-1])
    required_dim = int(priv_info_dim)
    if required_dim <= 0:
        raise ValueError(f"{context} requires positive priv_info_dim, got {required_dim}.")
    if critic_dim < actor_dim + required_dim:
        raise ValueError(
            f"Privileged residual SAC {context} requires a final {required_dim}D motor-strength "
            f"tail; got actor_dim={actor_dim}, critic_dim={critic_dim}."
        )
    return critic_obs[..., -required_dim:]


class PrivilegedResidualSACActor(nn.Module):
    """Frozen nominal SAC actor plus a bounded privileged additive residual."""

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
        nominal_checkpoint_path: str | Path | None = None,
        residual_scale: float = 0.2,
    ) -> None:
        super().__init__()
        if int(priv_info_dim) <= 0:
            raise ValueError(f"priv_info_dim must be positive, got {priv_info_dim}")
        if not 0.0 < float(residual_scale) <= 1.0:
            raise ValueError(f"residual_scale must be in (0, 1], got {residual_scale}")

        self.obs_dim = int(obs_dim)
        self.priv_info_dim = int(priv_info_dim)
        self.action_dim = int(action_dim)
        self.residual_scale = float(residual_scale)
        self.nominal_checkpoint_path: str | None = None
        self.nominal_checkpoint_sha256: str | None = None

        self.nominal_actor = SACActor(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            use_tanh=use_tanh,
            use_layer_norm=use_layer_norm,
            device=device,
        )
        self.residual_actor = HoraSACActor(
            obs_dim=self.obs_dim,
            priv_info_dim=self.priv_info_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            priv_info_embed_dim=priv_info_embed_dim,
            priv_mlp_hidden_dims=tuple(priv_mlp_hidden_dims),
            log_std_max=log_std_max,
            log_std_min=log_std_min,
            use_tanh=use_tanh,
            use_layer_norm=use_layer_norm,
            device=device,
        )
        if nominal_checkpoint_path:
            self.load_nominal_checkpoint(nominal_checkpoint_path, device=device)
        self._freeze_nominal_actor()

    def _freeze_nominal_actor(self) -> None:
        self.nominal_actor.requires_grad_(False)
        self.nominal_actor.eval()

    def load_nominal_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device,
    ) -> None:
        path = Path(checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Nominal SAC checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("actor"), dict):
            raise ValueError(f"Nominal SAC checkpoint is missing actor state: {path}")
        try:
            self.nominal_actor.load_state_dict(checkpoint["actor"], strict=True)
        except RuntimeError as error:
            raise ValueError(
                "Nominal SAC checkpoint actor is incompatible with the declared teacher dimensions: "
                f"{path}"
            ) from error
        self.nominal_checkpoint_path = str(path)
        self.nominal_checkpoint_sha256 = _checkpoint_sha256(path)
        self._freeze_nominal_actor()

    def train(self, mode: bool = True) -> PrivilegedResidualSACActor:
        super().train(mode)
        self.nominal_actor.eval()
        return self

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        if self.nominal_checkpoint_sha256 is not None:
            for name, expected in self.nominal_actor.state_dict().items():
                key = f"nominal_actor.{name}"
                actual = state_dict.get(key)
                if not isinstance(actual, torch.Tensor) or not torch.equal(
                    actual.detach().cpu(),
                    expected.detach().cpu(),
                ):
                    raise ValueError(
                        "Privileged residual actor state does not match the configured nominal "
                        f"checkpoint at tensor {key!r}."
                    )
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._freeze_nominal_actor()
        return result

    def nominal_action(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.nominal_actor.explore(obs, deterministic=True)

    def residual_action(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
        *,
        deterministic: bool,
    ) -> torch.Tensor:
        residual = self.residual_actor.explore(
            obs,
            priv_info,
            deterministic=deterministic,
        )
        return cast(torch.Tensor, residual) * self.residual_scale

    @staticmethod
    def fuse_action(nominal_action: torch.Tensor, delta_action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(nominal_action + delta_action, -1.0, 1.0)

    def forward(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, mean, log_std = self.residual_actor(obs, priv_info)
        delta_action = torch.tanh(mean) * self.residual_scale
        action = self.fuse_action(self.nominal_action(obs), delta_action)
        return action, mean, log_std

    def get_actions_and_log_probs(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual, residual_log_prob, log_std = self.residual_actor.get_actions_and_log_probs(
            obs,
            priv_info,
        )
        action = self.fuse_action(
            self.nominal_action(obs),
            residual * self.residual_scale,
        )
        return action, residual_log_prob, log_std

    @torch.no_grad()
    def explore(
        self,
        obs: torch.Tensor,
        priv_info: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        return self.fuse_action(
            self.nominal_action(obs),
            self.residual_action(obs, priv_info, deterministic=deterministic),
        )

    def as_export_module(self) -> nn.Module:
        actor = self

        class _Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = actor

            def forward(self, obs: torch.Tensor, priv_info: torch.Tensor) -> torch.Tensor:
                action, _, _ = self.base(obs, priv_info)
                return cast(torch.Tensor, action)

        return _Wrapper()


class PrivilegedResidualSACLearner(FastSACLearner):
    """FastSAC learner that optimizes only the privileged residual branch."""

    def __init__(
        self,
        *,
        obs_dim: int,
        critic_obs_dim: int,
        priv_info_dim: int,
        action_dim: int,
        nominal_checkpoint_path: str | Path,
        residual_scale: float = 0.2,
        device: str = "cpu",
        actor_hidden_dim: int = 512,
        priv_info_embed_dim: int = 16,
        priv_mlp_hidden_dims: Sequence[int] = (128, 64, 16),
        log_std_max: float = 0.0,
        log_std_min: float = -5.0,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        actor_lr: float = 3e-4,
        weight_decay: float = 0.001,
        use_symmetry: bool = False,
        symmetry_augmentation: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if use_symmetry or symmetry_augmentation is not None:
            raise ValueError("Privileged residual SAC does not support symmetry augmentation.")
        if int(critic_obs_dim) < int(obs_dim) + int(priv_info_dim):
            raise ValueError(
                "Privileged residual SAC critic observation is missing the declared motor-strength "
                f"tail: obs_dim={obs_dim}, critic_obs_dim={critic_obs_dim}, "
                f"priv_info_dim={priv_info_dim}."
            )
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
        self.actor = PrivilegedResidualSACActor(
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
            nominal_checkpoint_path=nominal_checkpoint_path,
            residual_scale=residual_scale,
        )
        _fused = isinstance(device, str) and device.startswith("cuda")
        self.actor_optimizer = optim.AdamW(
            self.actor.residual_actor.parameters(),
            lr=actor_lr,
            weight_decay=weight_decay,
            fused=_fused,
            betas=(0.9, 0.95),
        )

    def _get_actions_and_log_probs_for_critic(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        strength = derive_motor_strength_from_critic_obs(
            actor_obs,
            critic_obs,
            priv_info_dim=self.priv_info_dim,
            context="critic update",
        )
        actor = cast(PrivilegedResidualSACActor, self.actor)
        return actor.get_actions_and_log_probs(actor_obs, strength)

    def _get_actions_and_log_probs_for_actor(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        strength = derive_motor_strength_from_critic_obs(
            actor_obs,
            critic_obs,
            priv_info_dim=self.priv_info_dim,
            context="actor update",
        )
        actor = cast(PrivilegedResidualSACActor, self.actor)
        return actor.get_actions_and_log_probs(actor_obs, strength)

    def get_state_dict(self) -> dict[str, Any]:
        state = super().get_state_dict()
        actor = cast(PrivilegedResidualSACActor, self.actor)
        state["privileged_residual_teacher"] = {
            "schema": _CHECKPOINT_SCHEMA,
            "nominal_checkpoint_path": actor.nominal_checkpoint_path,
            "nominal_checkpoint_sha256": actor.nominal_checkpoint_sha256,
            "obs_dim": actor.obs_dim,
            "priv_info_dim": actor.priv_info_dim,
            "action_dim": actor.action_dim,
            "residual_scale": actor.residual_scale,
        }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        metadata = state_dict.get("privileged_residual_teacher")
        if not isinstance(metadata, dict) or metadata.get("schema") != _CHECKPOINT_SCHEMA:
            raise ValueError(
                "Privileged residual teacher checkpoint metadata is missing or invalid."
            )
        actor = cast(PrivilegedResidualSACActor, self.actor)
        expected_sha = actor.nominal_checkpoint_sha256
        if metadata.get("nominal_checkpoint_sha256") != expected_sha:
            raise ValueError(
                "Privileged residual teacher nominal checkpoint identity does not match the "
                "configured frozen nominal actor."
            )
        expected_contract = {
            "obs_dim": actor.obs_dim,
            "priv_info_dim": actor.priv_info_dim,
            "action_dim": actor.action_dim,
            "residual_scale": actor.residual_scale,
        }
        for key, expected in expected_contract.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"Privileged residual teacher checkpoint {key} mismatch: "
                    f"expected {expected}, got {metadata.get(key)}"
                )
        super().load_state_dict(state_dict)
        actor._freeze_nominal_actor()


@dataclass(frozen=True)
class PrivilegedResidualSACRuntime(OffPolicyRuntime):
    learner_cls: type[Any] | None = PrivilegedResidualSACLearner
    algo_type: str | None = PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE
    supports_symmetry: bool = False
    actor_cfg: dict[str, Any] = field(default_factory=dict)

    def validate_training_config(self, cfg: Any) -> None:
        from unilab.algos.torch.fada_context.formal_protocol import (
            validate_phase1_formal_training_config,
        )

        validate_phase1_formal_training_config(cfg)

    def build_model_kwargs(self, *, obs_dim: int, critic_obs_dim: int) -> dict[str, Any]:
        priv_info_dim = int(self.actor_cfg.get("priv_info_dim", PRIVILEGED_ACTUATOR_STRENGTH_DIM))
        if priv_info_dim != PRIVILEGED_ACTUATOR_STRENGTH_DIM:
            raise ValueError(
                f"Phase-1 privileged residual SAC requires priv_info_dim=29, got {priv_info_dim}."
            )
        if int(critic_obs_dim) < int(obs_dim) + priv_info_dim:
            raise ValueError(
                "Privileged residual SAC requires critic observations with a final "
                f"{priv_info_dim}D actuator-strength tail; got obs_dim={obs_dim}, "
                f"critic_obs_dim={critic_obs_dim}."
            )
        return {
            "priv_info_dim": priv_info_dim,
            "priv_info_embed_dim": int(self.actor_cfg.get("priv_info_embed_dim", 16)),
            "priv_mlp_hidden_dims": tuple(
                self.actor_cfg.get("priv_mlp_hidden_dims", (128, 64, 16))
            ),
            "nominal_checkpoint_path": str(self.actor_cfg["nominal_checkpoint_path"]),
            "residual_scale": float(self.actor_cfg.get("residual_scale", 0.2)),
        }


def resolve_privileged_residual_sac_runtime(
    rl_cfg: dict[str, Any],
) -> PrivilegedResidualSACRuntime | None:
    if rl_cfg.get("runtime_impl") != PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE:
        return None
    actor_cfg_raw = rl_cfg.get("actor", {})
    actor_cfg = actor_cfg_raw if isinstance(actor_cfg_raw, dict) else {}
    checkpoint_path = actor_cfg.get("nominal_checkpoint_path")
    if checkpoint_path in (None, ""):
        raise ValueError("Privileged residual SAC requires algo.actor.nominal_checkpoint_path.")
    return PrivilegedResidualSACRuntime(actor_cfg=dict(actor_cfg))


def load_privileged_residual_actor_checkpoint(
    actor: PrivilegedResidualSACActor,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    """Strict-load a teacher actor and validate its frozen nominal identity."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Privileged residual teacher checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("actor"), dict):
        raise ValueError(f"Privileged residual teacher checkpoint is missing actor state: {path}")
    metadata = checkpoint.get("privileged_residual_teacher")
    if not isinstance(metadata, dict) or metadata.get("schema") != _CHECKPOINT_SCHEMA:
        raise ValueError(
            f"Privileged residual teacher checkpoint metadata is missing or invalid: {path}"
        )
    expected = {
        "nominal_checkpoint_sha256": actor.nominal_checkpoint_sha256,
        "obs_dim": actor.obs_dim,
        "priv_info_dim": actor.priv_info_dim,
        "action_dim": actor.action_dim,
        "residual_scale": actor.residual_scale,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Privileged residual teacher checkpoint {key} mismatch: "
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
    "PRIVILEGED_RESIDUAL_SAC_ALGO_TYPE",
    "PrivilegedResidualSACActor",
    "PrivilegedResidualSACLearner",
    "PrivilegedResidualSACRuntime",
    "derive_motor_strength_from_critic_obs",
    "load_privileged_residual_actor_checkpoint",
    "resolve_privileged_residual_sac_runtime",
]
