"""FADA-owned generic privileged SAC runtime selection and preflight."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf

from unilab.algos.torch.distill.fada_privileged_oracle import (
    FADA_PRIVILEGED_SCHEMA,
    FADAOracleCheckpointContract,
    FADAOracleCheckpointGateway,
    canonical_fada_config_sha256,
    validate_fada_oracle_checkpoint_payload,
    validate_no_gait_reward,
)
from unilab.algos.torch.hora.sac_learner import HoraSACLearner
from unilab.algos.torch.offpolicy.runtime import OffPolicyRuntime

FADA_PRIVILEGED_SAC_RUNTIME_IMPL = "privileged_locomotion_sac"


class FADAPrivilegedSACLearner(HoraSACLearner):
    """HORA network implementation with FADA-owned artifact identity."""

    def __init__(
        self,
        *args: Any,
        oracle_lineage_id: str,
        privileged_schema: str = FADA_PRIVILEGED_SCHEMA,
        checkpoint_contract: FADAOracleCheckpointContract | None = None,
        **kwargs: Any,
    ) -> None:
        if not str(oracle_lineage_id).strip():
            raise ValueError("oracle_lineage_id must be non-empty")
        if privileged_schema != FADA_PRIVILEGED_SCHEMA:
            raise ValueError("privileged Oracle schema mismatch")
        self.oracle_lineage_id = str(oracle_lineage_id)
        self.privileged_schema = privileged_schema
        self.checkpoint_contract = checkpoint_contract
        super().__init__(*args, **kwargs)
        if checkpoint_contract is not None:
            if checkpoint_contract.oracle_lineage_id != self.oracle_lineage_id:
                raise ValueError("checkpoint contract lineage mismatch")
            if checkpoint_contract.obs_dim != int(self.actor.obs_dim):
                raise ValueError("checkpoint contract obs_dim mismatch")
            if checkpoint_contract.critic_obs_dim != int(self.critic_obs_dim):
                raise ValueError("checkpoint contract critic_obs_dim mismatch")
            if checkpoint_contract.action_dim != int(self.actor.action_dim):
                raise ValueError("checkpoint contract action_dim mismatch")

    def get_state_dict(self) -> dict[str, Any]:
        return super().get_state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.checkpoint_contract is None:
            raise ValueError("FADA Oracle checkpoint load requires a sealed checkpoint contract")
        metadata = state_dict.get("fada_privileged_oracle")
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint missing fada_privileged_oracle identity")
        iteration = metadata.get("iteration")
        if not isinstance(iteration, int):
            raise ValueError("checkpoint identity iteration must be an integer")
        validate_fada_oracle_checkpoint_payload(
            state_dict,
            self.checkpoint_contract,
            expected_iteration=iteration,
        )
        super().load_state_dict(state_dict)


def _object_items(value: Any) -> dict[str, Any]:
    if OmegaConf.is_config(value):
        resolved = OmegaConf.to_container(value, resolve=True)
        return dict(resolved) if isinstance(resolved, dict) else {}
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(vars(value))
    except TypeError:
        return {}


@dataclass(frozen=True)
class FADAPrivilegedSACRuntime(OffPolicyRuntime):
    learner_cls: type[Any] | None = FADAPrivilegedSACLearner
    algo_type: str | None = FADA_PRIVILEGED_SAC_RUNTIME_IMPL
    supports_symmetry: bool = False
    actor_cfg: dict[str, Any] = field(default_factory=dict)

    def build_model_kwargs(self, *, obs_dim: int, critic_obs_dim: int) -> dict[str, Any]:
        priv_info_dim = int(critic_obs_dim - obs_dim)
        if priv_info_dim <= 0:
            raise ValueError("privileged_locomotion_sac requires a privileged critic tail")
        return {
            "priv_info_dim": priv_info_dim,
            "priv_info_embed_dim": int(self.actor_cfg.get("priv_info_embed_dim", 32)),
            "priv_mlp_hidden_dims": tuple(
                self.actor_cfg.get("priv_mlp_hidden_dims", (256, 128, 32))
            ),
            "priv_info_normalization": bool(self.actor_cfg.get("priv_info_normalization", True)),
            "oracle_lineage_id": str(self.actor_cfg.get("oracle_lineage_id", "")),
            "privileged_schema": FADA_PRIVILEGED_SCHEMA,
        }

    def build_training_model_kwargs(
        self,
        *,
        cfg: Any,
        env: Any,
        obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
    ) -> dict[str, Any]:
        kwargs = self.build_model_kwargs(obs_dim=obs_dim, critic_obs_dim=critic_obs_dim)
        kwargs["obs_normalization"] = bool(cfg.algo.obs_normalization)
        kwargs["v_min"] = float(cfg.algo.value_support_min)
        kwargs["v_max"] = float(cfg.algo.value_support_max)
        layout_identity = env.get_fada_privileged_checkpoint_identity()

        def resolved(value: Any) -> Any:
            if OmegaConf.is_config(value):
                return OmegaConf.to_container(value, resolve=True)
            return value

        raw_scale = resolved(cfg.env.control_config.action_scale)
        if isinstance(raw_scale, (list, tuple)):
            action_scale = tuple(float(value) for value in raw_scale)
        else:
            action_scale = (float(raw_scale),)
        config_hashes = tuple(
            (name, canonical_fada_config_sha256(value))
            for name, value in (
                ("algo", resolved(cfg.algo)),
                ("env", resolved(cfg.env)),
                ("reward", resolved(cfg.reward)),
                (
                    "training",
                    {
                        "task_name": str(cfg.training.task_name),
                        "sim_backend": str(cfg.training.sim_backend),
                        "seed": int(cfg.algo.seed),
                    },
                ),
            )
        )
        kwargs["checkpoint_contract"] = FADAOracleCheckpointContract(
            oracle_lineage_id=str(self.actor_cfg.get("oracle_lineage_id", "")),
            privileged_schema=FADA_PRIVILEGED_SCHEMA,
            task_name=str(cfg.training.task_name),
            backend=str(cfg.training.sim_backend),
            action_scale=action_scale,
            seed=int(cfg.algo.seed),
            obs_dim=int(obs_dim),
            critic_obs_dim=int(critic_obs_dim),
            action_dim=int(action_dim),
            body_names=tuple(layout_identity.body_names),
            actuated_joint_names=tuple(layout_identity.actuated_joint_names),
            privileged_field_slices=tuple(layout_identity.field_slices),
            asset_sha256=str(layout_identity.asset_sha256),
            config_hashes=config_hashes,
        )
        return kwargs

    def build_checkpoint_saver(self, learner: Any) -> Any:
        contract = learner.checkpoint_contract
        if not isinstance(contract, FADAOracleCheckpointContract):
            raise ValueError("privileged_locomotion_sac learner lacks checkpoint contract")
        return FADAOracleCheckpointGateway(contract).save

    def validate_training_config(self, cfg: Any) -> None:
        if getattr(cfg.training, "task_name", None) != "G1WalkFlat":
            raise ValueError("privileged_locomotion_sac requires task G1WalkFlat")
        if getattr(cfg.training, "sim_backend", None) != "mujoco":
            raise ValueError("privileged_locomotion_sac Unit A requires MuJoCo")
        if int(getattr(cfg.algo, "max_iterations", -1)) != 5000:
            raise ValueError("privileged_locomotion_sac requires max_iterations=5000")
        if int(getattr(cfg.algo, "save_interval", -1)) != 240:
            raise ValueError("privileged_locomotion_sac requires save_interval=240")
        if bool(getattr(cfg.algo, "use_symmetry", True)):
            raise ValueError("privileged_locomotion_sac requires use_symmetry=false")
        if float(getattr(cfg.algo, "gamma", 0.0)) != 0.99:
            raise ValueError("privileged_locomotion_sac requires gamma=0.99")
        value_support = (
            float(getattr(cfg.algo, "value_support_min", 0.0)),
            float(getattr(cfg.algo, "value_support_max", 0.0)),
        )
        if value_support != (-30.0, 30.0):
            raise ValueError("privileged_locomotion_sac requires value support [-30, 30]")
        if not bool(getattr(cfg.algo, "obs_normalization", False)):
            raise ValueError("privileged_locomotion_sac requires observation normalization")
        actor_cfg = getattr(cfg.algo, "actor", None)
        lineage_id = (
            actor_cfg.get("oracle_lineage_id", "")
            if isinstance(actor_cfg, dict)
            else getattr(actor_cfg, "oracle_lineage_id", "")
        )
        if not str(lineage_id).strip() or str(lineage_id) == "REQUIRED":
            raise ValueError("privileged_locomotion_sac requires actor.oracle_lineage_id")
        actor_items = _object_items(actor_cfg)
        if not bool(actor_items.get("priv_info_normalization", False)):
            raise ValueError("privileged_locomotion_sac requires privileged normalization")
        privileged_cfg = getattr(cfg.env, "fada_privileged_observation", None)
        if privileged_cfg is None or not bool(getattr(privileged_cfg, "enabled", False)):
            raise ValueError("g1_fada_privileged_v1 observation must be enabled")
        if getattr(privileged_cfg, "schema", None) != FADA_PRIVILEGED_SCHEMA:
            raise ValueError("g1_fada_privileged_v1 schema mismatch")
        if bool(getattr(cfg.env, "mode_observation", False)):
            raise ValueError("privileged_locomotion_sac forbids mode observation")
        commands_cfg = getattr(cfg.env, "commands", None)
        if float(getattr(cfg.env, "ctrl_dt", 0.0)) != 0.02:
            raise ValueError("privileged_locomotion_sac requires ctrl_dt=0.02")
        if float(getattr(commands_cfg, "rel_transition_envs", 0.0)) != 0.0:
            raise ValueError("privileged_locomotion_sac forbids transition-mode samples")
        if float(getattr(commands_cfg, "resampling_time", -1.0)) != 0.0:
            raise ValueError("privileged_locomotion_sac forbids command resampling")
        if bool(getattr(commands_cfg, "heading_command", True)):
            raise ValueError("privileged_locomotion_sac forbids heading command mode")
        vel_limit = getattr(commands_cfg, "vel_limit", None)
        if list(vel_limit or []) != [[-0.6, -0.4, -0.8], [1.0, 0.4, 0.8]]:
            raise ValueError("privileged_locomotion_sac command vel_limit mismatch")
        curriculum_cfg = getattr(cfg.env, "curriculum", None)
        if bool(getattr(curriculum_cfg, "enabled", True)):
            raise ValueError("privileged_locomotion_sac forbids penalty curriculum")
        validate_no_gait_reward(_object_items(cfg.reward.scales))
        reward_mode = getattr(cfg.reward, "mode", None)
        if bool(getattr(reward_mode, "enabled", False)):
            raise ValueError("privileged_locomotion_sac forbids reward.mode")
        gait_constraint = getattr(cfg.reward, "gait_constraint", None)
        penalty_scale = (
            gait_constraint.get("penalty_scale", 0.0)
            if isinstance(gait_constraint, dict)
            else getattr(gait_constraint, "penalty_scale", 0.0)
        )
        if float(penalty_scale) != 0.0:
            raise ValueError("privileged_locomotion_sac requires gait constraint penalty_scale=0")
        if bool(getattr(gait_constraint, "enabled", False)):
            raise ValueError("privileged_locomotion_sac forbids gait constraint mode")


def resolve_privileged_locomotion_sac_runtime(
    rl_cfg: dict[str, Any],
) -> FADAPrivilegedSACRuntime | None:
    if rl_cfg.get("runtime_impl") != FADA_PRIVILEGED_SAC_RUNTIME_IMPL:
        return None
    actor_cfg = rl_cfg.get("actor", {})
    return FADAPrivilegedSACRuntime(
        actor_cfg=dict(actor_cfg) if isinstance(actor_cfg, dict) else {}
    )


__all__ = [
    "FADA_PRIVILEGED_SAC_RUNTIME_IMPL",
    "FADAPrivilegedSACLearner",
    "FADAPrivilegedSACRuntime",
    "resolve_privileged_locomotion_sac_runtime",
]
