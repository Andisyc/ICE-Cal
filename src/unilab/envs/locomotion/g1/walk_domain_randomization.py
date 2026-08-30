"""G1 joystick locomotion environments."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from unilab.dr import IntervalRandomizationPlan, ResetPlan, ResetRandomizationPayload
from unilab.dr.types import RESET_TERM_KD, RESET_TERM_KP
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_yaw_from_quat
from unilab.envs.locomotion.common.commands import (
    sample_heading_commands,
    sample_height_commands,
)
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.g1.fada_privileged import (
    build_fada_reset_info,
)
from unilab.envs.locomotion.g1.walk_actuator_randomization import (
    sample_actuator_strength_multipliers,
    scale_symmetric_range,
    validate_actuator_strength_config,
)
from unilab.envs.locomotion.g1.walk_config import GaitConstraintConfig
from unilab.envs.locomotion.g1.walk_math import (
    compute_command_active_mask,
    compute_external_command_mask,
    sample_g1_walk_commands,
)
from unilab.envs.locomotion.g1.walk_reset_randomization import (
    freeze_standing_phase,
    sample_gait_phase,
)


class G1WalkDomainRandomizationProvider(LocomotionDRProvider):
    def __init__(
        self,
        *,
        base_kp: np.ndarray | None = None,
        base_kd: np.ndarray | None = None,
        base_body_mass: np.ndarray | None = None,
        base_geom_friction: np.ndarray | None = None,
        ground_geom_id: int | None = None,
    ):
        self._base_kp = base_kp
        self._base_kd = base_kd
        self._base_body_mass = base_body_mass
        self._base_geom_friction = base_geom_friction
        self._ground_geom_id = ground_geom_id
        self._actuator_strength_curriculum_level = 0
        self._actuator_strength_curriculum_pending_episodes = 0
        self._actuator_strength_curriculum_last_brake_step = -(10**9)
        self._actuator_strength_curriculum_resume_after_step = 0

    def _get_base_actuator_gains(self, env: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._base_kp, self._base_kd

    def _get_reset_randomization_baselines(self, env: Any):
        del env
        return self._base_body_mass, self._base_geom_friction, self._ground_geom_id, None

    def build_interval_randomization_plan(
        self, env: Any, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        if not bool(
            getattr(getattr(env.cfg, "fada_privileged_observation", None), "enabled", False)
        ):
            return super().build_interval_randomization_plan(env, step_counter)
        cfg = self.effective_grouped_domain_rand_config(env)
        interval_steps = max(1, round(float(cfg.fada_push_interval_seconds) / env.cfg.ctrl_dt))
        if not cfg.push_robots or step_counter <= 0 or step_counter % interval_steps:
            return None
        limit = float(cfg.fada_max_push_velocity)
        velocity_delta = np.zeros((env._num_envs, 1, 3), dtype=get_global_dtype())
        velocity_delta[:, 0, :2] = np.random.uniform(-limit, limit, size=(env._num_envs, 2))
        return IntervalRandomizationPlan(
            body_ids=np.asarray([env._backend.get_body_id(cfg.push_body_name)], dtype=np.int32),
            body_linear_velocity_delta=velocity_delta,
        )

    def _validated_actuator_strength_config(self, env: Any) -> Any | None:
        strength_cfg = getattr(env.cfg.domain_rand, "actuator_strength", None)
        if strength_cfg is None:
            return None
        return validate_actuator_strength_config(
            strength_cfg,
            expected_actions=int(env._num_action),
        )

    def actuator_strength_curriculum_profile(self, env: Any) -> tuple[int, float, float]:
        strength_cfg = self._validated_actuator_strength_config(env)
        if strength_cfg is None or not bool(getattr(strength_cfg, "curriculum_enabled", False)):
            raise ValueError("actuator strength curriculum is not enabled")
        lows = list(strength_cfg.curriculum_multiplier_lows)
        probabilities = list(strength_cfg.curriculum_nominal_probabilities)
        level = min(self._actuator_strength_curriculum_level, len(lows) - 1)
        return level, float(lows[level]), float(probabilities[level])

    def update_actuator_strength_curriculum(
        self, env: Any, average_episode_length: float, num_completed: int
    ) -> bool:
        strength_cfg = self._validated_actuator_strength_config(env)
        if strength_cfg is None or not bool(getattr(strength_cfg, "curriculum_enabled", False)):
            return False
        self._actuator_strength_curriculum_pending_episodes += int(num_completed)
        if self._actuator_strength_curriculum_pending_episodes < int(
            strength_cfg.curriculum_update_episodes
        ):
            return False
        self._actuator_strength_curriculum_pending_episodes = 0
        previous = self._actuator_strength_curriculum_level
        last = len(strength_cfg.curriculum_multiplier_lows) - 1
        if average_episode_length >= float(strength_cfg.curriculum_promote_threshold):
            self._actuator_strength_curriculum_level = min(previous + 1, last)
        elif average_episode_length <= float(strength_cfg.curriculum_demote_threshold):
            self._actuator_strength_curriculum_level = max(previous - 1, 0)
        return self._actuator_strength_curriculum_level != previous

    def update_iteration_curriculum(
        self, env: Any, iteration: int, terminated_fraction: float
    ) -> bool:
        strength_cfg = self._validated_actuator_strength_config(env)
        if (
            strength_cfg is None
            or str(getattr(strength_cfg, "curriculum_progress_mode", "episode_quality"))
            != "iterations"
        ):
            return False
        boundaries = np.asarray(strength_cfg.curriculum_iteration_boundaries, dtype=np.int64)
        target = int(np.searchsorted(boundaries, int(iteration), side="right") - 1)
        previous = self._actuator_strength_curriculum_level
        if float(terminated_fraction) > float(strength_cfg.curriculum_max_termination_rate):
            cooldown = int(strength_cfg.curriculum_brake_cooldown_steps)
            if int(iteration) - self._actuator_strength_curriculum_last_brake_step >= cooldown:
                self._actuator_strength_curriculum_level = max(previous - 1, 0)
                self._actuator_strength_curriculum_last_brake_step = int(iteration)
                self._actuator_strength_curriculum_resume_after_step = int(iteration) + int(
                    strength_cfg.curriculum_recovery_hold_steps
                )
        elif (
            int(iteration) >= self._actuator_strength_curriculum_resume_after_step
            and previous < target
        ):
            self._actuator_strength_curriculum_level = previous + 1
        return self._actuator_strength_curriculum_level != previous

    def capture_actuator_strength_curriculum_state(self) -> tuple[int, int, int, int]:
        return (
            self._actuator_strength_curriculum_level,
            self._actuator_strength_curriculum_pending_episodes,
            self._actuator_strength_curriculum_last_brake_step,
            self._actuator_strength_curriculum_resume_after_step,
        )

    def restore_actuator_strength_curriculum_state(self, state: tuple[int, ...]) -> None:
        self._actuator_strength_curriculum_level = int(state[0])
        self._actuator_strength_curriculum_pending_episodes = int(state[1])
        self._actuator_strength_curriculum_last_brake_step = (
            int(state[2]) if len(state) > 2 else -(10**9)
        )
        self._actuator_strength_curriculum_resume_after_step = (
            int(state[3]) if len(state) > 3 else 0
        )

    @staticmethod
    def _scale_symmetric_range(values: Any, center: float, scale: float) -> list[float]:
        return scale_symmetric_range(values, center=center, scale=scale)

    def effective_grouped_domain_rand_config(self, env: Any) -> Any:
        cfg = env.cfg.domain_rand
        strength_cfg = self._validated_actuator_strength_config(env)
        if strength_cfg is None or not bool(
            getattr(strength_cfg, "group_curriculum_enabled", False)
        ):
            return cfg
        level, _, _ = self.actuator_strength_curriculum_profile(env)
        scale = float(strength_cfg.group_curriculum_scales[level])
        effective = copy.copy(cfg)
        static_enabled = scale > 0.0
        for field_name in (
            "randomize_kp",
            "randomize_kd",
            "randomize_ground_friction",
            "randomize_base_mass",
            "randomize_body_mass",
            "random_com",
        ):
            setattr(effective, field_name, bool(getattr(cfg, field_name, False)) and static_enabled)
        effective.kp_multiplier_range = self._scale_symmetric_range(
            cfg.kp_multiplier_range, 1.0, scale
        )
        effective.kd_multiplier_range = self._scale_symmetric_range(
            cfg.kd_multiplier_range, 1.0, scale
        )
        effective.ground_friction_multiplier_range = self._scale_symmetric_range(
            cfg.ground_friction_multiplier_range, 1.0, scale
        )
        effective.added_mass_range = self._scale_symmetric_range(cfg.added_mass_range, 0.0, scale)
        effective.body_mass_multiplier_range = self._scale_symmetric_range(
            cfg.body_mass_multiplier_range, 1.0, scale
        )
        effective.com_offset_x = self._scale_symmetric_range(cfg.com_offset_x, 0.0, scale)
        effective.com_offset_y = self._scale_symmetric_range(cfg.com_offset_y, 0.0, scale)
        effective.com_offset_z = self._scale_symmetric_range(cfg.com_offset_z, 0.0, scale)
        effective.randomize_dof_position_bias = bool(cfg.randomize_dof_position_bias) and (
            scale >= 0.6
        )
        effective.dof_position_bias_range = self._scale_symmetric_range(
            cfg.dof_position_bias_range, 0.0, scale
        )
        effective.randomize_control_delay = bool(cfg.randomize_control_delay) and scale >= 0.6
        effective.push_robots = bool(cfg.push_robots) and scale >= 1.0
        return effective

    def _get_effective_domain_rand_config(self, env: Any) -> Any:
        return self.effective_grouped_domain_rand_config(env)

    def _sample_actuator_strength_multipliers(
        self,
        env: Any,
        num_reset: int,
    ) -> np.ndarray | None:
        strength_cfg = self._validated_actuator_strength_config(env)
        if strength_cfg is None:
            return None
        curriculum_profile = (
            self.actuator_strength_curriculum_profile(env)
            if bool(getattr(strength_cfg, "curriculum_enabled", False))
            else None
        )
        return sample_actuator_strength_multipliers(
            strength_cfg,
            num_reset=num_reset,
            expected_actions=int(env._num_action),
            curriculum_profile=curriculum_profile,
        )

    def validate(self, env: Any, capabilities: Any) -> None:
        super().validate(env, capabilities)
        if self._validated_actuator_strength_config(env) is None:
            return
        unsupported = {
            term
            for term in (RESET_TERM_KP, RESET_TERM_KD)
            if not capabilities.supports_reset_term(term)
        }
        if unsupported:
            raise NotImplementedError(
                "G1 actuator strength requires reset-time actuator gain support; "
                f"missing terms: {sorted(unsupported)}"
            )
        expected = int(env._num_action)
        if self._base_kp is None or np.asarray(self._base_kp).shape != (expected,):
            raise ValueError("G1 actuator strength requires one baseline Kp per actuator")
        if self._base_kd is None or np.asarray(self._base_kd).shape != (expected,):
            raise ValueError("G1 actuator strength requires one baseline Kd per actuator")

    def _apply_actuator_strength_to_reset_plan(
        self,
        env: Any,
        env_ids: np.ndarray,
        plan: ResetPlan,
    ) -> ResetPlan:
        multipliers = self._sample_actuator_strength_multipliers(env, len(env_ids))
        if multipliers is None:
            return plan
        if self._base_kp is None or self._base_kd is None:
            raise ValueError("G1 actuator strength baselines were not initialized")

        payload = plan.randomization or ResetRandomizationPayload()
        num_reset = len(env_ids)
        base_kp = np.broadcast_to(np.asarray(self._base_kp), (num_reset, env._num_action))
        base_kd = np.broadcast_to(np.asarray(self._base_kd), (num_reset, env._num_action))
        source_kp = base_kp if payload.kp is None else np.asarray(payload.kp)
        source_kd = base_kd if payload.kd is None else np.asarray(payload.kd)
        payload.kp = np.asarray(source_kp * multipliers, dtype=np.float64)
        payload.kd = np.asarray(source_kd * multipliers, dtype=np.float64)
        plan.randomization = payload
        plan.info_updates["privileged_actuator_strength"] = multipliers.copy()
        return plan

    def _get_qvel_limit(self, env: Any) -> float:
        return float(env.cfg.reset_base_qvel_limit)

    def build_reset_plan(self, env: Any, env_ids: np.ndarray):
        plan = super().build_reset_plan(env, env_ids)
        reward_scales = getattr(getattr(env.cfg, "reward_config", None), "scales", {})
        needs_episode_frame = any(
            float(reward_scales.get(name, 0.0)) != 0.0
            for name in (
                "penalty_lateral_displacement",
                "penalty_yaw_drift",
                "penalty_lateral_corridor_violation",
                "penalty_yaw_corridor_violation",
            )
        ) or bool(getattr(getattr(env.cfg, "forward_progress_termination", None), "enabled", False))
        if needs_episode_frame:
            plan.info_updates["episode_start_base_pos"] = np.asarray(
                plan.qpos[:, :3], dtype=get_global_dtype()
            ).copy()
            plan.info_updates["episode_start_base_yaw"] = np.asarray(
                np_yaw_from_quat(plan.qpos[:, 3:7]), dtype=get_global_dtype()
            ).copy()
        gait_enabled = self._command_gait_mask(env, plan.info_updates["commands"]).astype(bool)
        standing = ~gait_enabled
        if np.any(standing):
            limit = float(getattr(env.cfg, "standing_reset_base_qvel_limit", 0.0))
            if limit <= 0.0:
                plan.qvel[standing, 0:6] = 0.0
            else:
                plan.qvel[standing, 0:6] = np.asarray(
                    np.random.uniform(-limit, limit, size=(int(np.sum(standing)), 6)),
                    dtype=get_global_dtype(),
                )
        plan = self._apply_actuator_strength_to_reset_plan(env, env_ids, plan)
        if not bool(
            getattr(getattr(env.cfg, "fada_privileged_observation", None), "enabled", False)
        ):
            return plan
        num_reset = len(env_ids)
        payload = plan.randomization or ResetRandomizationPayload()
        dtype = get_global_dtype()
        effective_domain_rand = self.effective_grouped_domain_rand_config(env)
        low, high = effective_domain_rand.dof_position_bias_range
        if effective_domain_rand.randomize_dof_position_bias:
            dof_bias = np.random.uniform(low, high, size=(num_reset, env._num_action))
        else:
            dof_bias = np.zeros((num_reset, env._num_action))
        control_delay = (
            np.random.randint(0, 2, size=(num_reset, 1))
            if effective_domain_rand.randomize_control_delay
            else np.zeros((num_reset, 1))
        )
        if any(
            baseline is None
            for baseline in (
                env._fada_base_kp,
                env._fada_base_kd,
                env._fada_base_body_mass,
                env._fada_base_geom_friction,
                env._fada_ground_geom_id,
            )
        ):
            raise ValueError("FADA privilege reset baselines were not initialized")
        plan.info_updates.update(
            build_fada_reset_info(
                payload,
                rows=num_reset,
                num_actions=env._num_action,
                base_kp=env._fada_base_kp,
                base_kd=env._fada_base_kd,
                base_body_mass=env._fada_base_body_mass,
                base_geom_friction=env._fada_base_geom_friction,
                ground_geom_id=env._fada_ground_geom_id,
                dof_position_bias=dof_bias,
                control_delay=control_delay,
                push_interval_seconds=float(env.cfg.domain_rand.fada_push_interval_seconds),
                push_velocity=float(env.cfg.domain_rand.fada_max_push_velocity),
                dtype=np.dtype(dtype),
            )
        )
        return plan

    def _build_extra_info_updates_for_commands(
        self, env: Any, num_reset: int, commands: np.ndarray
    ) -> dict[str, np.ndarray]:
        gait_enabled = self._command_gait_mask(env, commands)
        gait_phase = self._sample_gait_phase(env, num_reset)
        self._apply_standing_reset_phase(env, gait_phase, gait_enabled)
        updates = {"gait_phase": gait_phase, "gait_enabled": gait_enabled}
        if getattr(env.cfg.commands, "heading_command", False):
            updates["heading_commands"] = sample_heading_commands(env, num_reset)
        if getattr(env.cfg.commands, "observe_height_command", False) or getattr(
            env.cfg.commands, "random_height_during_walking", False
        ):
            updates["height_commands"] = sample_height_commands(
                np.random.default_rng(),
                num_reset,
                env.cfg.commands.height_range,
                default_height=float(env.cfg.commands.default_height),
                random_height=bool(env.cfg.commands.random_height_during_walking),
            )
        return updates

    def _command_gait_mask(self, env: Any, commands: np.ndarray) -> np.ndarray:
        reward_cfg = getattr(env.cfg, "reward_config", None)
        gait_cfg = getattr(reward_cfg, "gait_constraint", None)
        if isinstance(gait_cfg, dict):
            gait_cfg = GaitConstraintConfig(**gait_cfg)
        if gait_cfg is None:
            return compute_external_command_mask(commands)
        return compute_command_active_mask(
            commands,
            xy_threshold=float(gait_cfg.command_xy_threshold),
            yaw_threshold=float(gait_cfg.command_yaw_threshold),
        )

    def _apply_standing_reset_phase(
        self, env: Any, gait_phase: np.ndarray, gait_enabled: np.ndarray
    ) -> None:
        reward_cfg = getattr(env.cfg, "reward_config", None)
        gait_cfg = getattr(reward_cfg, "gait_constraint", None)
        if gait_cfg is None:
            return
        if isinstance(gait_cfg, dict):
            enabled = bool(gait_cfg.get("enabled", False))
            freeze = bool(gait_cfg.get("freeze_phase_in_stand_mode", False))
            stand_phase = gait_cfg.get("stand_phase", [np.pi, np.pi])
        else:
            enabled = bool(getattr(gait_cfg, "enabled", False))
            freeze = bool(getattr(gait_cfg, "freeze_phase_in_stand_mode", False))
            stand_phase = getattr(gait_cfg, "stand_phase", [np.pi, np.pi])
        freeze_standing_phase(
            gait_phase,
            gait_enabled=gait_enabled,
            enabled=enabled,
            freeze=freeze,
            stand_phase=stand_phase,
        )

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        return sample_g1_walk_commands(env, num_reset)

    def _sample_gait_phase(self, env: Any, num_reset: int) -> np.ndarray:
        return sample_gait_phase(
            num_reset=num_reset,
            enabled=bool(getattr(env.cfg, "gait_phase_enabled", True)),
            mode=str(env.cfg.gait_phase_init_mode),
        )

    def _compute_reset_obs(
        self,
        env: Any,
        env_ids: Any,
        info_updates: Any,
        linvel: Any,
        gyro: Any,
        gravity: Any,
        dof_pos: Any,
        dof_vel: Any,
    ) -> dict[str, np.ndarray]:
        return env._compute_obs(  # type: ignore[no-any-return]
            info_updates,
            linvel,
            gyro,
            gravity,
            dof_pos,
            dof_vel,
            row_ids=np.asarray(env_ids, dtype=np.int64),
        )
