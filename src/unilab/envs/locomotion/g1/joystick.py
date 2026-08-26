"""G1 joystick locomotion environments."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from etils import epath

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.augmentation import SymmetryObsLayout
from unilab.base.backend import create_backend
from unilab.base.curriculum import EpisodeLengthTracker, PenaltyCurriculum
from unilab.base.np_env import NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr import IntervalRandomizationPlan, ResetPlan, ResetRandomizationPayload
from unilab.dr.types import RESET_TERM_KD, RESET_TERM_KP
from unilab.dtype_config import get_global_dtype
from unilab.envs.common.rotation import np_wrap_to_pi, np_yaw_from_quat
from unilab.envs.locomotion.common import rewards
from unilab.envs.locomotion.common.commands import (
    Commands,
    apply_heading_yaw_feedback,
    sample_heading_commands,
    sample_height_commands,
    sample_velocity_commands,
    zero_small_xy_commands,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.common.dr_provider import LocomotionDRProvider
from unilab.envs.locomotion.common.rewards import RewardContext
from unilab.envs.locomotion.g1.base import G1BaseCfg, G1BaseEnv
from unilab.envs.locomotion.g1.calibration_fault import (
    G1ActionExecutionFaultConfig,
    apply_action_execution_fault,
)
from unilab.envs.locomotion.g1.fada_privileged import (
    DOF_POSITION_BIAS_LIMIT_RAD,
    TORQUE_RFI_FRACTION,
    G1FADAPrivilegedCheckpointLayoutIdentity,
    G1FADAPrivilegedObservationConfig,
    apply_fada_pd_target_perturbation,
    build_fada_reset_info,
    build_g1_fada_checkpoint_layout_identity,
    pack_fada_runtime_observation,
)


@dataclass
class G1ActuatorStrengthConfig:
    """Optional per-actuator position-servo gain multipliers.

    This is a gain-based approximation of actuator effectiveness for controlled
    simulation experiments. It does not model a measured torque-limit curve.
    """

    enabled: bool = False
    multipliers: list[float] = field(default_factory=list)
    sampling_mode: str = "fixed"
    candidate_actuator_indices: list[int] = field(default_factory=list)
    multiplier_range: list[float] = field(default_factory=lambda: [1.0, 1.0])
    nominal_probability: float = 0.0
    include_in_critic_obs: bool = False


@dataclass
class G1DomainRandConfig(DomainRandConfig):
    randomize_kp: bool = True
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    randomize_kd: bool = True
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.9, 1.1])

    actuator_strength: G1ActuatorStrengthConfig = field(default_factory=G1ActuatorStrengthConfig)
    randomize_dof_position_bias: bool = False
    dof_position_bias_range: list[float] = field(
        default_factory=lambda: [-DOF_POSITION_BIAS_LIMIT_RAD, DOF_POSITION_BIAS_LIMIT_RAD]
    )
    torque_rfi_fraction: float = 0.0
    randomize_control_delay: bool = False
    com_offset_y: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    com_offset_z: list[float] = field(default_factory=lambda: [-0.05, 0.05])
    fada_push_interval_seconds: float = 7.5
    fada_max_push_velocity: float = 0.8


@dataclass
class InitState:
    pos = [0.0, 0.0, 0.754]


def sample_gait_phase_pairs(rng, num_samples: int, mode: str) -> np.ndarray:
    if mode == "independent":
        return np.asarray(
            np.column_stack(
                [
                    rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,)),
                    rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,)),
                ]
            ),
            dtype=get_global_dtype(),
        )

    phase = rng.uniform(0.0, 2.0 * np.pi, size=(num_samples,))
    return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())


def sample_reset_base_qvel(rng, num_samples: int, limit: float) -> np.ndarray:
    return np.asarray(rng.uniform(-limit, limit, size=(num_samples, 6)), dtype=get_global_dtype())


def sample_g1_walk_commands(env: Any, num_samples: int) -> np.ndarray:
    low = np.asarray(env.cfg.commands.vel_limit[0], dtype=get_global_dtype())
    high = np.asarray(env.cfg.commands.vel_limit[1], dtype=get_global_dtype())
    commands = sample_velocity_commands(np.random.default_rng(), num_samples, low, high)
    zero_small_xy_commands(
        commands,
        threshold=float(getattr(env.cfg.commands, "small_xy_threshold", 0.0)),
    )
    standing_prob = float(getattr(env.cfg.commands, "rel_standing_envs", 0.0))
    transition_prob = float(getattr(env.cfg.commands, "rel_transition_envs", 0.0))
    standing_prob = min(max(standing_prob, 0.0), 1.0)
    transition_prob = min(max(transition_prob, 0.0), max(1.0 - standing_prob, 0.0))
    draw = np.random.uniform(size=(num_samples,))
    if transition_prob > 0.0:
        low = np.asarray(env.cfg.commands.transition_vel_limit[0], dtype=get_global_dtype())
        high = np.asarray(env.cfg.commands.transition_vel_limit[1], dtype=get_global_dtype())
        transition = (draw >= standing_prob) & (draw < standing_prob + transition_prob)
        if np.any(transition):
            commands[transition] = sample_velocity_commands(
                np.random.default_rng(), int(np.sum(transition)), low, high
            )
    if standing_prob > 0.0:
        commands[draw < standing_prob] = 0.0
    if getattr(env.cfg.commands, "heading_command", False):
        commands[:, 2] = 0.0
    return commands


def build_upper_body_pose_weights(pose_weights: list[float]) -> np.ndarray:
    weights = np.asarray(pose_weights, dtype=get_global_dtype()).copy()
    weights[:12] = 0.0
    return np.asarray(weights, dtype=get_global_dtype())


def compute_feet_phase_height_targets(
    gait_phase: np.ndarray, swing_height: float
) -> tuple[np.ndarray, np.ndarray]:
    def cubic_bezier_height(phi: np.ndarray, swing_height: float) -> np.ndarray:
        phi_normalized = np.fmod(phi + np.pi, 2 * np.pi) - np.pi
        x = (phi_normalized + np.pi) / (2 * np.pi)

        def cubic_bezier_interpolation(
            y_start: np.ndarray, y_end: np.ndarray, t: np.ndarray
        ) -> np.ndarray:
            y_diff = y_end - y_start
            bezier = t**3 + 3 * (t**2 * (1 - t))
            return np.asarray(y_start + y_diff * bezier, dtype=get_global_dtype())

        stance = cubic_bezier_interpolation(np.zeros_like(x), np.full_like(x, swing_height), 2 * x)
        swing = cubic_bezier_interpolation(
            np.full_like(x, swing_height), np.zeros_like(x), 2 * x - 1
        )
        return np.where(x <= 0.5, stance, swing)

    left_target = cubic_bezier_height(gait_phase[:, 0], swing_height)
    right_target = cubic_bezier_height(gait_phase[:, 1], swing_height)
    return left_target, right_target


LEFT_FOOT_CONTACT_SENSORS = [f"left_foot_contact_{i}" for i in range(4)]
RIGHT_FOOT_CONTACT_SENSORS = [f"right_foot_contact_{i}" for i in range(4)]


def _debug_env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _debug_env_int(name: str, *, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _debug_stats(name: str, value: Any) -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return f"{name}: empty"
    finite = np.isfinite(arr)
    finite_arr = arr[finite]
    if finite_arr.size == 0:
        return f"{name}: shape={arr.shape} finite=0/{arr.size}"
    return (
        f"{name}: shape={arr.shape} finite={finite_arr.size}/{arr.size} "
        f"mean={float(np.mean(finite_arr)):.6g} "
        f"min={float(np.min(finite_arr)):.6g} "
        f"max={float(np.max(finite_arr)):.6g} "
        f"l1_mean={float(np.mean(np.sum(np.abs(np.atleast_2d(arr)), axis=1))):.6g} "
        f"max_abs={float(np.max(np.abs(finite_arr))):.6g}"
    )


def _debug_head(name: str, value: Any, *, count: int = 8) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return f"{name}: {float(arr):.6g}"
    row = arr[0] if arr.ndim > 1 else arr
    head = np.asarray(row[:count], dtype=np.float64)
    return f"{name}[0,:{count}]: {np.array2string(head, precision=4, suppress_small=False)}"


def _scalarize_sensor_values(sensor_values: np.ndarray) -> np.ndarray:
    sensor_array = np.asarray(sensor_values, dtype=get_global_dtype())
    if sensor_array.ndim == 1:
        return sensor_array
    if sensor_array.ndim == 2 and sensor_array.shape[1] == 1:
        return sensor_array[:, 0]
    raise ValueError(f"Expected scalar sensor values, got shape {sensor_array.shape}")


def compute_aggregated_foot_contact(backend: Any, sensor_names: list[str]) -> np.ndarray:
    contacts = [_scalarize_sensor_values(backend.get_sensor_data(name)) for name in sensor_names]
    return np.asarray(np.any(np.stack(contacts, axis=1) > 0.5, axis=1), dtype=np.bool_)


def compute_aggregated_foot_contact_count(backend: Any, sensor_names: list[str]) -> np.ndarray:
    contacts = [_scalarize_sensor_values(backend.get_sensor_data(name)) for name in sensor_names]
    return np.sum(np.stack(contacts, axis=1) > 0.5, axis=1).astype(get_global_dtype())


def compute_feet_phase_contact_targets(
    gait_phase: np.ndarray, swing_height: float
) -> tuple[np.ndarray, np.ndarray]:
    left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
    contact_height_threshold = swing_height * 0.5
    return left_target <= contact_height_threshold, right_target <= contact_height_threshold


def compute_forward_speed_gate(linvel: np.ndarray, min_forward_speed: float) -> np.ndarray:
    forward_speed = np.maximum(linvel[:, 0], 0.0)
    return np.asarray(forward_speed >= min_forward_speed, dtype=get_global_dtype())


def compute_forward_command_mask(commands: np.ndarray) -> np.ndarray:
    return np.asarray(np.maximum(commands[:, 0], 0.0) > 1.0e-6, dtype=get_global_dtype())


def compute_command_active_mask(
    commands: np.ndarray, *, xy_threshold: float, yaw_threshold: float
) -> np.ndarray:
    xy_norm = np.linalg.norm(commands[:, :2], axis=1)
    yaw_abs = np.abs(commands[:, 2])
    return np.asarray(
        (xy_norm > xy_threshold) | (yaw_abs > yaw_threshold), dtype=get_global_dtype()
    )


def compute_external_command_mask(commands: np.ndarray, *, epsilon: float = 1.0e-6) -> np.ndarray:
    return np.asarray(np.any(np.abs(commands) > epsilon, axis=1), dtype=get_global_dtype())


def compute_tracking_gate(
    commands: np.ndarray,
    linvel: np.ndarray,
    gyro: np.ndarray,
    *,
    tracking_sigma: float,
    threshold: float,
) -> np.ndarray:
    lin_error = np.sum(np.square(commands[:, :2] - linvel[:, :2]), axis=1)
    yaw_error = np.square(commands[:, 2] - gyro[:, 2])
    tracking_score = np.exp(-(lin_error + yaw_error) / tracking_sigma)
    return np.asarray(tracking_score > threshold, dtype=get_global_dtype())


def compute_gait_phase_height_violation(
    left_foot_z: np.ndarray,
    right_foot_z: np.ndarray,
    gait_phase: np.ndarray,
    swing_height: float,
) -> np.ndarray:
    left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
    return np.asarray(
        np.square(left_foot_z - left_target) + np.square(right_foot_z - right_target),
        dtype=get_global_dtype(),
    )


def compute_gait_phase_contrast_violation(
    left_foot_z: np.ndarray,
    right_foot_z: np.ndarray,
    gait_phase: np.ndarray,
    swing_height: float,
) -> np.ndarray:
    left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
    actual_delta = left_foot_z - right_foot_z
    target_delta = left_target - right_target
    return np.asarray(np.square(actual_delta - target_delta), dtype=get_global_dtype())


def compute_gait_phase_contact_violation(
    left_contact: np.ndarray,
    right_contact: np.ndarray,
    gait_phase: np.ndarray,
    swing_height: float,
) -> np.ndarray:
    left_target, right_target = compute_feet_phase_contact_targets(gait_phase, swing_height)
    left_error = np.asarray(left_contact != left_target, dtype=get_global_dtype())
    right_error = np.asarray(right_contact != right_target, dtype=get_global_dtype())
    return np.asarray(0.5 * (left_error + right_error), dtype=get_global_dtype())


@dataclass
class GaitConstraintConfig:
    enabled: bool = False
    command_xy_threshold: float = 0.05
    command_yaw_threshold: float = 0.05
    height_weight: float = 1.0
    contrast_weight: float = 1.0
    contact_weight: float = 1.0
    epsilon: float = 0.02
    penalty_scale: float = 1.0
    apply_in_stand_mode: bool = False
    apply_when_tracking: bool = False
    tracking_threshold: float = 0.3
    freeze_phase_in_stand_mode: bool = False
    stand_phase: list[float] = field(default_factory=lambda: [math.pi, math.pi])


@dataclass
class RewardModeConfig:
    enabled: bool = False
    standing_enabled: bool = False
    balance_common_terms: list[str] = field(default_factory=list)
    stand_terms: list[str] = field(default_factory=list)
    stand_recovery_terms: list[str] = field(default_factory=list)
    walk_terms: list[str] = field(default_factory=list)
    stand_scale_overrides: dict[str, float] = field(default_factory=dict)
    stand_recovery_scale_overrides: dict[str, float] = field(default_factory=dict)
    walk_scale_overrides: dict[str, float] = field(default_factory=dict)


@dataclass
class G1RewardConfig:
    scales: dict[str, float]
    tracking_sigma: float
    gait_frequency: float
    feet_phase_swing_height: float
    feet_phase_tracking_sigma: float
    base_height_target: float
    min_base_height: float
    max_tilt_deg: float
    min_forward_speed_for_gait_reward: float = 0.0
    stand_recovery_lin_vel_xy_threshold: float = 0.2
    stand_recovery_tilt_deg_threshold: float = 8.0
    close_feet_threshold: float = 0.15
    straight_line_lateral_tolerance_m: float = 0.10
    straight_line_yaw_tolerance_rad: float = 0.10
    stand_feet_x_target: float = 0.0
    stand_feet_y_width_target: float = 0.21
    stand_base_feet_center_x_target: float = 0.0
    stand_base_feet_center_y_target: float = 0.0
    stand_foot_contact_balance_epsilon: float = 1.0e-6
    stand_support_height_margin: float = 0.02
    gait_constraint: GaitConstraintConfig | dict[str, Any] = field(
        default_factory=GaitConstraintConfig
    )
    mode: RewardModeConfig | dict[str, Any] = field(default_factory=RewardModeConfig)
    pose_weights: list[float] = field(
        default_factory=lambda: [
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            0.01,
            1.0,
            5.0,
            0.01,
            5.0,
            5.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
            50.0,
        ]
    )

    def __post_init__(self) -> None:
        if isinstance(self.gait_constraint, dict):
            self.gait_constraint = GaitConstraintConfig(**self.gait_constraint)
        if isinstance(self.mode, dict):
            self.mode = RewardModeConfig(**self.mode)


@dataclass
class G1WalkLegacyRewardConfig(G1RewardConfig):
    pass


@dataclass
class CurriculumConfig:
    enabled: bool = False
    initial_scale: float = 0.5
    min_scale: float = 0.5
    max_scale: float = 1.0
    level_down_threshold: float = 150.0
    level_up_threshold: float = 750.0
    degree: float = 0.001


@dataclass
class ForwardProgressTerminationConfig:
    enabled: bool = False
    grace_steps: int = 50
    min_command_forward_speed: float = 0.1
    min_average_forward_speed: float = 0.2


def compute_forward_progress_failure(
    current_position: np.ndarray,
    initial_position: np.ndarray,
    initial_yaw: np.ndarray,
    steps_before_increment: np.ndarray,
    commands: np.ndarray,
    *,
    ctrl_dt: float,
    grace_steps: int,
    min_command_forward_speed: float,
    min_average_forward_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return progress-failure mask and reset-frame episode-average forward speed."""
    current = np.asarray(current_position, dtype=get_global_dtype())
    initial = np.asarray(initial_position, dtype=get_global_dtype())
    yaw = np.asarray(initial_yaw, dtype=get_global_dtype())
    steps = np.asarray(steps_before_increment)
    command = np.asarray(commands, dtype=get_global_dtype())
    batch = int(current.shape[0])
    if current.shape != (batch, 3) or initial.shape != (batch, 3):
        raise ValueError("forward-progress positions must both have shape (N, 3)")
    if yaw.shape != (batch,) or steps.shape != (batch,) or command.shape != (batch, 3):
        raise ValueError("forward-progress yaw/steps/commands shapes do not match the batch")
    if float(ctrl_dt) <= 0.0 or int(grace_steps) <= 0:
        raise ValueError("forward-progress ctrl_dt and grace_steps must be positive")

    delta = current[:, :2] - initial[:, :2]
    forward_displacement = np.cos(yaw) * delta[:, 0] + np.sin(yaw) * delta[:, 1]
    completed_steps = steps.astype(np.int64, copy=False) + 1
    elapsed_seconds = completed_steps.astype(get_global_dtype()) * float(ctrl_dt)
    average_forward_speed = forward_displacement / elapsed_seconds
    failure = (
        (completed_steps >= int(grace_steps))
        & (command[:, 0] >= float(min_command_forward_speed))
        & (average_forward_speed < float(min_average_forward_speed) - 1.0e-6)
    )
    return np.asarray(failure, dtype=np.bool_), np.asarray(
        average_forward_speed, dtype=get_global_dtype()
    )


@dataclass
class G1WalkEnvCfg(G1BaseCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
        )
    )
    max_episode_seconds: float = 20.0
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    reward_config: G1RewardConfig | None = None
    domain_rand: G1DomainRandConfig = field(default_factory=G1DomainRandConfig)
    gait_phase_init_mode: str = "offset_phase"
    mode_observation: bool = False
    reset_base_qvel_limit: float = 0.5
    standing_reset_base_qvel_limit: float = 0.0
    stand_action_authority: bool = False
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    forward_progress_termination: ForwardProgressTerminationConfig = field(
        default_factory=ForwardProgressTerminationConfig
    )
    action_execution_fault: G1ActionExecutionFaultConfig | None = None
    fada_privileged_observation: G1FADAPrivilegedObservationConfig = field(
        default_factory=G1FADAPrivilegedObservationConfig
    )

    def validate(self) -> None:
        super().validate()
        if self.action_execution_fault is not None:
            self.action_execution_fault.validate()
        if self.fada_privileged_observation.enabled:
            if self.fada_privileged_observation.schema != "g1_fada_privileged_v1":
                raise ValueError("unsupported FADA privileged observation schema")
            bias_range = np.asarray(self.domain_rand.dof_position_bias_range, dtype=np.float64)
            if (
                bias_range.shape != (2,)
                or bias_range[0] < -DOF_POSITION_BIAS_LIMIT_RAD
                or bias_range[1] > DOF_POSITION_BIAS_LIMIT_RAD
            ):
                raise ValueError(
                    "FADA DoF position bias range exceeds the confirmed moderate limit"
                )
            if not 0.0 <= float(self.domain_rand.torque_rfi_fraction) <= TORQUE_RFI_FRACTION:
                raise ValueError("FADA torque RFI fraction exceeds the confirmed moderate limit")
            if self.domain_rand.push_robots:
                if not 5.0 <= float(self.domain_rand.fada_push_interval_seconds) <= 10.0:
                    raise ValueError("FADA push interval must remain in [5, 10] seconds")
                if not 0.1 <= float(self.domain_rand.fada_max_push_velocity) <= 1.5:
                    raise ValueError("FADA max push velocity must remain in [0.1, 1.5] m/s")
                if not self.domain_rand.push_body_name:
                    raise ValueError("FADA velocity push requires push_body_name")


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
        cfg = env.cfg.domain_rand
        interval_steps = max(1, round(float(cfg.fada_push_interval_seconds) / env.cfg.ctrl_dt))
        if not cfg.push_robots or step_counter % interval_steps:
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
        enabled = bool(getattr(strength_cfg, "enabled", False))
        include_in_critic = bool(getattr(strength_cfg, "include_in_critic_obs", False))
        if not enabled:
            if include_in_critic:
                raise ValueError(
                    "domain_rand.actuator_strength.include_in_critic_obs requires enabled=true"
                )
            return None

        expected = int(env._num_action)
        sampling_mode = str(getattr(strength_cfg, "sampling_mode", "fixed"))
        if sampling_mode == "fixed":
            multipliers = np.asarray(strength_cfg.multipliers, dtype=np.float64)
            if multipliers.shape != (expected,):
                raise ValueError(
                    "domain_rand.actuator_strength requires exactly "
                    f"{expected} multipliers, got shape {multipliers.shape}"
                )
            if not np.isfinite(multipliers).all():
                raise ValueError("domain_rand.actuator_strength multipliers must be finite")
            if np.any(multipliers <= 0.0) or np.any(multipliers > 1.0):
                raise ValueError(
                    "domain_rand.actuator_strength multipliers must be in the interval (0, 1]"
                )
            if list(getattr(strength_cfg, "candidate_actuator_indices", [])):
                raise ValueError("fixed actuator strength cannot define candidate_actuator_indices")
            return strength_cfg

        if sampling_mode != "single_candidate":
            raise ValueError(
                "domain_rand.actuator_strength.sampling_mode must be 'fixed' or "
                f"'single_candidate', got {sampling_mode!r}"
            )
        if list(getattr(strength_cfg, "multipliers", [])):
            raise ValueError("single_candidate actuator strength cannot define fixed multipliers")
        candidates = np.asarray(strength_cfg.candidate_actuator_indices, dtype=np.int64)
        if candidates.ndim != 1 or candidates.size == 0:
            raise ValueError(
                "single_candidate actuator strength requires candidate_actuator_indices"
            )
        if np.unique(candidates).size != candidates.size:
            raise ValueError("actuator strength candidate indices must be unique")
        if np.any(candidates < 0) or np.any(candidates >= expected):
            raise ValueError(f"actuator strength candidate indices must be in [0, {expected})")
        multiplier_range = np.asarray(strength_cfg.multiplier_range, dtype=np.float64)
        if multiplier_range.shape != (2,) or not np.isfinite(multiplier_range).all():
            raise ValueError("actuator strength multiplier_range must contain two finite values")
        low, high = multiplier_range.tolist()
        if low <= 0.0 or high < low or high > 1.0:
            raise ValueError("actuator strength multiplier_range must satisfy 0 < low <= high <= 1")
        nominal_probability = float(strength_cfg.nominal_probability)
        if not np.isfinite(nominal_probability) or not 0.0 <= nominal_probability <= 1.0:
            raise ValueError("actuator strength nominal_probability must be in [0, 1]")
        return strength_cfg

    def _sample_actuator_strength_multipliers(
        self,
        env: Any,
        num_reset: int,
    ) -> np.ndarray | None:
        strength_cfg = self._validated_actuator_strength_config(env)
        if strength_cfg is None:
            return None
        expected = int(env._num_action)
        sampling_mode = str(getattr(strength_cfg, "sampling_mode", "fixed"))
        if sampling_mode == "fixed":
            fixed = np.asarray(strength_cfg.multipliers, dtype=np.float64)
            return np.broadcast_to(fixed, (num_reset, expected)).copy()

        sampled = np.ones((num_reset, expected), dtype=np.float64)
        anomaly_rows = np.flatnonzero(
            np.random.uniform(size=(num_reset,)) >= float(strength_cfg.nominal_probability)
        )
        if anomaly_rows.size == 0:
            return sampled
        candidates = np.asarray(strength_cfg.candidate_actuator_indices, dtype=np.int64)
        selected = np.random.choice(candidates, size=anomaly_rows.size, replace=True)
        low, high = np.asarray(strength_cfg.multiplier_range, dtype=np.float64).tolist()
        sampled[anomaly_rows, selected] = np.random.uniform(low, high, size=anomaly_rows.size)
        return sampled

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
        low, high = env.cfg.domain_rand.dof_position_bias_range
        if env.cfg.domain_rand.randomize_dof_position_bias:
            dof_bias = np.random.uniform(low, high, size=(num_reset, env._num_action))
        else:
            dof_bias = np.zeros((num_reset, env._num_action))
        control_delay = (
            np.random.randint(0, 2, size=(num_reset, 1))
            if env.cfg.domain_rand.randomize_control_delay
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
            stand_phase = gait_cfg.get("stand_phase", [math.pi, math.pi])
        else:
            enabled = bool(getattr(gait_cfg, "enabled", False))
            freeze = bool(getattr(gait_cfg, "freeze_phase_in_stand_mode", False))
            stand_phase = getattr(gait_cfg, "stand_phase", [math.pi, math.pi])
        if not (enabled and freeze):
            return
        stand_phase_arr = np.asarray(stand_phase, dtype=get_global_dtype())
        if stand_phase_arr.shape != (2,):
            raise ValueError(f"gait_constraint.stand_phase must have shape (2,), got {stand_phase}")
        standing = np.asarray(gait_enabled <= 0.5, dtype=bool)
        if np.any(standing):
            gait_phase[standing, :] = stand_phase_arr[None, :]

    def _sample_commands(self, env: Any, num_reset: int) -> np.ndarray:
        return sample_g1_walk_commands(env, num_reset)

    def _sample_gait_phase(self, env: Any, num_reset: int) -> np.ndarray:
        mode = env.cfg.gait_phase_init_mode
        if mode == "independent":
            left = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            right = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
            return np.asarray(np.column_stack([left, right]), dtype=get_global_dtype())

        phase = np.random.uniform(0.0, 2.0 * np.pi, size=(num_reset,))
        return np.asarray(np.column_stack([phase, phase + np.pi]), dtype=get_global_dtype())

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
        return env._compute_obs(info_updates, linvel, gyro, gravity, dof_pos, dof_vel)  # type: ignore[no-any-return]


class G1WalkEnv(G1BaseEnv):
    _cfg: G1WalkEnvCfg
    _reward_cfg: Any

    def __init__(self, cfg: G1WalkEnvCfg, num_envs=1, backend_type="mujoco"):
        if cfg.reward_config is None:
            raise ValueError("reward_config must be provided via Hydra configuration")
        progress_cfg = cfg.forward_progress_termination
        if progress_cfg.grace_steps <= 0:
            raise ValueError("forward_progress_termination.grace_steps must be positive")
        if progress_cfg.min_command_forward_speed < 0.0:
            raise ValueError(
                "forward_progress_termination.min_command_forward_speed must be non-negative"
            )
        if progress_cfg.min_average_forward_speed < 0.0:
            raise ValueError(
                "forward_progress_termination.min_average_forward_speed must be non-negative"
            )
        if cfg.reward_config.straight_line_lateral_tolerance_m <= 0.0:
            raise ValueError("straight_line_lateral_tolerance_m must be positive")
        if cfg.reward_config.straight_line_yaw_tolerance_rad <= 0.0:
            raise ValueError("straight_line_yaw_tolerance_rad must be positive")
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.asset.base_name,
            push_body_name=cfg.domain_rand.push_body_name,
            motrix_max_iterations=cfg.motrix_max_iterations,
            post_step_forward_sensor=cfg.post_step_forward_sensor,
        )
        super().__init__(cfg, backend, num_envs)
        self._enable_reward_log = True
        self._reward_cfg = cfg.reward_config

        self._gait_phase_delta = float(
            2.0 * math.pi * self._reward_cfg.gait_frequency * cfg.ctrl_dt
        )
        self._pose_weights = np.array(self._reward_cfg.pose_weights, dtype=get_global_dtype())
        if self._pose_weights.shape[0] != self._num_action:
            raise ValueError("pose_weights length mismatch")
        self._upper_body_pose_weights = build_upper_body_pose_weights(self._reward_cfg.pose_weights)
        self._episode_tracker: EpisodeLengthTracker | None = None
        self._penalty_curriculum: PenaltyCurriculum | None = None
        if cfg.curriculum.enabled:
            self._episode_tracker = EpisodeLengthTracker(num_envs)
            self._penalty_curriculum = PenaltyCurriculum(
                self,
                enabled=True,
                initial_scale=cfg.curriculum.initial_scale,
                min_scale=cfg.curriculum.min_scale,
                max_scale=cfg.curriculum.max_scale,
                level_down_threshold=cfg.curriculum.level_down_threshold,
                level_up_threshold=cfg.curriculum.level_up_threshold,
                degree=cfg.curriculum.degree,
            )

        self._init_reward_functions()
        self._fada_base_kp: np.ndarray | None = None
        self._fada_base_kd: np.ndarray | None = None
        self._fada_base_body_mass: np.ndarray | None = None
        self._fada_ground_geom_id: int | None = None
        self._fada_base_geom_friction: np.ndarray | None = None
        self._fada_tau_max: np.ndarray | None = None
        self._fada_body_names: tuple[str, ...] = ()
        self._fada_checkpoint_layout_identity: (
            G1FADAPrivilegedCheckpointLayoutIdentity | None
        ) = None
        strength_enabled = bool(
            getattr(getattr(cfg.domain_rand, "actuator_strength", None), "enabled", False)
        )
        privileged_enabled = bool(cfg.fada_privileged_observation.enabled)
        if (
            cfg.domain_rand.randomize_kp
            or cfg.domain_rand.randomize_kd
            or strength_enabled
            or privileged_enabled
        ):
            base_kp, base_kd = backend.get_actuator_gains()
            self._fada_base_kp = np.asarray(base_kp, dtype=np.float64)
            self._fada_base_kd = np.asarray(base_kd, dtype=np.float64)
        else:
            base_kp = base_kd = None
        if privileged_enabled:
            if backend.backend_type != "mujoco":
                raise ValueError("g1_fada_privileged_v1 currently requires the MuJoCo backend")
            self._fada_base_body_mass = backend.get_body_mass()
            self._fada_base_geom_friction = backend.get_geom_friction()
            self._fada_ground_geom_id = backend.get_geom_id("floor")
            self._fada_tau_max = np.max(np.abs(backend.get_actuator_force_range()), axis=1)
            self._fada_body_names = backend.get_body_names()
            actuated_joint_names = backend.get_actuated_joint_names()
            if len(actuated_joint_names) != self._num_action:
                raise ValueError(
                    "FADA actuated joint order must contain exactly one name per action"
                )
            self._fada_checkpoint_layout_identity = build_g1_fada_checkpoint_layout_identity(
                body_names=self._fada_body_names,
                actuated_joint_names=actuated_joint_names,
                model_file=cfg.scene.model_file,
            )
        dr_provider = G1WalkDomainRandomizationProvider(
            base_kp=base_kp,
            base_kd=base_kd,
            base_body_mass=self._fada_base_body_mass,
            base_geom_friction=self._fada_base_geom_friction,
            ground_geom_id=self._fada_ground_geom_id,
        )
        self._init_domain_randomization(dr_provider)

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        # gyro(3) + gravity(3) + diff(29) + dof_vel(29) + action(29) + cmd(3)
        # + phase(2) [+ mode(1)] = 98/99. The ordinary critic adds linvel(3);
        # the FADA bundle already owns base linear velocity and must not duplicate it.
        mode_dim = 1 if self._cfg.mode_observation else 0
        height_dim = 1 if self._uses_height_command_observation() else 0
        privileged_strength_dim = 29 if self._includes_privileged_actuator_strength_obs() else 0
        fada_privileged_dim = (
            174 + len(self._fada_body_names) if self._fada_privileged_enabled() else 0
        )
        critic_base_dim = 98 if self._fada_privileged_enabled() else 101
        return {
            "obs": 98 + mode_dim + height_dim,
            "critic": critic_base_dim
            + mode_dim
            + height_dim
            + privileged_strength_dim
            + fada_privileged_dim,
        }

    def get_fada_privileged_checkpoint_identity(
        self,
    ) -> G1FADAPrivilegedCheckpointLayoutIdentity:
        """Return the immutable checkpoint layout sealed during environment initialization."""

        identity = self._fada_checkpoint_layout_identity
        if identity is None:
            raise ValueError("FADA privileged checkpoint identity is unavailable")
        return identity

    def _init_reward_functions(self):
        self._reward_fns: dict[str, Any] = {
            "tracking_lin_vel": rewards.tracking_lin_vel,
            "tracking_ang_vel": rewards.tracking_ang_vel,
            "forward_progress": rewards.forward_progress,
            "under_speed": rewards.under_speed,
            "lin_vel_z": rewards.lin_vel_z,
            "orientation": rewards.orientation,
            "penalty_orientation": rewards.orientation,
            "upright": rewards.upright,
            "ang_vel_xy": rewards.ang_vel_xy,
            "penalty_ang_vel_xy": rewards.ang_vel_xy,
            "action_rate": rewards.action_rate,
            "penalty_action_rate": rewards.action_rate,
            "penalty_lateral_displacement": self._reward_lateral_displacement,
            "penalty_yaw_drift": self._reward_yaw_drift,
            "penalty_lateral_corridor_violation": self._reward_lateral_corridor_violation,
            "penalty_yaw_corridor_violation": self._reward_yaw_corridor_violation,
            "base_height": rewards.base_height,
            "track_base_height_exp_smooth": rewards.track_base_height_exp_smooth,
            "pose": rewards.weighted_pose,
            "upper_body_pose": self._reward_upper_body_pose,
            "penalty_close_feet_xy": self._reward_close_feet_xy,
            "penalty_feet_ori": self._reward_feet_ori,
            "stand_still": self._reward_stand_still,
            "stand_action_l2": self._reward_stand_action_l2,
            "stand_dof_vel_l2": self._reward_stand_dof_vel_l2,
            "stand_lin_vel_xy_l2": self._reward_stand_lin_vel_xy_l2,
            "stand_yaw_vel_l2": self._reward_stand_yaw_vel_l2,
            "stand_tilt_l2": self._reward_stand_tilt_l2,
            "stand_tilt_margin_l2": self._reward_stand_tilt_margin_l2,
            "stand_fall_l2": self._reward_stand_fall_l2,
            "stand_base_height_deficit_l1": self._reward_stand_base_height_deficit_l1,
            "stand_support_height_margin_l2": self._reward_stand_support_height_margin_l2,
            "stand_both_feet_contact": self._reward_stand_both_feet_contact,
            "stand_foot_contact_balance": self._reward_stand_foot_contact_balance,
            "stand_feet_slide_l2": self._reward_stand_feet_slide_l2,
            "stand_feet_x_l2": self._reward_stand_feet_x_l2,
            "stand_feet_y_width_l2": self._reward_stand_feet_y_width_l2,
            "stand_feet_yaw_l2": self._reward_stand_feet_yaw_l2,
            "stand_base_feet_center_x_l2": self._reward_stand_base_feet_center_x_l2,
            "stand_base_feet_center_y_l2": self._reward_stand_base_feet_center_y_l2,
            "feet_phase": self._reward_feet_phase,
            "feet_phase_contrast": self._reward_feet_phase_contrast,
            "feet_phase_contact": self._reward_feet_phase_contact,
            "feet_double_stance": self._reward_feet_double_stance,
            "feet_air_time": self._reward_feet_air_time,
            "alive": rewards.alive,
        }

    def _episode_frame_errors(self, info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        try:
            initial_position = np.asarray(info["episode_start_base_pos"], dtype=get_global_dtype())
            initial_yaw = np.asarray(info["episode_start_base_yaw"], dtype=get_global_dtype())
        except KeyError as exc:
            raise ValueError(
                "trajectory-precision rewards require reset-time episode frame state"
            ) from exc
        current_position = np.asarray(self.get_base_pos(), dtype=get_global_dtype())
        current_yaw = np.asarray(np_yaw_from_quat(self.get_base_quat()), dtype=get_global_dtype())
        expected_position_shape = (self._num_envs, 3)
        if initial_position.shape != expected_position_shape:
            raise ValueError(
                "episode_start_base_pos must have shape "
                f"{expected_position_shape}, got {initial_position.shape}"
            )
        if initial_yaw.shape != (self._num_envs,):
            raise ValueError(
                "episode_start_base_yaw must have shape "
                f"({self._num_envs},), got {initial_yaw.shape}"
            )
        delta = current_position[:, :2] - initial_position[:, :2]
        lateral = -np.sin(initial_yaw) * delta[:, 0] + np.cos(initial_yaw) * delta[:, 1]
        yaw_drift = np_wrap_to_pi(current_yaw - initial_yaw)
        return lateral, yaw_drift

    def _reward_lateral_displacement(self, ctx: RewardContext) -> np.ndarray:
        lateral, _ = self._episode_frame_errors(ctx.info)
        return np.asarray(np.square(lateral), dtype=get_global_dtype())

    def _reward_yaw_drift(self, ctx: RewardContext) -> np.ndarray:
        _, yaw_drift = self._episode_frame_errors(ctx.info)
        return np.asarray(np.square(yaw_drift), dtype=get_global_dtype())

    @staticmethod
    def _normalized_corridor_violation(error: np.ndarray, tolerance: float) -> np.ndarray:
        excess = np.maximum(np.abs(error) - float(tolerance), 0.0)
        return np.asarray(np.square(excess / float(tolerance)), dtype=get_global_dtype())

    def _reward_lateral_corridor_violation(self, ctx: RewardContext) -> np.ndarray:
        lateral, _ = self._episode_frame_errors(ctx.info)
        return self._normalized_corridor_violation(
            lateral, self._reward_cfg.straight_line_lateral_tolerance_m
        )

    def _reward_yaw_corridor_violation(self, ctx: RewardContext) -> np.ndarray:
        _, yaw_drift = self._episode_frame_errors(ctx.info)
        return self._normalized_corridor_violation(
            yaw_drift, self._reward_cfg.straight_line_yaw_tolerance_rad
        )

    def _forward_progress_failure(self, info: dict[str, Any]) -> np.ndarray:
        cfg = self._cfg.forward_progress_termination
        if not cfg.enabled:
            return np.zeros((self._num_envs,), dtype=np.bool_)
        try:
            initial_position = info["episode_start_base_pos"]
            initial_yaw = info["episode_start_base_yaw"]
            steps = info["steps"]
            commands = info["commands"]
        except KeyError as error:
            raise ValueError(
                "forward-progress termination requires episode frame, steps, and commands"
            ) from error
        failure, average_speed = compute_forward_progress_failure(
            self.get_base_pos(),
            initial_position,
            initial_yaw,
            steps,
            commands,
            ctrl_dt=float(self._cfg.ctrl_dt),
            grace_steps=int(cfg.grace_steps),
            min_command_forward_speed=float(cfg.min_command_forward_speed),
            min_average_forward_speed=float(cfg.min_average_forward_speed),
        )
        info["forward_progress_average_speed"] = average_speed
        info["forward_progress_failure"] = failure
        return failure

    def _terrain_relative_base_height(self) -> np.ndarray:
        return np.asarray(self._backend.get_base_pos()[:, 2], dtype=get_global_dtype())

    def _debug_action_trace_enabled(self) -> bool:
        return _debug_env_flag("UNILAB_G1_ACTION_TRACE")

    def _debug_action_trace_step(self, info: dict) -> int:
        steps = info.get("steps")
        if steps is not None:
            steps_arr = np.asarray(steps)
            if steps_arr.size > 0:
                return int(steps_arr.reshape(-1)[0])
        step = int(info.get("_g1_action_trace_step", 0))
        info["_g1_action_trace_step"] = step + 1
        return step

    def _debug_virtual_pd_torque(
        self, ctrl: np.ndarray | None, dof_pos: np.ndarray, dof_vel: np.ndarray
    ) -> np.ndarray | None:
        if ctrl is None:
            return None
        try:
            kp, kd = self._backend.get_actuator_gains()
        except (AttributeError, NotImplementedError, RuntimeError):
            return None
        kp_arr = np.asarray(kp, dtype=get_global_dtype())
        kd_arr = np.asarray(kd, dtype=get_global_dtype())
        if kp_arr.shape[0] != ctrl.shape[1] or kd_arr.shape[0] != ctrl.shape[1]:
            return None
        return np.asarray(kp_arr[None, :] * (ctrl - dof_pos) - kd_arr[None, :] * dof_vel)

    def _debug_action_trace(
        self,
        info: dict,
        *,
        reward: np.ndarray,
        terminated: np.ndarray,
        linvel: np.ndarray,
        gyro: np.ndarray,
        gravity: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
    ) -> None:
        if not self._debug_action_trace_enabled():
            return
        step = self._debug_action_trace_step(info)
        interval = _debug_env_int("UNILAB_G1_ACTION_TRACE_INTERVAL", default=20)
        if step % interval != 0:
            return

        current_actions = np.asarray(
            info.get("current_actions", np.zeros((self._num_envs, self._num_action))),
            dtype=get_global_dtype(),
        )
        executed_actions = np.asarray(
            info.get("executed_actions", np.zeros_like(current_actions)),
            dtype=get_global_dtype(),
        )
        ctrl = info.get("_g1_action_trace_ctrl")
        ctrl_arr = None if ctrl is None else np.asarray(ctrl, dtype=get_global_dtype())
        default = np.broadcast_to(self.default_angles, current_actions.shape).astype(
            get_global_dtype(),
            copy=False,
        )
        ctrl_delta = None if ctrl_arr is None else ctrl_arr - default
        ctrl_error = None if ctrl_arr is None else ctrl_arr - dof_pos
        virtual_pd_tau = self._debug_virtual_pd_torque(ctrl_arr, dof_pos, dof_vel)
        torques = info.get("torques")

        commands = np.asarray(info.get("commands", np.zeros((self._num_envs, 3))))
        gait_enabled = self._gait_enabled_mask(info)
        dynamic_mode = self._dynamic_mode_mask(info)
        base_height = self._terrain_relative_base_height()
        base_height_target = float(self._reward_cfg.base_height_target)
        base_height_deficit = np.maximum(base_height_target - base_height, 0.0)
        tilt_deg = np.rad2deg(np.arccos(np.clip(gravity[:, 2], -1.0, 1.0)))
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        left_count, right_count = self._foot_contact_counts()
        base_feet_delta = self._base_delta_from_feet_center_in_base_yaw_frame()

        print("[G1ActionTrace] begin")
        print(
            "[G1ActionTrace] "
            f"step={step} task={type(self._cfg).__name__} "
            f"action_scale={float(self._cfg.control_config.action_scale):.6g} "
            f"stand_action_authority={bool(self._cfg.stand_action_authority)} "
            f"mode_observation={bool(self._cfg.mode_observation)} "
            f"reward_mean={float(np.mean(reward)):.6g} "
            f"terminated_frac={float(np.mean(terminated.astype(get_global_dtype()))):.6g}"
        )
        print("[G1ActionTrace] " + _debug_stats("commands", commands))
        print("[G1ActionTrace] " + _debug_head("commands", commands, count=3))
        print("[G1ActionTrace] " + _debug_stats("gait_enabled", gait_enabled))
        print("[G1ActionTrace] " + _debug_stats("dynamic_mode", dynamic_mode))
        print("[G1ActionTrace] " + _debug_stats("current_actions", current_actions))
        print("[G1ActionTrace] " + _debug_head("current_actions", current_actions))
        print("[G1ActionTrace] " + _debug_stats("executed_actions", executed_actions))
        print("[G1ActionTrace] " + _debug_head("executed_actions", executed_actions))
        print(
            "[G1ActionTrace] "
            + _debug_stats("executed_minus_current", executed_actions - current_actions)
        )
        if ctrl_arr is not None:
            print("[G1ActionTrace] " + _debug_stats("ctrl", ctrl_arr))
            print("[G1ActionTrace] " + _debug_head("ctrl", ctrl_arr))
            print("[G1ActionTrace] " + _debug_stats("ctrl_minus_default", ctrl_delta))
            print("[G1ActionTrace] " + _debug_stats("ctrl_minus_dof_pos", ctrl_error))
        print("[G1ActionTrace] " + _debug_stats("dof_pos_minus_default", dof_pos - default))
        print("[G1ActionTrace] " + _debug_stats("dof_vel", dof_vel))
        if virtual_pd_tau is not None:
            print("[G1ActionTrace] " + _debug_stats("virtual_pd_tau", virtual_pd_tau))
            print("[G1ActionTrace] " + _debug_head("virtual_pd_tau", virtual_pd_tau))
        if torques is not None:
            print("[G1ActionTrace] " + _debug_stats("info_torques", torques))
        print("[G1ActionTrace] " + _debug_stats("linvel", linvel))
        print("[G1ActionTrace] " + _debug_stats("gyro", gyro))
        print(f"[G1ActionTrace] base_height_target={base_height_target:.6g}")
        print("[G1ActionTrace] " + _debug_stats("base_height", base_height))
        print("[G1ActionTrace] " + _debug_stats("base_height_deficit", base_height_deficit))
        print("[G1ActionTrace] " + _debug_stats("tilt_deg", tilt_deg))
        print("[G1ActionTrace] " + _debug_stats("left_contact", left_contact.astype(float)))
        print("[G1ActionTrace] " + _debug_stats("right_contact", right_contact.astype(float)))
        print("[G1ActionTrace] " + _debug_stats("left_contact_count", left_count))
        print("[G1ActionTrace] " + _debug_stats("right_contact_count", right_count))
        print("[G1ActionTrace] " + _debug_stats("base_minus_feet_center_xy", base_feet_delta))
        reward_log = info.get("log", {})
        if isinstance(reward_log, dict) and reward_log:
            keys = sorted(k for k in reward_log if k.startswith("reward/"))
            for key in keys:
                print(f"[G1ActionTrace] {key}={float(reward_log[key]):.6g}")
        print("[G1ActionTrace] end")

    def update_state(self, state: NpEnvState) -> NpEnvState:
        self._update_commands(state.info)
        linvel = self.get_local_linvel()
        gyro = self.get_gyro()
        gravity = self._backend.get_sensor_data(self._cfg.sensor.upvector)
        dof_pos = self.get_dof_pos()
        dof_vel = self.get_dof_vel()

        max_tilt_rad = np.deg2rad(self._reward_cfg.max_tilt_deg)
        tilt = np.arccos(np.clip(gravity[:, 2], -1, 1))
        terminated = np.logical_or(
            tilt > max_tilt_rad,
            self._terrain_relative_base_height() < self._reward_cfg.min_base_height,
        )
        np.logical_or(terminated, self._forward_progress_failure(state.info), out=terminated)

        reward = self._compute_reward(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        self._debug_action_trace(
            state.info,
            reward=reward,
            terminated=terminated,
            linvel=linvel,
            gyro=gyro,
            gravity=gravity,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
        )
        obs = self._compute_obs(state.info, linvel, gyro, gravity, dof_pos, dof_vel)
        state = state.replace(obs=obs, reward=reward, terminated=terminated)

        done = state.terminated | state.truncated
        if self._episode_tracker is None or self._penalty_curriculum is None or not np.any(done):
            return state

        done_indices = np.where(done)[0]
        episode_lengths = state.info["steps"][done_indices] + 1
        self._episode_tracker.update(episode_lengths)
        self._penalty_curriculum.update(self._episode_tracker.average_length)

        if "log" not in state.info:
            state.info["log"] = {}
        state.info["log"]["curriculum/average_episode_length"] = float(
            self._episode_tracker.average_length
        )
        state.info["log"]["curriculum/penalty_scale"] = float(
            self._penalty_curriculum.current_scale
        )
        return state

    def _capture_task_rollout_state(self) -> dict[str, Any]:
        """Capture G1 curriculum state that may change on a shadow termination."""

        return {
            "episode_average_length": (
                None
                if self._episode_tracker is None
                else float(self._episode_tracker.average_length)
            ),
            "penalty_scale": (
                None
                if self._penalty_curriculum is None
                else float(self._penalty_curriculum.current_scale)
            ),
            "reward_scales": copy.deepcopy(self._reward_cfg.scales),
        }

    def _restore_task_rollout_state(self, snapshot: Any) -> None:
        """Restore G1 curriculum and its derived reward-scale mutation."""

        if not isinstance(snapshot, dict) or set(snapshot) != {
            "episode_average_length",
            "penalty_scale",
            "reward_scales",
        }:
            raise ValueError("invalid G1 task rollout snapshot")
        if self._episode_tracker is not None:
            average = snapshot["episode_average_length"]
            if average is None:
                raise ValueError("G1 rollout snapshot is missing episode tracker state")
            self._episode_tracker.average_length = float(average)
        if self._penalty_curriculum is not None:
            scale = snapshot["penalty_scale"]
            if scale is None:
                raise ValueError("G1 rollout snapshot is missing penalty curriculum state")
            self._penalty_curriculum.current_scale = float(scale)
        self._reward_cfg.scales.clear()
        self._reward_cfg.scales.update(copy.deepcopy(snapshot["reward_scales"]))

    def _compute_obs(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        command = info["commands"]
        command_obs = self._command_observation(info, command.shape[0])
        last_actions = info.get("current_actions", np.zeros_like(diff))
        gait_phase = self._gait_phase_for_observation(info)
        mode_obs = self._mode_observation(info)
        walk_profile = self._uses_walk_observation_profile()

        noisy_gyro = self._obs_noise(gyro, noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(gravity, noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(dof_vel, noise_cfg.scale_joint_vel)
        actor_gyro_scale = 0.25 if walk_profile else 1.0
        actor_dof_vel_scale = 0.05 if walk_profile else 1.0

        actor_parts = [
            noisy_gyro * actor_gyro_scale,
            -noisy_gravity,
            noisy_diff,
            noisy_dof_vel * actor_dof_vel_scale,
            last_actions,
            command_obs,
            gait_phase,
        ]
        if self._cfg.mode_observation:
            actor_parts.append(mode_obs)
        actor = np.concatenate(actor_parts, axis=1, dtype=get_global_dtype())

        critic_gyro_scale = 0.25 if walk_profile else 1.0
        critic_dof_vel_scale = 0.05 if walk_profile else 1.0
        critic_linvel_scale = 2.0 if walk_profile else 1.0
        critic_parts = [
            gyro * critic_gyro_scale,
            -gravity,
            diff,
            dof_vel * critic_dof_vel_scale,
            last_actions,
            command_obs,
            gait_phase,
        ]
        if self._cfg.mode_observation:
            critic_parts.append(mode_obs)
        critic_base = np.concatenate(critic_parts, axis=1, dtype=get_global_dtype())
        if self._fada_privileged_enabled():
            critic = critic_base
        else:
            critic = np.concatenate(
                [
                    critic_base,
                    np.asarray(linvel * critic_linvel_scale, dtype=get_global_dtype()),
                ],
                axis=1,
                dtype=get_global_dtype(),
            )
        if self._includes_privileged_actuator_strength_obs():
            strength = np.asarray(
                info.get("privileged_actuator_strength"),
                dtype=get_global_dtype(),
            )
            expected_shape = (actor.shape[0], self._num_action)
            if strength.shape != expected_shape:
                raise ValueError(
                    "critic actuator-strength observation requires "
                    f"info['privileged_actuator_strength'] shape {expected_shape}, "
                    f"got {strength.shape}"
                )
            if not np.isfinite(strength).all():
                raise ValueError("critic actuator-strength observation must be finite")
            critic = np.concatenate([critic, strength], axis=1, dtype=get_global_dtype())

        if self._fada_privileged_enabled():
            critic = np.concatenate(
                [critic, self._materialize_fada_privileged_observation(info, linvel)],
                axis=1,
                dtype=get_global_dtype(),
            )

        return {"obs": actor, "critic": critic}

    def _fada_privileged_enabled(self) -> bool:
        return bool(
            getattr(getattr(self._cfg, "fada_privileged_observation", None), "enabled", False)
        )

    def _materialize_fada_privileged_observation(
        self, info: dict, linvel: np.ndarray
    ) -> np.ndarray:
        rows = int(np.asarray(linvel).shape[0])
        if self._fada_tau_max is None:
            raise ValueError("FADA privileged observation requires cached actuator force limits")
        return pack_fada_runtime_observation(
            body_names=self._fada_body_names,
            tau_max=self._fada_tau_max,
            linvel=linvel,
            left_contact_sensor=self._backend.get_sensor_data("left_foot_net_contact"),
            right_contact_sensor=self._backend.get_sensor_data("right_foot_net_contact"),
            root_clearance=self._terrain_relative_base_height(),
            torques=np.asarray(
                info.get("torques", np.zeros((rows, self._num_action))),
                dtype=get_global_dtype(),
            ),
            info=info,
            dtype=np.dtype(get_global_dtype()),
        )

    def _includes_privileged_actuator_strength_obs(self) -> bool:
        domain_rand = getattr(self._cfg, "domain_rand", None)
        strength_cfg = getattr(domain_rand, "actuator_strength", None)
        return bool(
            strength_cfg is not None
            and getattr(strength_cfg, "enabled", False)
            and getattr(strength_cfg, "include_in_critic_obs", False)
        )

    def _uses_height_command_observation(self) -> bool:
        command_cfg = getattr(self._cfg, "commands", None)
        if isinstance(command_cfg, dict):
            return bool(command_cfg.get("observe_height_command", False))
        return bool(getattr(command_cfg, "observe_height_command", False))

    def _command_observation(self, info: dict, num_obs: int) -> np.ndarray:
        command = np.asarray(info["commands"], dtype=get_global_dtype())
        if not self._uses_height_command_observation():
            return command
        height = self._height_command_column(info, num_obs)
        return np.concatenate([command, height], axis=1, dtype=get_global_dtype())

    def _height_command_column(self, info: dict, num_obs: int) -> np.ndarray:
        target = info.get("height_commands")
        if target is None:
            command_cfg = self._cfg.commands
            if isinstance(command_cfg, dict):
                default_target = command_cfg.get("default_height")
            else:
                default_target = getattr(command_cfg, "default_height", None)
            if default_target is None:
                default_target = getattr(
                    getattr(self, "_reward_cfg", None), "base_height_target", 0.0
                )
            target = info.get("commands_height", default_target)

        target_arr = np.asarray(target, dtype=get_global_dtype())
        if target_arr.ndim == 0:
            return np.full((num_obs, 1), float(target_arr), dtype=get_global_dtype())
        if target_arr.ndim == 1:
            target_arr = target_arr.reshape(-1, 1)
        if target_arr.shape != (num_obs, 1):
            raise ValueError(
                f"height command must have shape ({num_obs}, 1), got {target_arr.shape}"
            )
        return np.asarray(target_arr, dtype=get_global_dtype())

    def _gait_constraint_cfg(self) -> GaitConstraintConfig:
        cfg = getattr(self._reward_cfg, "gait_constraint", GaitConstraintConfig())
        if isinstance(cfg, dict):
            cfg = GaitConstraintConfig(**cfg)
            self._reward_cfg.gait_constraint = cfg
        return cast(GaitConstraintConfig, cfg)

    def _reward_mode_cfg(self) -> RewardModeConfig:
        cfg = getattr(self._reward_cfg, "mode", RewardModeConfig())
        if isinstance(cfg, dict):
            cfg = RewardModeConfig(**cfg)
            self._reward_cfg.mode = cfg
        return cfg

    def _command_active_mask(self, info: dict) -> np.ndarray:
        commands = info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        cfg = self._gait_constraint_cfg()
        return compute_command_active_mask(
            commands,
            xy_threshold=cfg.command_xy_threshold,
            yaw_threshold=cfg.command_yaw_threshold,
        )

    def _current_command_gait_mask(self, info: dict) -> np.ndarray | None:
        commands = info.get("commands")
        if commands is None:
            return None

        commands_arr = np.asarray(commands, dtype=get_global_dtype())
        if commands_arr.ndim == 1:
            commands_arr = commands_arr[None, :]
        if commands_arr.ndim != 2:
            raise ValueError(f"commands must have shape (N, C), got {commands_arr.shape}")
        cfg = self._gait_constraint_cfg()
        return compute_command_active_mask(
            commands_arr,
            xy_threshold=cfg.command_xy_threshold,
            yaw_threshold=cfg.command_yaw_threshold,
        )

    def _gait_enabled_mask(self, info: dict) -> np.ndarray:
        command_mask = self._current_command_gait_mask(info)
        if command_mask is not None:
            info["gait_enabled"] = command_mask
            return command_mask

        if "gait_enabled" not in info:
            return self._command_active_mask(info)

        mask = np.asarray(info["gait_enabled"], dtype=get_global_dtype())
        if mask.ndim == 2 and mask.shape[1] == 1:
            mask = mask[:, 0]
        commands = info.get("commands")
        expected_size = np.asarray(commands).shape[0] if commands is not None else self._num_envs
        if mask.shape != (expected_size,):
            raise ValueError(f"gait_enabled must have shape ({expected_size},), got {mask.shape}")
        return np.asarray(mask > 0.5, dtype=get_global_dtype())

    def _stand_recovery_mask(self, ctx: RewardContext, stand_mask: np.ndarray) -> np.ndarray:
        speed_xy = np.linalg.norm(ctx.linvel[:, :2], axis=1)
        if ctx.gravity is None:
            tilt_deg = np.zeros_like(speed_xy)
        else:
            tilt_deg = np.rad2deg(np.arccos(np.clip(ctx.gravity[:, 2], -1.0, 1.0)))
        recovery = (speed_xy > float(self._reward_cfg.stand_recovery_lin_vel_xy_threshold)) | (
            tilt_deg > float(self._reward_cfg.stand_recovery_tilt_deg_threshold)
        )
        recovery_mask = np.asarray(recovery, dtype=get_global_dtype()) * stand_mask
        ctx.info["stand_recovery_active"] = recovery_mask
        return recovery_mask

    def _dynamic_mode_mask(self, info: dict) -> np.ndarray:
        gait_enabled = self._gait_enabled_mask(info)
        recovery = np.asarray(
            info.get("stand_recovery_active", np.zeros_like(gait_enabled)),
            dtype=get_global_dtype(),
        )
        if recovery.ndim == 2 and recovery.shape[1] == 1:
            recovery = recovery[:, 0]
        if recovery.shape != gait_enabled.shape:
            recovery = np.zeros_like(gait_enabled)
        return np.asarray(np.maximum(gait_enabled, recovery), dtype=get_global_dtype())

    def _stand_phase_array(self) -> np.ndarray:
        cfg = self._gait_constraint_cfg()
        stand_phase = np.asarray(cfg.stand_phase, dtype=get_global_dtype())
        if stand_phase.shape != (2,):
            raise ValueError(f"gait_constraint.stand_phase must have shape (2,), got {stand_phase}")
        return stand_phase

    def _actions_for_execution(self, actions: np.ndarray, info: dict) -> np.ndarray:
        active = self._dynamic_mode_mask(info).astype(bool)
        if not self._cfg.stand_action_authority:
            self._log_action_authority(info, actions, actions, active)
            return actions
        if np.all(active):
            self._log_action_authority(info, actions, actions, active)
            return actions
        exec_actions = np.array(actions, copy=True)
        exec_actions[~active] = 0.0
        self._log_action_authority(info, actions, exec_actions, active)
        return exec_actions

    def _log_action_authority(
        self,
        info: dict,
        raw_actions: np.ndarray,
        exec_actions: np.ndarray,
        active: np.ndarray,
    ) -> None:
        if not self._enable_reward_log:
            return
        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        if int(step_count[0]) % 4 != 0:
            return
        log = info.get("log", {})
        stand = ~active.astype(bool)
        log["reward/action_authority_stand_frac"] = float(np.mean(stand))
        log["reward/raw_action_l1"] = float(np.mean(np.sum(np.abs(raw_actions), axis=1)))
        log["reward/executed_action_l1"] = float(np.mean(np.sum(np.abs(exec_actions), axis=1)))
        if np.any(stand):
            log["reward/stand_raw_action_l1"] = float(
                np.mean(np.sum(np.abs(raw_actions[stand]), axis=1))
            )
            log["reward/stand_executed_action_l1"] = float(
                np.mean(np.sum(np.abs(exec_actions[stand]), axis=1))
            )
        else:
            log["reward/stand_raw_action_l1"] = 0.0
            log["reward/stand_executed_action_l1"] = 0.0
        info["log"] = log

    def _log_current_action_authority(self, info: dict) -> None:
        raw_actions = info.get("current_actions")
        exec_actions = info.get("executed_actions")
        if raw_actions is None or exec_actions is None:
            return
        active = self._dynamic_mode_mask(info).astype(bool)
        self._log_action_authority(info, raw_actions, exec_actions, active)

    def _gait_phase_for_observation(self, info: dict) -> np.ndarray:
        gait_phase = np.asarray(
            info.get("gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())),
            dtype=get_global_dtype(),
        )
        cfg = self._gait_constraint_cfg()
        if not (cfg.enabled and cfg.freeze_phase_in_stand_mode):
            return gait_phase
        active = self._dynamic_mode_mask(info).astype(bool)
        stand_phase = self._stand_phase_array()
        return np.asarray(
            np.where(active[:, None], gait_phase, stand_phase[None, :]), dtype=get_global_dtype()
        )

    def _mode_observation(self, info: dict) -> np.ndarray:
        dynamic_mode = self._dynamic_mode_mask(info)
        return np.asarray(dynamic_mode[:, None], dtype=get_global_dtype())

    def _uses_walk_observation_profile(self) -> bool:
        scales = getattr(getattr(self, "_reward_cfg", None), "scales", None)
        if scales is None:
            reward_cfg = getattr(self._cfg, "reward_config", None)
            scales = getattr(reward_cfg, "scales", None)

        if scales is not None:
            if any(
                key in scales
                for key in (
                    "penalty_orientation",
                    "penalty_ang_vel_xy",
                    "penalty_action_rate",
                    "alive",
                )
            ):
                return True
            if any(key in scales for key in ("orientation", "ang_vel_xy", "action_rate")):
                return False

        curriculum = getattr(self._cfg, "curriculum", None)
        return bool(curriculum is not None and curriculum.enabled)

    def _actor_symmetry_obs_layout(self) -> SymmetryObsLayout:
        command_dim = 4 if self._uses_height_command_observation() else 3
        layout = [
            ("gyro", 3),
            ("gravity", 3),
            ("dof_pos", self._num_action),
            ("dof_vel", self._num_action),
            ("actions", self._num_action),
            ("command", command_dim),
            ("gait_phase", 2),
        ]
        if self._cfg.mode_observation:
            layout.append(("mode", 1))
        return tuple(layout)

    def get_symmetry_obs_layouts(self) -> dict[str, SymmetryObsLayout]:
        actor_layout = self._actor_symmetry_obs_layout()
        return {
            "obs": actor_layout,
            "critic": (*actor_layout, ("linvel", 3)),
        }

    def build_symmetry_augmentation(self, *, device: str):
        if self._backend.backend_type != "mujoco":
            return None
        from unilab.envs.locomotion.g1.symmetry import G1SymmetryAugmentation

        return G1SymmetryAugmentation(
            self._backend.model,
            self.get_symmetry_obs_layouts(),
            device=device,
        )

    def _build_reward_context(
        self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel
    ) -> RewardContext:
        height_target: float | np.ndarray | None = info.get("height_commands")
        if height_target is None:
            height_target = info.get("commands_height", self._reward_cfg.base_height_target)
        height_target_arr = np.asarray(height_target, dtype=get_global_dtype())
        if height_target_arr.ndim == 0:
            height_target = float(height_target_arr)
        elif height_target_arr.ndim == 2 and height_target_arr.shape[1] == 1:
            height_target = height_target_arr[:, 0]
        else:
            height_target = height_target_arr
        return RewardContext(
            info=info,
            linvel=linvel,
            gyro=gyro,
            dof_pos=dof_pos,
            num_envs=self._num_envs,
            default_angles=self.default_angles,
            tracking_sigma=self._reward_cfg.tracking_sigma,
            base_height_target=height_target,
            base_height=self._terrain_relative_base_height(),
            gravity=gravity,
            dof_vel=dof_vel,
            pose_weights=self._pose_weights,
        )

    def _compute_reward(self, info: dict, linvel, gyro, gravity, dof_pos, dof_vel) -> np.ndarray:
        cfg = self._reward_cfg
        ctx = self._build_reward_context(info, linvel, gyro, gravity, dof_pos, dof_vel)
        reward = self._compute_mode_reward(ctx, cfg)
        reward = self._apply_gait_constraint_bridge(ctx, reward)
        self._log_current_action_authority(info)
        return reward

    def _compute_mode_reward(self, ctx: RewardContext, cfg: G1RewardConfig) -> np.ndarray:
        mode_cfg = self._reward_mode_cfg()
        if not mode_cfg.enabled:
            return rewards.run_reward_dispatch(
                scales=cfg.scales,
                fns=self._reward_fns,
                ctx=ctx,
                info=ctx.info,
                enable_log=self._enable_reward_log,
                ctrl_dt=self._cfg.ctrl_dt,
            )

        walk_mask = self._gait_enabled_mask(ctx.info)
        stand_mask = np.asarray(1.0 - walk_mask, dtype=get_global_dtype())
        stand_recovery_mask = self._stand_recovery_mask(ctx, stand_mask)
        stand_static_mask = np.asarray(stand_mask - stand_recovery_mask, dtype=get_global_dtype())
        self._reset_mode_reward_log(ctx.info)
        walk_terms = self._combine_mode_terms(mode_cfg.balance_common_terms, mode_cfg.walk_terms)
        if mode_cfg.standing_enabled:
            stand_terms = self._combine_mode_terms(
                mode_cfg.balance_common_terms, mode_cfg.stand_terms
            )
            stand_recovery_terms = self._combine_mode_terms(
                mode_cfg.balance_common_terms, mode_cfg.stand_recovery_terms
            )
            stand_reward = self._run_masked_mode_reward_dispatch(
                ctx, cfg, stand_terms, stand_static_mask, mode_cfg.stand_scale_overrides
            )
            stand_recovery_reward = self._run_masked_mode_reward_dispatch(
                ctx,
                cfg,
                stand_recovery_terms,
                stand_recovery_mask,
                mode_cfg.stand_recovery_scale_overrides,
            )
        else:
            stand_reward = np.zeros((ctx.num_envs,), dtype=get_global_dtype())
            stand_recovery_reward = np.zeros((ctx.num_envs,), dtype=get_global_dtype())
        walk_reward = self._run_masked_mode_reward_dispatch(
            ctx, cfg, walk_terms, walk_mask, mode_cfg.walk_scale_overrides
        )
        reward = np.asarray(
            stand_reward + stand_recovery_reward + walk_reward, dtype=get_global_dtype()
        )
        self._log_reward_mode(
            ctx.info,
            stand_static_mask,
            stand_recovery_mask,
            walk_mask,
            stand_reward,
            stand_recovery_reward,
            walk_reward,
        )
        return reward

    @staticmethod
    def _combine_mode_terms(common_terms: list[str], mode_terms: list[str]) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for name in [*common_terms, *mode_terms]:
            if name in seen:
                continue
            seen.add(name)
            terms.append(name)
        return terms

    def _reset_mode_reward_log(self, info: dict) -> None:
        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        if self._enable_reward_log and int(step_count[0]) % 4 == 0:
            info["log"] = {}

    def _run_masked_mode_reward_dispatch(
        self,
        ctx: RewardContext,
        cfg: G1RewardConfig,
        terms: list[str],
        mask: np.ndarray,
        scale_overrides: dict[str, float] | None = None,
    ) -> np.ndarray:
        dtype = get_global_dtype()
        term_set = set(terms)
        scales = {name: scale for name, scale in cfg.scales.items() if name in term_set}
        if scale_overrides:
            scales.update(
                {name: scale for name, scale in scale_overrides.items() if name in term_set}
            )
        reward = np.zeros((ctx.num_envs,), dtype=dtype)
        step_count = ctx.info.get("steps", np.zeros((ctx.num_envs,), dtype=np.uint32))
        should_log = self._enable_reward_log and int(step_count[0]) % 4 == 0
        log = ctx.info.get("log", {})
        mode_mask = np.asarray(mask, dtype=dtype)

        for name, scale in scales.items():
            if scale == 0 or name not in self._reward_fns:
                continue
            rew = self._reward_fns[name](ctx)
            weighted_rew = np.asarray(rew * scale * mode_mask, dtype=dtype)
            reward += weighted_rew
            if should_log:
                key = f"reward/{name}"
                log[key] = float(log.get(key, 0.0) + np.mean(weighted_rew))

        ctx.info["log"] = log
        return reward * self._cfg.ctrl_dt

    def _log_reward_mode(
        self,
        info: dict,
        stand_static_mask: np.ndarray,
        stand_recovery_mask: np.ndarray,
        walk_mask: np.ndarray,
        stand_reward: np.ndarray,
        stand_recovery_reward: np.ndarray,
        walk_reward: np.ndarray,
    ) -> None:
        if not self._enable_reward_log:
            return
        log = info.get("log", {})
        log["mode/stand_reward_mask"] = float(np.mean(stand_static_mask + stand_recovery_mask))
        log["mode/stand_static_reward_mask"] = float(np.mean(stand_static_mask))
        log["mode/stand_recovery_reward_mask"] = float(np.mean(stand_recovery_mask))
        log["mode/walk_reward_mask"] = float(np.mean(walk_mask))
        log["reward/mode_stand_frac"] = float(np.mean(stand_static_mask + stand_recovery_mask))
        log["reward/mode_stand_static_frac"] = float(np.mean(stand_static_mask))
        log["reward/mode_stand_recovery_frac"] = float(np.mean(stand_recovery_mask))
        log["reward/mode_walk_frac"] = float(np.mean(walk_mask))
        log["reward/stand_total"] = float(np.mean(stand_static_mask * stand_reward))
        log["reward/stand_recovery_total"] = float(
            np.mean(stand_recovery_mask * stand_recovery_reward)
        )
        log["reward/walk_total"] = float(np.mean(walk_mask * walk_reward))
        info["log"] = log

    def _apply_gait_constraint_bridge(self, ctx: RewardContext, reward: np.ndarray) -> np.ndarray:
        cfg = self._gait_constraint_cfg()
        if not cfg.enabled:
            return reward

        components = self._compute_gait_constraint_components(ctx, cfg)
        excess = np.maximum(components["total"] - cfg.epsilon, 0.0)
        gated_cost = excess * components["gate"]
        self._log_gait_constraint(ctx.info, components, gated_cost)
        return np.asarray(
            reward - cfg.penalty_scale * gated_cost * self._cfg.ctrl_dt,
            dtype=get_global_dtype(),
        )

    def _compute_gait_constraint_components(
        self, ctx: RewardContext, cfg: GaitConstraintConfig
    ) -> dict[str, np.ndarray]:
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        height = compute_gait_phase_height_violation(
            left_foot[:, 2], right_foot[:, 2], gait_phase, swing_height
        )
        contrast = compute_gait_phase_contrast_violation(
            left_foot[:, 2], right_foot[:, 2], gait_phase, swing_height
        )
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        contact = compute_gait_phase_contact_violation(
            left_contact, right_contact, gait_phase, swing_height
        )
        total = np.asarray(
            cfg.height_weight * height
            + cfg.contrast_weight * contrast
            + cfg.contact_weight * contact,
            dtype=get_global_dtype(),
        )

        commands = ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        command_active = self._command_active_mask(ctx.info)
        gait_enabled = self._gait_enabled_mask(ctx.info)
        gate = (
            np.ones_like(gait_enabled, dtype=get_global_dtype())
            if cfg.apply_in_stand_mode
            else gait_enabled
        )
        if cfg.apply_when_tracking:
            gate = gate * compute_tracking_gate(
                commands,
                ctx.linvel,
                ctx.gyro,
                tracking_sigma=self._reward_cfg.tracking_sigma,
                threshold=cfg.tracking_threshold,
            )

        return {
            "height": height,
            "contrast": contrast,
            "contact": contact,
            "total": total,
            "gate": np.asarray(gate, dtype=get_global_dtype()),
            "command_active": command_active,
            "gait_enabled": gait_enabled,
        }

    def _log_gait_constraint(
        self, info: dict, components: dict[str, np.ndarray], gated_cost: np.ndarray
    ) -> None:
        step_count = info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32))
        if not self._enable_reward_log or int(step_count[0]) % 4 != 0:
            return
        log = info.get("log", {})
        log["constraint/gait_height"] = float(np.mean(components["height"]))
        log["constraint/gait_contrast"] = float(np.mean(components["contrast"]))
        log["constraint/gait_contact"] = float(np.mean(components["contact"]))
        log["constraint/gait_total"] = float(np.mean(components["total"]))
        log["constraint/gait_gated_cost"] = float(np.mean(gated_cost))
        log["mode/command_active"] = float(np.mean(components["command_active"]))
        log["mode/gait_enabled"] = float(np.mean(components["gait_enabled"]))
        log["mode/gait_constraint_gate"] = float(np.mean(components["gate"]))
        info["log"] = log

    def _reward_feet_phase(self, ctx: RewardContext):
        """Reward gait phase tracking by encouraging the expected swing-foot height."""
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        stance_z = np.minimum(left_foot[:, 2], right_foot[:, 2])
        left_height = left_foot[:, 2] - stance_z
        right_height = right_foot[:, 2] - stance_z
        left_error = np.square(left_height - left_target)
        right_error = np.square(right_height - right_target)
        reward = np.exp(-(left_error + right_error) / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _gait_reward_gate(self, linvel: np.ndarray) -> np.ndarray:
        min_forward_speed = getattr(self._reward_cfg, "min_forward_speed_for_gait_reward", 0.0)
        return compute_forward_speed_gate(linvel, min_forward_speed)

    def _reward_feet_phase_contrast(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target, right_target = compute_feet_phase_height_targets(gait_phase, swing_height)
        stance_z = np.minimum(left_foot[:, 2], right_foot[:, 2])
        actual_delta = (left_foot[:, 2] - stance_z) - (right_foot[:, 2] - stance_z)
        target_delta = left_target - right_target
        error = np.square(actual_delta - target_delta)
        reward = np.exp(-error / self._reward_cfg.feet_phase_tracking_sigma)
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_phase_contact(self, ctx: RewardContext):
        gait_phase = ctx.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        swing_height = self._reward_cfg.feet_phase_swing_height
        left_target_contact, right_target_contact = compute_feet_phase_contact_targets(
            gait_phase, swing_height
        )
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        left_match = np.asarray(left_contact == left_target_contact, dtype=get_global_dtype())
        right_match = np.asarray(right_contact == right_target_contact, dtype=get_global_dtype())
        reward = np.asarray(0.5 * (left_match + right_match), dtype=get_global_dtype())
        return np.asarray(reward * self._gait_reward_gate(ctx.linvel), dtype=get_global_dtype())

    def _reward_feet_double_stance(self, ctx: RewardContext):
        commands = ctx.info.get("commands", np.zeros((self._num_envs, 3), dtype=get_global_dtype()))
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        double_stance = np.asarray(
            np.logical_and(left_contact, right_contact), dtype=get_global_dtype()
        )
        return np.asarray(
            double_stance * compute_forward_command_mask(commands), dtype=get_global_dtype()
        )

    def _reward_feet_ori(self, ctx: RewardContext):
        left_foot_quat = self._backend.get_sensor_data("left_foot_quat")
        right_foot_quat = self._backend.get_sensor_data("right_foot_quat")
        return (
            np.square(left_foot_quat[:, 1])
            + np.square(left_foot_quat[:, 2])
            + np.square(right_foot_quat[:, 1])
            + np.square(right_foot_quat[:, 2])
        )

    def _reward_close_feet_xy(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        feet_dist = np.linalg.norm(left_foot[:, :2] - right_foot[:, :2], axis=1)
        return np.where(
            feet_dist < self._reward_cfg.close_feet_threshold,
            np.square(feet_dist - self._reward_cfg.close_feet_threshold),
            0.0,
        )

    def _reward_feet_air_time(self, ctx: RewardContext):
        air_time = ctx.info.get(
            "feet_air_time", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        in_range = (air_time > 0.05) & (air_time < 0.5)
        return np.sum(in_range.astype(float), axis=1)

    def _stand_mode_mask(self, ctx: RewardContext) -> np.ndarray:
        return np.asarray(1.0 - self._gait_enabled_mask(ctx.info), dtype=get_global_dtype())

    def _reward_stand_still(self, ctx: RewardContext):
        diff = ctx.dof_pos - self.default_angles
        return np.asarray(
            np.sum(np.abs(diff), axis=1) * self._stand_mode_mask(ctx), dtype=get_global_dtype()
        )

    def _reward_stand_action_l2(self, ctx: RewardContext):
        actions = ctx.info.get("current_actions", np.zeros_like(ctx.dof_pos))
        return np.asarray(
            np.sum(np.square(actions), axis=1) * self._stand_mode_mask(ctx),
            dtype=get_global_dtype(),
        )

    def _reward_stand_dof_vel_l2(self, ctx: RewardContext):
        assert ctx.dof_vel is not None
        return np.asarray(
            np.sum(np.square(ctx.dof_vel), axis=1) * self._stand_mode_mask(ctx),
            dtype=get_global_dtype(),
        )

    def _reward_stand_lin_vel_xy_l2(self, ctx: RewardContext):
        return np.asarray(
            np.sum(np.square(ctx.linvel[:, :2]), axis=1) * self._stand_mode_mask(ctx),
            dtype=get_global_dtype(),
        )

    def _reward_stand_yaw_vel_l2(self, ctx: RewardContext):
        return np.asarray(
            np.square(ctx.gyro[:, 2]) * self._stand_mode_mask(ctx),
            dtype=get_global_dtype(),
        )

    def _reward_stand_tilt_l2(self, ctx: RewardContext):
        if ctx.gravity is None:
            return np.zeros((ctx.num_envs,), dtype=get_global_dtype())
        return np.asarray(
            np.sum(np.square(ctx.gravity[:, :2]), axis=1) * self._stand_mode_mask(ctx),
            dtype=get_global_dtype(),
        )

    def _reward_stand_tilt_margin_l2(self, ctx: RewardContext):
        if ctx.gravity is None:
            return np.zeros((ctx.num_envs,), dtype=get_global_dtype())
        tilt = np.arccos(np.clip(ctx.gravity[:, 2], -1.0, 1.0))
        soft_limit = np.deg2rad(float(self._reward_cfg.stand_recovery_tilt_deg_threshold))
        hard_limit = np.deg2rad(float(self._reward_cfg.max_tilt_deg))
        span = max(float(hard_limit - soft_limit), 1.0e-6)
        margin = np.maximum((tilt - soft_limit) / span, 0.0)
        return np.asarray(np.square(margin) * self._stand_mode_mask(ctx), dtype=get_global_dtype())

    def _reward_stand_fall_l2(self, ctx: RewardContext):
        if ctx.gravity is None or ctx.base_height is None:
            return np.zeros((ctx.num_envs,), dtype=get_global_dtype())
        tilt = np.arccos(np.clip(ctx.gravity[:, 2], -1.0, 1.0))
        fallen = (tilt > np.deg2rad(float(self._reward_cfg.max_tilt_deg))) | (
            ctx.base_height < float(self._reward_cfg.min_base_height)
        )
        return np.asarray(
            fallen.astype(get_global_dtype()) * self._stand_mode_mask(ctx), dtype=get_global_dtype()
        )

    def _foot_contact_counts(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            compute_aggregated_foot_contact_count(self._backend, LEFT_FOOT_CONTACT_SENSORS),
            compute_aggregated_foot_contact_count(self._backend, RIGHT_FOOT_CONTACT_SENSORS),
        )

    def _stand_support_relative_height(self) -> np.ndarray:
        dtype = get_global_dtype()
        base_z = np.asarray(self._backend.get_base_pos()[:, 2], dtype=dtype)
        left_foot = np.asarray(self._backend.get_sensor_data("left_foot_pos"), dtype=dtype)
        right_foot = np.asarray(self._backend.get_sensor_data("right_foot_pos"), dtype=dtype)
        left_contact = compute_aggregated_foot_contact(
            self._backend, LEFT_FOOT_CONTACT_SENSORS
        ).astype(dtype)
        right_contact = compute_aggregated_foot_contact(
            self._backend, RIGHT_FOOT_CONTACT_SENSORS
        ).astype(dtype)
        contact_count = left_contact + right_contact
        eps = float(self._reward_cfg.stand_foot_contact_balance_epsilon)
        contacted_foot_z = np.where(
            contact_count > eps,
            (left_foot[:, 2] * left_contact + right_foot[:, 2] * right_contact)
            / np.maximum(contact_count, eps),
            0.5 * (left_foot[:, 2] + right_foot[:, 2]),
        )
        return np.asarray(base_z - contacted_foot_z, dtype=dtype)

    def _reward_stand_support_height_margin_l2(self, ctx: RewardContext):
        target = self._stand_height_target(ctx)
        margin = float(self._reward_cfg.stand_support_height_margin)
        support_height = self._stand_support_relative_height()
        low_deficit = np.maximum(target - margin - support_height, 0.0)
        return np.asarray(
            np.square(low_deficit) * self._stand_mode_mask(ctx),
            dtype=get_global_dtype(),
        )

    def _reward_stand_base_height_deficit_l1(self, ctx: RewardContext):
        target = self._stand_height_target(ctx)
        margin = float(self._reward_cfg.stand_support_height_margin)
        if ctx.base_height is None:
            base_height = self._terrain_relative_base_height()
        else:
            base_height = np.asarray(ctx.base_height, dtype=get_global_dtype())
        low_deficit = np.maximum(target - margin - base_height, 0.0)
        return np.asarray(low_deficit * self._stand_mode_mask(ctx), dtype=get_global_dtype())

    def _stand_height_target(self, ctx: RewardContext) -> np.ndarray:
        """Resolve the per-environment standing target with scalar legacy fallback."""
        target = np.asarray(ctx.base_height_target, dtype=get_global_dtype())
        if target.ndim == 0:
            return np.full((ctx.num_envs,), float(target), dtype=get_global_dtype())
        if target.ndim == 2 and target.shape == (ctx.num_envs, 1):
            target = target[:, 0]
        if target.shape != (ctx.num_envs,):
            raise ValueError(
                "standing height target must be scalar or have shape "
                f"({ctx.num_envs},), got {target.shape}"
            )
        return np.asarray(target, dtype=get_global_dtype())

    def _reward_stand_both_feet_contact(self, ctx: RewardContext):
        left_contact = compute_aggregated_foot_contact(self._backend, LEFT_FOOT_CONTACT_SENSORS)
        right_contact = compute_aggregated_foot_contact(self._backend, RIGHT_FOOT_CONTACT_SENSORS)
        missing = (
            2.0 - left_contact.astype(get_global_dtype()) - right_contact.astype(get_global_dtype())
        )
        return np.asarray(missing * self._stand_mode_mask(ctx), dtype=get_global_dtype())

    def _reward_stand_foot_contact_balance(self, ctx: RewardContext):
        left_count, right_count = self._foot_contact_counts()
        total = left_count + right_count
        eps = float(self._reward_cfg.stand_foot_contact_balance_epsilon)
        imbalance = np.where(
            total > eps,
            np.abs(left_count - right_count) / np.maximum(total, eps),
            1.0,
        )
        return np.asarray(imbalance * self._stand_mode_mask(ctx), dtype=get_global_dtype())

    def _reward_stand_feet_slide_l2(self, ctx: RewardContext):
        left_vel = self._backend.get_sensor_data("left_foot_linvel")
        right_vel = self._backend.get_sensor_data("right_foot_linvel")
        left_contact = compute_aggregated_foot_contact(
            self._backend, LEFT_FOOT_CONTACT_SENSORS
        ).astype(get_global_dtype())
        right_contact = compute_aggregated_foot_contact(
            self._backend, RIGHT_FOOT_CONTACT_SENSORS
        ).astype(get_global_dtype())
        slide = (
            np.sum(np.square(left_vel[:, :2]), axis=1) * left_contact
            + np.sum(np.square(right_vel[:, :2]), axis=1) * right_contact
        )
        return np.asarray(slide * self._stand_mode_mask(ctx), dtype=get_global_dtype())

    def _reward_stand_feet_x_l2(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        delta = self._feet_delta_in_base_yaw_frame(left_foot, right_foot)
        x_delta = delta[:, 0]
        target = float(self._reward_cfg.stand_feet_x_target)
        return np.asarray(
            np.square(x_delta - target) * self._stand_mode_mask(ctx), dtype=get_global_dtype()
        )

    def _reward_stand_feet_y_width_l2(self, ctx: RewardContext):
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        delta = self._feet_delta_in_base_yaw_frame(left_foot, right_foot)
        width = np.abs(delta[:, 1])
        target = float(self._reward_cfg.stand_feet_y_width_target)
        return np.asarray(
            np.square(width - target) * self._stand_mode_mask(ctx), dtype=get_global_dtype()
        )

    def _reward_stand_feet_yaw_l2(self, ctx: RewardContext):
        left_foot_quat = self._backend.get_sensor_data("left_foot_quat")
        right_foot_quat = self._backend.get_sensor_data("right_foot_quat")
        base_yaw = np_yaw_from_quat(self._backend.get_base_quat())
        left_yaw = np_wrap_to_pi(np_yaw_from_quat(left_foot_quat) - base_yaw)
        right_yaw = np_wrap_to_pi(np_yaw_from_quat(right_foot_quat) - base_yaw)
        relative_yaw = np_wrap_to_pi(left_yaw - right_yaw)
        return np.asarray(
            np.square(relative_yaw) * self._stand_mode_mask(ctx), dtype=get_global_dtype()
        )

    def _reward_stand_base_feet_center_x_l2(self, ctx: RewardContext):
        delta = self._base_delta_from_feet_center_in_base_yaw_frame()
        target = float(self._reward_cfg.stand_base_feet_center_x_target)
        return np.asarray(
            np.square(delta[:, 0] - target) * self._stand_mode_mask(ctx),
            dtype=get_global_dtype(),
        )

    def _reward_stand_base_feet_center_y_l2(self, ctx: RewardContext):
        delta = self._base_delta_from_feet_center_in_base_yaw_frame()
        target = float(self._reward_cfg.stand_base_feet_center_y_target)
        return np.asarray(
            np.square(delta[:, 1] - target) * self._stand_mode_mask(ctx),
            dtype=get_global_dtype(),
        )

    def _feet_delta_in_base_yaw_frame(
        self, left_foot: np.ndarray, right_foot: np.ndarray
    ) -> np.ndarray:
        delta = np.asarray(left_foot[:, :2] - right_foot[:, :2], dtype=get_global_dtype())
        return self._rotate_xy_to_base_yaw_frame(delta)

    def _base_delta_from_feet_center_in_base_yaw_frame(self) -> np.ndarray:
        left_foot = self._backend.get_sensor_data("left_foot_pos")
        right_foot = self._backend.get_sensor_data("right_foot_pos")
        base_pos = self._backend.get_base_pos()
        feet_center = 0.5 * (left_foot[:, :2] + right_foot[:, :2])
        delta = np.asarray(base_pos[:, :2] - feet_center, dtype=get_global_dtype())
        return self._rotate_xy_to_base_yaw_frame(delta)

    def _rotate_xy_to_base_yaw_frame(self, delta: np.ndarray) -> np.ndarray:
        base_yaw = np_yaw_from_quat(self._backend.get_base_quat())
        cos_yaw = np.cos(base_yaw)
        sin_yaw = np.sin(base_yaw)
        return np.stack(
            [
                cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1],
                -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1],
            ],
            axis=1,
        )

    def _reward_upper_body_pose(self, ctx: RewardContext):
        diff = ctx.dof_pos - self.default_angles
        return np.asarray(
            np.sum(self._upper_body_pose_weights * np.square(diff), axis=1),
            dtype=get_global_dtype(),
        )

    def _update_commands(self, info: dict) -> None:
        commands = info.get("commands")
        if commands is None:
            return

        commands_arr = np.asarray(commands, dtype=get_global_dtype())
        resampling_time = float(getattr(self._cfg.commands, "resampling_time", 0.0))
        if resampling_time > 0.0:
            interval_steps = max(int(round(resampling_time / self._cfg.ctrl_dt)), 1)
            steps = np.asarray(info.get("steps", np.zeros((self._num_envs,), dtype=np.uint32)))
            resample_mask = (steps > 0) & ((steps % interval_steps) == 0)
            if np.any(resample_mask):
                num_resample = int(np.count_nonzero(resample_mask))
                commands_arr[resample_mask] = sample_g1_walk_commands(self, num_resample)
                if getattr(self._cfg.commands, "heading_command", False):
                    heading_commands = self._ensure_heading_commands(info, commands_arr.shape[0])
                    heading_commands[resample_mask] = sample_heading_commands(self, num_resample)
                    info["heading_commands"] = heading_commands

        if getattr(self._cfg.commands, "heading_command", False):
            heading_commands = self._ensure_heading_commands(info, commands_arr.shape[0])
            base_quat = np.asarray(self._backend.get_base_quat(), dtype=get_global_dtype())
            if base_quat.shape[0] == commands_arr.shape[0]:
                stiffness = float(getattr(self._cfg.commands, "heading_control_stiffness", 0.5))
                apply_heading_yaw_feedback(
                    commands_arr, base_quat, heading_commands, stiffness=stiffness
                )

        info["commands"] = commands_arr
        gait_enabled = self._gait_enabled_mask(info)
        cfg = self._gait_constraint_cfg()
        if cfg.enabled and cfg.freeze_phase_in_stand_mode:
            gait_phase = info.get(
                "gait_phase", np.zeros((commands_arr.shape[0], 2), dtype=get_global_dtype())
            )
            gait_phase = np.asarray(gait_phase, dtype=get_global_dtype())
            inactive = gait_enabled <= 0.5
            if np.any(inactive):
                gait_phase[inactive, :] = self._stand_phase_array()
                info["gait_phase"] = gait_phase

    def _ensure_heading_commands(self, info: dict, num_obs: int) -> np.ndarray:
        heading_commands = info.get("heading_commands")
        if heading_commands is None or np.asarray(heading_commands).shape != (num_obs,):
            heading_commands = sample_heading_commands(self, num_obs)
        heading_commands = np.asarray(heading_commands, dtype=get_global_dtype())
        info["heading_commands"] = heading_commands
        return heading_commands

    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        state.info["last_actions"] = state.info.get("current_actions", np.zeros_like(actions))
        state.info["current_actions"] = actions

        gait_phase = state.info.get(
            "gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())
        )
        cfg = self._gait_constraint_cfg()
        if cfg.enabled and cfg.freeze_phase_in_stand_mode:
            active = self._dynamic_mode_mask(state.info).astype(bool)
            gait_phase[active, 0] = (gait_phase[active, 0] + self._gait_phase_delta) % (2 * np.pi)
            gait_phase[active, 1] = (gait_phase[active, 1] + self._gait_phase_delta) % (2 * np.pi)
            gait_phase[~active, :] = self._stand_phase_array()
        else:
            gait_phase[:, 0] = (gait_phase[:, 0] + self._gait_phase_delta) % (2 * np.pi)
            gait_phase[:, 1] = (gait_phase[:, 1] + self._gait_phase_delta) % (2 * np.pi)
        state.info["gait_phase"] = gait_phase

        authority_actions = self._actions_for_execution(actions, state.info)
        state.info["authority_actions"] = authority_actions
        exec_actions = apply_action_execution_fault(
            authority_actions,
            getattr(self._cfg, "action_execution_fault", None),
            num_envs=self._num_envs,
        )
        previous_exec_actions = np.asarray(
            state.info.get("executed_actions", exec_actions), dtype=get_global_dtype()
        )
        if self._fada_privileged_enabled():
            delay = np.asarray(state.info["fada_control_delay"]).reshape(self._num_envs, 1)
            exec_actions = np.where(delay > 0.5, previous_exec_actions, exec_actions)
        state.info["executed_actions"] = exec_actions
        ctrl: np.ndarray = (
            exec_actions * self._cfg.control_config.action_scale + self.default_angles
        )
        if self._fada_privileged_enabled():
            if self._fada_tau_max is None or self._fada_base_kp is None:
                raise ValueError("FADA PD perturbation baselines were not initialized")
            rfi_fraction = float(self._cfg.domain_rand.torque_rfi_fraction)
            torque_rfi = np.random.uniform(
                -rfi_fraction * self._fada_tau_max,
                rfi_fraction * self._fada_tau_max,
                size=(self._num_envs, self._num_action),
            )
            state.info["fada_torque_rfi"] = np.asarray(torque_rfi, dtype=get_global_dtype())
            kp = np.asarray(state.info["fada_kp_scale"]) * self._fada_base_kp[None, :]
            ctrl = apply_fada_pd_target_perturbation(
                ctrl,
                dof_position_bias=np.asarray(state.info["fada_dof_position_bias"]),
                torque_rfi=torque_rfi,
                kp=kp,
                tau_max=self._fada_tau_max,
            )
        if self._debug_action_trace_enabled():
            state.info["_g1_action_trace_ctrl"] = ctrl
        return ctrl


def _walk_curriculum() -> CurriculumConfig:
    return CurriculumConfig(
        enabled=True,
        initial_scale=0.5,
        min_scale=0.5,
        max_scale=1.0,
        level_down_threshold=150.0,
        level_up_threshold=750.0,
        degree=0.001,
    )


@dataclass
class G1WalkControlConfig:
    action_scale: float = 1.0
    simulate_action_latency: bool = False


@dataclass
class G1WalkRewardConfig(G1RewardConfig):
    """Align reward weights with holosoma G1 walking."""


@registry.envcfg("G1WalkFlat")
@dataclass
class G1WalkFlatCfg(G1WalkEnvCfg):
    reward_config: G1WalkRewardConfig | None = None
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml")
        )
    )
    control_config: G1WalkControlConfig = field(default_factory=G1WalkControlConfig)  # type: ignore[assignment]
    curriculum: CurriculumConfig = field(default_factory=_walk_curriculum)


@registry.envcfg("G1WalkHeight")
@dataclass
class G1WalkHeightCfg(G1WalkFlatCfg):
    pass


@registry.envcfg("G1StandStill")
@dataclass
class G1StandStillCfg(G1WalkFlatCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_flat.xml"),
            fragment_files=[str(ASSETS_ROOT_PATH / "robots" / "g1" / "stand_support_task.xml")],
        )
    )


@registry.envcfg("G1StandHeight")
@dataclass
class G1StandHeightCfg(G1StandStillCfg):
    pass


@registry.envcfg("G1WalkRough")
@dataclass
class G1WalkRoughCfg(G1WalkFlatCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "g1" / "scene_rough.xml")
        )
    )


registry.register_env("G1WalkFlat", G1WalkEnv, sim_backend="mujoco")
registry.register_env("G1WalkFlat", G1WalkEnv, sim_backend="motrix")
registry.register_env("G1WalkHeight", G1WalkEnv, sim_backend="mujoco")
registry.register_env("G1StandStill", G1WalkEnv, sim_backend="mujoco")
registry.register_env("G1StandHeight", G1WalkEnv, sim_backend="mujoco")
registry.register_env("G1WalkRough", G1WalkEnv, sim_backend="mujoco")
registry.register_env("G1WalkRough", G1WalkEnv, sim_backend="motrix")
