"""G1 joystick locomotion environments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.scene import SceneCfg
from unilab.envs.locomotion.common.commands import (
    Commands,
)
from unilab.envs.locomotion.common.domain_rand import DomainRandConfig
from unilab.envs.locomotion.g1.base import G1BaseCfg
from unilab.envs.locomotion.g1.calibration_fault import (
    G1ActionExecutionFaultConfig,
)
from unilab.envs.locomotion.g1.fada_privileged import (
    DOF_POSITION_BIAS_LIMIT_RAD,
    TORQUE_RFI_FRACTION,
    G1FADAPrivilegedObservationConfig,
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
    curriculum_enabled: bool = False
    curriculum_multiplier_lows: list[float] = field(default_factory=list)
    curriculum_nominal_probabilities: list[float] = field(default_factory=list)
    curriculum_promote_threshold: float = 800.0
    curriculum_demote_threshold: float = 500.0
    curriculum_update_episodes: int = 1024
    group_curriculum_enabled: bool = False
    group_curriculum_scales: list[float] = field(default_factory=list)
    curriculum_progress_mode: str = "episode_quality"
    curriculum_iteration_boundaries: list[int] = field(default_factory=list)
    curriculum_max_termination_rate: float = 0.1
    curriculum_brake_cooldown_steps: int = 100
    curriculum_recovery_hold_steps: int = 500


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
    gait_phase_enabled: bool = True
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
