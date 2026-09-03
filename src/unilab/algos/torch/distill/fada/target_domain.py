from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin
from numbers import Integral
from typing import Any, Literal, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf


def _finite_float(raw: Any, name: str) -> float:
    value = float(raw)
    if not isfinite(value):
        raise ValueError(f"FADA target {name} must be finite")
    return value


def _positive_float(raw: Any, name: str) -> float:
    value = _finite_float(raw, name)
    if value <= 0.0:
        raise ValueError(f"FADA target {name} must be positive")
    return value


def validate_fada_slope_commands(raw: Any) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("FADA target command_sequence must be non-empty")
    commands: list[tuple[float, float, float]] = []
    for command in raw:
        if not isinstance(command, (list, tuple)) or len(command) != 3:
            raise ValueError("FADA target commands must be 3-D")
        values = tuple(_finite_float(value, "command") for value in command)
        parsed = (values[0], values[1], values[2])
        if parsed[1] != 0.0:
            raise ValueError("FADA slope command must have zero lateral velocity")
        if parsed[2] != 0.0:
            raise ValueError("FADA slope command must have zero yaw velocity")
        commands.append(parsed)
    return tuple(commands)


def _sample_fada_slope_commands(raw: Any) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(raw, dict):
        raise ValueError("FADA target command_sampling must be a mapping")
    expected = {"forward_speed_range", "num_trials", "seed"}
    if set(raw) != expected:
        raise ValueError(
            "FADA target command_sampling fields must be exactly "
            f"{sorted(expected)}, got {sorted(raw)}"
        )
    speed_range = raw["forward_speed_range"]
    if not isinstance(speed_range, (list, tuple)) or len(speed_range) != 2:
        raise ValueError("FADA target command_sampling forward_speed_range must have two values")
    lower = _positive_float(speed_range[0], "command_sampling forward speed range")
    upper = _positive_float(speed_range[1], "command_sampling forward speed range")
    if lower >= upper:
        raise ValueError("FADA target command_sampling forward speed range must increase")
    num_trials = raw["num_trials"]
    if isinstance(num_trials, bool) or not isinstance(num_trials, Integral) or num_trials < 2:
        raise ValueError("FADA target command_sampling num_trials must be an integer >= 2")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("FADA target command_sampling seed must be a non-negative integer")

    count = int(num_trials)
    centers = lower + (np.arange(count, dtype=np.float64) + 0.5) * ((upper - lower) / count)
    order = np.random.default_rng(int(seed)).permutation(count)
    return tuple((float(centers[index]), 0.0, 0.0) for index in order)


def _resolve_slope_commands(raw: dict[str, Any]) -> tuple[tuple[float, float, float], ...]:
    has_sequence = "command_sequence" in raw
    has_sampling = "command_sampling" in raw
    if has_sequence == has_sampling:
        raise ValueError(
            "FADA slope target must define exactly one of command_sequence or command_sampling"
        )
    if has_sequence:
        return validate_fada_slope_commands(raw["command_sequence"])
    return _sample_fada_slope_commands(raw["command_sampling"])


@dataclass(frozen=True)
class FADASlopeGeometry:
    angle_deg: float
    width_m: float
    approach_length_m: float
    surface_length_m: float
    entry_margin_m: float
    finish_margin_m: float

    def __post_init__(self) -> None:
        if not 0.0 < self.angle_deg < 90.0:
            raise ValueError("FADA slope angle_deg must be in (0, 90)")
        for name in ("width_m", "approach_length_m", "surface_length_m"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"FADA slope {name} must be positive")
        if not 0.0 <= self.entry_margin_m < self.surface_length_m:
            raise ValueError("FADA slope entry_margin_m is outside the surface")
        if not 0.0 <= self.finish_margin_m < self.surface_length_m:
            raise ValueError("FADA slope finish_margin_m is outside the surface")

    def surface_coordinates(self, positions_w: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions_w, dtype=np.float64)
        if positions.shape[-1] != 3 or not np.all(np.isfinite(positions)):
            raise ValueError("FADA slope positions must be finite (..., 3) arrays")
        angle = radians(self.angle_deg)
        relative_x = positions[..., 0] - self.approach_length_m
        surface = cos(angle) * relative_x + sin(angle) * positions[..., 2]
        return np.stack((surface, positions[..., 1]), axis=-1)

    def has_entered(self, base_pos_w: np.ndarray, feet_pos_w: np.ndarray) -> bool:
        base = self.surface_coordinates(np.asarray(base_pos_w))
        feet = self.surface_coordinates(np.asarray(feet_pos_w))
        if base.shape != (2,) or feet.shape != (2, 2):
            raise ValueError("FADA slope entry requires one base and two foot positions")
        return bool(base[0] >= self.entry_margin_m and np.all(feet[:, 0] >= 0.0))

    def has_finished(self, base_pos_w: np.ndarray) -> bool:
        base = self.surface_coordinates(np.asarray(base_pos_w))
        if base.shape != (2,):
            raise ValueError("FADA slope finish requires one base position")
        return bool(base[0] >= self.surface_length_m - self.finish_margin_m)

    def foot_exited(self, feet_pos_w: np.ndarray) -> bool:
        feet = self.surface_coordinates(np.asarray(feet_pos_w))
        if feet.shape != (2, 2):
            raise ValueError("FADA slope exit requires two foot positions")
        return bool(np.any(np.abs(feet[:, 1]) > 0.5 * self.width_m))


FADA_SLOPE_15_GEOMETRY = FADASlopeGeometry(
    angle_deg=15.0,
    width_m=0.8,
    approach_length_m=1.5,
    surface_length_m=8.0,
    entry_margin_m=0.25,
    finish_margin_m=0.5,
)

FADA_SLOPE_10_GEOMETRY = FADASlopeGeometry(
    angle_deg=10.0,
    width_m=0.8,
    approach_length_m=1.5,
    surface_length_m=8.0,
    entry_margin_m=0.25,
    finish_margin_m=0.5,
)

FADA_SLOPE_GEOMETRY_BY_SCENE = {
    "scene_slope_10.xml": FADA_SLOPE_10_GEOMETRY,
    "scene_slope_15.xml": FADA_SLOPE_15_GEOMETRY,
}
FADA_SLOPE_GEOMETRY_BY_TARGET_DOMAIN_ID = {
    "g1_slope_10_mujoco": FADA_SLOPE_10_GEOMETRY,
    "g1_slope_15_mujoco": FADA_SLOPE_15_GEOMETRY,
}


@dataclass(frozen=True)
class FADATargetDomainSpec:
    target_domain_id: str
    kind: Literal["slope", "actuator_gain"]
    task: str
    task_name: str
    backend: str
    command_sequence: tuple[tuple[float, float, float], ...]
    slope: FADASlopeGeometry | None = None
    actuator_index: int | None = None
    actuator_strength: float | None = None
    actuator_count: int | None = None
    legacy_fault_profile: str | None = None


_SLOPE_DISABLED_FLAGS = (
    "env.curriculum.enabled",
    "env.control_config.simulate_action_latency",
    "env.domain_rand.randomize_reset_pose",
    "env.domain_rand.randomize_base_mass",
    "env.domain_rand.randomize_body_mass",
    "env.domain_rand.random_com",
    "env.domain_rand.randomize_gravity",
    "env.domain_rand.randomize_ground_friction",
    "env.domain_rand.randomize_dof_armature",
    "env.domain_rand.push_robots",
    "env.domain_rand.randomize_kp",
    "env.domain_rand.randomize_kd",
    "env.domain_rand.randomize_dof_position_bias",
    "env.domain_rand.randomize_control_delay",
    "env.domain_rand.actuator_strength.enabled",
)


def assert_nominal_slope_environment(
    cfg: DictConfig,
    domain: FADATargetDomainSpec,
    *,
    task_choice: str | None,
) -> None:
    """Reject any perturbation mixed into a slope collection or evaluation."""

    if domain.kind != "slope" or domain.slope is None:
        raise ValueError("nominal slope validation requires a slope target domain")
    if task_choice != domain.task:
        raise ValueError(f"FADA target task must be {domain.task}")
    if str(OmegaConf.select(cfg, "training.task_name")) != domain.task_name:
        raise ValueError(f"FADA target training.task_name must be {domain.task_name}")
    if str(OmegaConf.select(cfg, "training.sim_backend")) != domain.backend:
        raise ValueError(f"FADA target training.sim_backend must be {domain.backend}")
    scene = str(OmegaConf.select(cfg, "env.scene.model_file"))
    scene_name = scene.rsplit("/", maxsplit=1)[-1]
    expected_geometry = FADA_SLOPE_GEOMETRY_BY_SCENE.get(scene_name)
    if expected_geometry is None:
        raise ValueError("FADA target env.scene.model_file must select a registered slope scene")
    if domain.slope != expected_geometry:
        raise ValueError(
            f"FADA target slope geometry must match the canonical {scene_name} geometry"
        )
    noise_level = OmegaConf.select(cfg, "env.noise_config.level")
    if isinstance(noise_level, bool) or float(noise_level) != 0.0:
        raise ValueError("FADA target env.noise_config.level must be 0.0")
    for path in _SLOPE_DISABLED_FLAGS:
        if OmegaConf.select(cfg, path) is not False:
            raise ValueError(f"FADA target {path} must be false")
    torque_rfi = OmegaConf.select(cfg, "env.domain_rand.torque_rfi_fraction")
    if isinstance(torque_rfi, bool) or float(torque_rfi) != 0.0:
        raise ValueError("FADA target env.domain_rand.torque_rfi_fraction must be 0.0")


def _mapping(cfg: DictConfig, key: str) -> dict[str, Any] | None:
    selected = OmegaConf.select(cfg, key)
    if selected is None:
        return None
    value = OmegaConf.to_container(selected, resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"FADA {key} config must be a mapping")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"FADA {key} config keys must be strings")
    return cast(dict[str, Any], value)


def _required(raw: dict[str, Any], key: str) -> Any:
    try:
        return raw[key]
    except KeyError as exc:
        raise ValueError(f"incomplete FADA target config: missing {key}") from exc


def _resolve_slope(raw: dict[str, Any]) -> FADATargetDomainSpec:
    forbidden = {"actuator_index", "actuator_strength", "actuator_count"} & raw.keys()
    if forbidden:
        raise ValueError("FADA slope target must not contain actuator fields")
    slope_raw = _required(raw, "slope")
    if not isinstance(slope_raw, dict):
        raise ValueError("FADA slope config must be a mapping")
    try:
        slope = FADASlopeGeometry(
            angle_deg=_finite_float(slope_raw["angle_deg"], "slope angle_deg"),
            width_m=_positive_float(slope_raw["width_m"], "slope width_m"),
            approach_length_m=_positive_float(
                slope_raw["approach_length_m"], "slope approach_length_m"
            ),
            surface_length_m=_positive_float(
                slope_raw["surface_length_m"], "slope surface_length_m"
            ),
            entry_margin_m=_finite_float(slope_raw["entry_margin_m"], "slope entry_margin_m"),
            finish_margin_m=_finite_float(slope_raw["finish_margin_m"], "slope finish_margin_m"),
        )
    except KeyError as exc:
        raise ValueError("incomplete FADA slope config") from exc
    return FADATargetDomainSpec(
        target_domain_id=str(_required(raw, "target_domain_id")),
        kind="slope",
        task=str(_required(raw, "task")),
        task_name=str(_required(raw, "task_name")),
        backend=str(_required(raw, "backend")),
        command_sequence=_resolve_slope_commands(raw),
        slope=slope,
    )


def _resolve_legacy_fault(raw: dict[str, Any]) -> FADATargetDomainSpec:
    command_limit = _required(raw, "command_limit")
    if not isinstance(command_limit, list) or not command_limit:
        raise ValueError("incomplete FADA fault command_limit")
    command = validate_fada_slope_commands([command_limit[0]])
    try:
        actuator_index = int(raw["actuator_index"])
        actuator_count = int(raw["actuator_count"])
        actuator_strength = _finite_float(raw["actuator_strength"], "actuator_strength")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("incomplete FADA actuator fault config") from exc
    if not 0 <= actuator_index < actuator_count or actuator_count <= 0:
        raise ValueError("FADA actuator index/count is invalid")
    return FADATargetDomainSpec(
        target_domain_id=str(_required(raw, "name")),
        kind="actuator_gain",
        task=str(_required(raw, "task")),
        task_name=str(_required(raw, "task_name")),
        backend=str(_required(raw, "backend")),
        command_sequence=command,
        actuator_index=actuator_index,
        actuator_strength=actuator_strength,
        actuator_count=actuator_count,
        legacy_fault_profile=str(_required(raw, "fault_profile")),
    )


def resolve_fada_target_domain(cfg: DictConfig) -> FADATargetDomainSpec:
    target = _mapping(cfg, "target_domain")
    fault = _mapping(cfg, "fault")
    if target is not None and fault is not None:
        raise ValueError("FADA config must not contain both target_domain and fault")
    if target is None:
        if fault is None:
            raise ValueError("FADA target_domain config is missing")
        return _resolve_legacy_fault(fault)
    kind = target.get("kind")
    if kind == "slope":
        return _resolve_slope(target)
    if kind == "actuator_gain":
        return _resolve_legacy_fault(target)
    raise ValueError(f"unsupported FADA target kind: {kind!r}")
