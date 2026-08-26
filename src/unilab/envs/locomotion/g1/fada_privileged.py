"""G1 owner helpers for the v012 FADA privileged observation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

DOF_POSITION_BIAS_LIMIT_RAD = 0.025
TORQUE_RFI_FRACTION = 0.05
FADA_PRIVILEGED_SCHEMA = "g1_fada_privileged_v1"

_FIELD_WIDTHS = (
    ("base_linear_velocity", 3),
    ("foot_contact_resultants", 6),
    ("foot_contact_flags", 2),
    ("terrain_heights", 9),
    ("root_clearance", 1),
    ("kp_scale", 29),
    ("kd_scale", 29),
    ("normalized_torque", 29),
    ("ground_friction", 1),
    ("base_com_shift", 3),
    ("added_base_mass", 1),
    ("body_mass_scale", None),
    ("dof_position_bias", 29),
    ("torque_rfi", 29),
    ("control_delay", 1),
    ("push_interval", 1),
    ("push_velocity", 1),
)


@dataclass
class G1FADAPrivilegedObservationConfig:
    enabled: bool = False
    schema: str = "g1_fada_privileged_v1"


@dataclass
class G1FADAPrivilegedObservation:
    base_linear_velocity: np.ndarray
    foot_contact_resultants: np.ndarray
    foot_contact_flags: np.ndarray
    terrain_heights: np.ndarray
    root_clearance: np.ndarray
    kp_scale: np.ndarray
    kd_scale: np.ndarray
    normalized_torque: np.ndarray
    ground_friction: np.ndarray
    base_com_shift: np.ndarray
    added_base_mass: np.ndarray
    body_mass_scale: np.ndarray
    dof_position_bias: np.ndarray
    torque_rfi: np.ndarray
    control_delay: np.ndarray
    push_interval: np.ndarray
    push_velocity: np.ndarray


@dataclass(frozen=True)
class G1FADAPrivilegedLayout:
    schema: str
    body_names: tuple[str, ...]
    field_slices: tuple[tuple[str, int, int], ...]
    width: int

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.field_slices)

    def slice_for(self, field_name: str) -> slice:
        for name, start, stop in self.field_slices:
            if name == field_name:
                return slice(start, stop)
        raise KeyError(field_name)


@dataclass(frozen=True)
class G1FADAPrivilegedCheckpointLayoutIdentity:
    body_names: tuple[str, ...]
    actuated_joint_names: tuple[str, ...]
    field_slices: tuple[tuple[str, int, int], ...]
    asset_sha256: str


def build_g1_fada_privileged_layout(
    body_names: Sequence[str],
) -> G1FADAPrivilegedLayout:
    ordered_bodies = tuple(str(name) for name in body_names)
    if not ordered_bodies or len(set(ordered_bodies)) != len(ordered_bodies):
        raise ValueError("body_names must be non-empty, unique, and in fixed asset order")
    cursor = 0
    slices: list[tuple[str, int, int]] = []
    for name, static_width in _FIELD_WIDTHS:
        width = len(ordered_bodies) if static_width is None else static_width
        slices.append((name, cursor, cursor + width))
        cursor += width
    return G1FADAPrivilegedLayout(
        schema=FADA_PRIVILEGED_SCHEMA,
        body_names=ordered_bodies,
        field_slices=tuple(slices),
        width=cursor,
    )


def build_g1_fada_checkpoint_layout_identity(
    *,
    body_names: Sequence[str],
    actuated_joint_names: Sequence[str],
    model_file: str | Path,
) -> G1FADAPrivilegedCheckpointLayoutIdentity:
    """Seal the G1 layout and XML asset tree on the environment cold path."""

    layout = build_g1_fada_privileged_layout(body_names)
    joints = tuple(str(name) for name in actuated_joint_names)
    if not joints or len(joints) != len(set(joints)):
        raise ValueError("actuated_joint_names must be non-empty, unique, and ordered")
    model_path = Path(model_file).resolve()
    if not model_path.is_file():
        raise ValueError(f"G1 model_file does not exist: {model_path}")
    xml_paths = sorted(path for path in model_path.parent.rglob("*.xml") if path.is_file())
    if model_path not in xml_paths:
        xml_paths.append(model_path)
        xml_paths.sort()
    digest = hashlib.sha256()
    for path in xml_paths:
        digest.update(path.relative_to(model_path.parent).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return G1FADAPrivilegedCheckpointLayoutIdentity(
        body_names=layout.body_names,
        actuated_joint_names=joints,
        field_slices=layout.field_slices,
        asset_sha256=digest.hexdigest(),
    )


def pack_g1_fada_privileged_observation(
    observation: G1FADAPrivilegedObservation,
    layout: G1FADAPrivilegedLayout,
) -> np.ndarray:
    if layout.schema != FADA_PRIVILEGED_SCHEMA:
        raise ValueError(f"unsupported privileged schema: {layout.schema!r}")
    declared = tuple(field.name for field in fields(G1FADAPrivilegedObservation))
    if declared != layout.field_names:
        raise ValueError("privileged layout field order does not match the typed payload")
    arrays: list[np.ndarray] = []
    batch_size: int | None = None
    for name in layout.field_names:
        value = np.asarray(getattr(observation, name))
        field_slice = layout.slice_for(name)
        expected_width = field_slice.stop - field_slice.start
        if value.ndim != 2 or value.shape[1] != expected_width:
            raise ValueError(f"{name} must have shape (batch, {expected_width}), got {value.shape}")
        if batch_size is None:
            batch_size = int(value.shape[0])
        elif value.shape[0] != batch_size:
            raise ValueError(f"{name} batch size differs from the preceding fields")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
        arrays.append(value.astype(np.float32, copy=False))
    return np.concatenate(arrays, axis=1)


def build_fada_reset_info(
    payload: object,
    *,
    rows: int,
    num_actions: int,
    base_kp: np.ndarray,
    base_kd: np.ndarray,
    base_body_mass: np.ndarray,
    base_geom_friction: np.ndarray,
    ground_geom_id: int,
    dof_position_bias: np.ndarray,
    control_delay: np.ndarray,
    push_interval_seconds: float,
    push_velocity: float,
    dtype: np.dtype,
) -> dict[str, np.ndarray]:
    """Build reset-persistent privilege fields from one applied DR payload."""

    def value(name: str) -> np.ndarray | None:
        raw = getattr(payload, name, None)
        return None if raw is None else np.asarray(raw)

    def zeros_or(name: str, width: int) -> np.ndarray:
        raw = value(name)
        return (
            np.zeros((rows, width), dtype=dtype)
            if raw is None
            else np.asarray(raw, dtype=dtype).reshape(rows, width)
        )

    def gain_scale(name: str, baseline: np.ndarray) -> np.ndarray:
        base = np.asarray(baseline, dtype=np.float64)
        raw = value(name)
        sampled = np.broadcast_to(base, (rows, base.size)) if raw is None else raw
        return np.asarray(sampled / base[None, :], dtype=dtype)

    friction = value("geom_friction")
    friction_table = (
        np.broadcast_to(base_geom_friction, (rows, *base_geom_friction.shape))
        if friction is None
        else friction
    )
    body_mass = value("body_mass")
    mass = np.asarray(base_body_mass, dtype=np.float64)
    sampled_mass = np.broadcast_to(mass, (rows, mass.size)) if body_mass is None else body_mass
    safe_mass = np.where(mass == 0.0, 1.0, mass)
    mass_scale = sampled_mass / safe_mass[None, :]
    mass_scale[:, mass == 0.0] = 1.0

    return {
        "fada_dof_position_bias": np.asarray(dof_position_bias, dtype=dtype),
        "fada_torque_rfi": np.zeros((rows, num_actions), dtype=dtype),
        "fada_control_delay": np.asarray(control_delay, dtype=dtype),
        "fada_kp_scale": gain_scale("kp", base_kp),
        "fada_kd_scale": gain_scale("kd", base_kd),
        "fada_ground_friction": np.asarray(friction_table[:, ground_geom_id, 0:1], dtype=dtype),
        "fada_base_com_shift": zeros_or("base_com_offset", 3),
        "fada_added_base_mass": zeros_or("base_mass_delta", 1),
        "fada_body_mass_scale": np.asarray(mass_scale, dtype=dtype),
        "fada_push_interval": np.full((rows, 1), push_interval_seconds, dtype=dtype),
        "fada_push_velocity": np.full((rows, 1), push_velocity, dtype=dtype),
    }


def pack_fada_runtime_observation(
    *,
    body_names: Sequence[str],
    tau_max: np.ndarray,
    linvel: np.ndarray,
    left_contact_sensor: np.ndarray,
    right_contact_sensor: np.ndarray,
    root_clearance: np.ndarray,
    torques: np.ndarray,
    info: dict[str, np.ndarray],
    dtype: np.dtype,
) -> np.ndarray:
    """Pack one runtime privilege batch from already-cached numeric sources."""
    velocity = np.asarray(linvel, dtype=dtype)
    rows = velocity.shape[0]
    left_force, left_flag = split_net_contact_sensor(left_contact_sensor)
    right_force, right_flag = split_net_contact_sensor(right_contact_sensor)
    normalized_torque = np.asarray(torques, dtype=dtype) / np.maximum(
        np.asarray(tau_max)[None, :], 1.0e-6
    )
    bundle = G1FADAPrivilegedObservation(
        base_linear_velocity=velocity,
        foot_contact_resultants=np.concatenate([left_force, right_force], axis=1),
        foot_contact_flags=np.concatenate([left_flag, right_flag], axis=1),
        terrain_heights=np.zeros((rows, 9), dtype=dtype),
        root_clearance=np.asarray(root_clearance, dtype=dtype).reshape(rows, 1),
        kp_scale=np.asarray(info["fada_kp_scale"]),
        kd_scale=np.asarray(info["fada_kd_scale"]),
        normalized_torque=normalized_torque,
        ground_friction=np.asarray(info["fada_ground_friction"]),
        base_com_shift=np.asarray(info["fada_base_com_shift"]),
        added_base_mass=np.asarray(info["fada_added_base_mass"]),
        body_mass_scale=np.asarray(info["fada_body_mass_scale"]),
        dof_position_bias=np.asarray(info["fada_dof_position_bias"]),
        torque_rfi=np.asarray(info["fada_torque_rfi"]),
        control_delay=np.asarray(info["fada_control_delay"]),
        push_interval=np.asarray(info["fada_push_interval"]),
        push_velocity=np.asarray(info["fada_push_velocity"]),
    )
    return pack_g1_fada_privileged_observation(bundle, build_g1_fada_privileged_layout(body_names))


def split_net_contact_sensor(sensor_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split MuJoCo ``found force`` net-contact output into force and flag."""
    values = np.asarray(sensor_data)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"net contact sensor must have shape (num_envs, 4), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("net contact sensor must be finite")
    return values[:, 1:4], np.asarray(values[:, 0:1] > 0.0, dtype=values.dtype)


def apply_fada_pd_target_perturbation(
    target: np.ndarray,
    *,
    dof_position_bias: np.ndarray,
    torque_rfi: np.ndarray,
    kp: np.ndarray,
    tau_max: np.ndarray,
) -> np.ndarray:
    """Apply motor-zero bias and additive torque noise to position-servo targets.

    For a position actuator, shifting the target by ``tau_rfi / Kp`` adds
    exactly ``tau_rfi`` before actuator force clipping.
    """
    target_np = np.asarray(target)
    bias_np = np.asarray(dof_position_bias)
    rfi_np = np.asarray(torque_rfi)
    kp_np = np.asarray(kp)
    tau_limit = np.asarray(tau_max)
    if not (target_np.shape == bias_np.shape == rfi_np.shape == kp_np.shape):
        raise ValueError("target, bias, torque RFI, and Kp must have identical shapes")
    if tau_limit.shape != (target_np.shape[1],):
        raise ValueError(f"tau_max must have shape ({target_np.shape[1]},)")
    if np.any(kp_np <= 0.0) or not np.isfinite(kp_np).all():
        raise ValueError("Kp must be finite and strictly positive")
    if np.any(np.abs(bias_np) > DOF_POSITION_BIAS_LIMIT_RAD + 1.0e-12):
        raise ValueError("DoF position bias exceeds the confirmed moderate limit")
    if np.any(np.abs(rfi_np) > TORQUE_RFI_FRACTION * tau_limit[None, :] + 1.0e-12):
        raise ValueError("torque RFI exceeds the confirmed moderate limit")
    return target_np + bias_np + rfi_np / kp_np


__all__ = [
    "DOF_POSITION_BIAS_LIMIT_RAD",
    "FADA_PRIVILEGED_SCHEMA",
    "G1FADAPrivilegedCheckpointLayoutIdentity",
    "G1FADAPrivilegedLayout",
    "G1FADAPrivilegedObservation",
    "G1FADAPrivilegedObservationConfig",
    "TORQUE_RFI_FRACTION",
    "apply_fada_pd_target_perturbation",
    "build_fada_reset_info",
    "build_g1_fada_checkpoint_layout_identity",
    "build_g1_fada_privileged_layout",
    "pack_g1_fada_privileged_observation",
    "pack_fada_runtime_observation",
    "split_net_contact_sensor",
]
