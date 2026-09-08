"""Typed v012 privileged-Oracle observation and checkpoint contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from unilab.envs.locomotion.g1.fada_privileged import (
    FADA_PRIVILEGED_SCHEMA,
    G1FADAPrivilegedLayout,
    G1FADAPrivilegedObservation,
    build_g1_fada_privileged_layout,
    pack_g1_fada_privileged_observation,
)

FADA_ORACLE_INTERMEDIATE_ITERATIONS = tuple(range(240, 4801, 240))
FADA_ORACLE_FINAL_ITERATION = 5000
FADA_ORACLE_CHECKPOINT_SCHEMA_VERSION = 3
FADA_ORACLE_PHASE_NEUTRAL_PROFILE = "phase_neutral_mixed_v1"
FADA_ORACLE_PHASE_LOCOMOTION_PROFILE = "phase_locomotion_v1"
FADA_ORACLE_COMMAND_PHASE_PROFILE = "command_phase_mixed_v1"
FADA_ORACLE_SIMPLE_HEIGHT_PROFILE = "simple_height_mixed_v1"
FADA_ORACLE_PHASE_HEIGHT_PROFILE = "phase_height_mixed_v3"
FADA_ORACLE_ORIGINAL_HEIGHT_PROFILE = "original_height_mixed_v1"
FADA_ORACLE_COMMAND_GATED_PHASE_CONTACT_PROFILE = "command_gated_phase_contact_v1"
FADA_ORACLE_FIXED_PHASE_CONTACT_PROFILE = "fixed_phase_contact_v1"
FADA_ORACLE_FIXED_PHASE_CONTACT_V2_PROFILE = "fixed_phase_contact_v2"
FADA_ORACLE_CONFIGURED_PROFILE = "configured_gait"
FADA_ORACLE_BEHAVIOR_PROFILES = frozenset(
    {
        FADA_ORACLE_CONFIGURED_PROFILE,
        FADA_ORACLE_PHASE_NEUTRAL_PROFILE,
        FADA_ORACLE_PHASE_LOCOMOTION_PROFILE,
        FADA_ORACLE_COMMAND_PHASE_PROFILE,
        FADA_ORACLE_SIMPLE_HEIGHT_PROFILE,
        FADA_ORACLE_PHASE_HEIGHT_PROFILE,
        FADA_ORACLE_ORIGINAL_HEIGHT_PROFILE,
        FADA_ORACLE_COMMAND_GATED_PHASE_CONTACT_PROFILE,
        FADA_ORACLE_FIXED_PHASE_CONTACT_PROFILE,
        FADA_ORACLE_FIXED_PHASE_CONTACT_V2_PROFILE,
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class FADAOracleBehaviorSpec:
    """Observation and reward semantics; experiment numbers belong to YAML."""

    gait_phase_enabled: bool
    gait_phase_init_mode: str | None
    feet_phase_mode: str = "legacy"
    mode_observation: bool = False


_FADA_ORACLE_BEHAVIOR_SPECS = {
    FADA_ORACLE_PHASE_NEUTRAL_PROFILE: FADAOracleBehaviorSpec(False, None),
    FADA_ORACLE_PHASE_LOCOMOTION_PROFILE: FADAOracleBehaviorSpec(True, "offset_phase"),
    FADA_ORACLE_COMMAND_PHASE_PROFILE: FADAOracleBehaviorSpec(
        True, "offset_phase", "filtered_command_height"
    ),
    FADA_ORACLE_SIMPLE_HEIGHT_PROFILE: FADAOracleBehaviorSpec(
        True, "offset_phase", "command_height"
    ),
    FADA_ORACLE_PHASE_HEIGHT_PROFILE: FADAOracleBehaviorSpec(
        True, "offset_phase", "phase_height"
    ),
    FADA_ORACLE_ORIGINAL_HEIGHT_PROFILE: FADAOracleBehaviorSpec(
        True, "offset_phase", "relative_command_height"
    ),
    FADA_ORACLE_COMMAND_GATED_PHASE_CONTACT_PROFILE: FADAOracleBehaviorSpec(
        True, "command_gated", "command_gated_contact"
    ),
    FADA_ORACLE_FIXED_PHASE_CONTACT_PROFILE: FADAOracleBehaviorSpec(
        True, "offset_phase", "fixed_contact"
    ),
    FADA_ORACLE_FIXED_PHASE_CONTACT_V2_PROFILE: FADAOracleBehaviorSpec(
        True, "offset_phase", "fixed_contact"
    ),
}


def fada_oracle_behavior_spec(behavior_profile: str) -> FADAOracleBehaviorSpec:
    try:
        return _FADA_ORACLE_BEHAVIOR_SPECS[behavior_profile]
    except KeyError as exc:
        raise ValueError(f"unsupported FADA Oracle behavior profile: {behavior_profile!r}") from exc


def _numeric(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _identity_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_fada_config_sha256(value: Any) -> str:
    """Return the canonical digest used by the v012 Oracle checkpoint contract."""

    return _canonical_json_sha256(value)


@dataclass(frozen=True)
class FADAOracleCheckpointIdentity:
    contract: "FADAOracleCheckpointContract"
    iteration: int
    role: str

    def to_record(self) -> dict[str, Any]:
        contract = self.contract
        return {
            **({"checkpoint_schedule": [contract.final_iteration, contract.save_interval]}
               if contract.behavior_profile == FADA_ORACLE_CONFIGURED_PROFILE else {}),
            "schema_version": FADA_ORACLE_CHECKPOINT_SCHEMA_VERSION,
            "oracle_lineage_id": contract.oracle_lineage_id,
            "iteration": self.iteration,
            "role": self.role,
            "privileged_schema": contract.privileged_schema,
            "task_name": contract.task_name,
            "backend": contract.backend,
            "action_scale": list(contract.action_scale),
            "seed": contract.seed,
            "dimensions": {
                "obs": contract.obs_dim,
                "critic": contract.critic_obs_dim,
                "privileged": contract.critic_obs_dim - contract.obs_dim,
                "action": contract.action_dim,
            },
            "body_names": list(contract.body_names),
            "actuated_joint_names": list(contract.actuated_joint_names),
            "privileged_field_slices": [list(row) for row in contract.privileged_field_slices],
            "asset_sha256": contract.asset_sha256,
            "config_hashes": dict(contract.config_hashes),
            "actor_directly_privileged": True,
            "behavior_profile": contract.behavior_profile,
        }


@dataclass(frozen=True)
class FADAOracleCheckpointContract:
    oracle_lineage_id: str
    privileged_schema: str
    task_name: str
    backend: str
    action_scale: tuple[float, ...]
    seed: int
    obs_dim: int
    critic_obs_dim: int
    action_dim: int
    body_names: tuple[str, ...]
    actuated_joint_names: tuple[str, ...]
    privileged_field_slices: tuple[tuple[str, int, int], ...]
    asset_sha256: str
    config_hashes: tuple[tuple[str, str], ...]
    behavior_profile: str = FADA_ORACLE_PHASE_NEUTRAL_PROFILE
    final_iteration: int = FADA_ORACLE_FINAL_ITERATION
    save_interval: int = 240

    def __post_init__(self) -> None:
        if self.final_iteration <= 0 or self.save_interval <= 0:
            raise ValueError("checkpoint final_iteration and save_interval must be positive")
        if self.behavior_profile != FADA_ORACLE_CONFIGURED_PROFILE and (
            self.final_iteration != FADA_ORACLE_FINAL_ITERATION or self.save_interval != 240
        ):
            raise ValueError("historical Oracle checkpoint schedule is fixed")
        if not self.oracle_lineage_id.strip():
            raise ValueError("oracle_lineage_id must be non-empty")
        if self.privileged_schema != FADA_PRIVILEGED_SCHEMA:
            raise ValueError("privileged_schema mismatch")
        if self.task_name != "G1WalkFlat" or self.backend != "mujoco":
            raise ValueError("FADA Oracle checkpoint requires G1WalkFlat/MuJoCo")
        if not self.action_scale or not all(math.isfinite(value) for value in self.action_scale):
            raise ValueError("action_scale must be finite and non-empty")
        if self.obs_dim <= 0 or self.critic_obs_dim <= self.obs_dim or self.action_dim <= 0:
            raise ValueError("checkpoint dimensions are incompatible")
        if len(set(self.body_names)) != len(self.body_names) or not self.body_names:
            raise ValueError("body_names must be non-empty and unique")
        if len(set(self.actuated_joint_names)) != len(self.actuated_joint_names):
            raise ValueError("actuated_joint_names must be unique")
        if len(self.actuated_joint_names) != self.action_dim:
            raise ValueError("actuated_joint_names must match action_dim")
        if not self.privileged_field_slices:
            raise ValueError("privileged_field_slices must be non-empty")
        if not _SHA256_RE.fullmatch(self.asset_sha256):
            raise ValueError("asset_sha256 must be a lowercase SHA-256 digest")
        config_keys = [key for key, _ in self.config_hashes]
        if not config_keys or len(config_keys) != len(set(config_keys)):
            raise ValueError("config_hashes must have unique non-empty keys")
        for key, digest in self.config_hashes:
            if not key or not _SHA256_RE.fullmatch(digest):
                raise ValueError("config_hashes must contain lowercase SHA-256 digests")
        if self.behavior_profile not in FADA_ORACLE_BEHAVIOR_PROFILES:
            raise ValueError(f"unsupported FADA Oracle behavior profile: {self.behavior_profile!r}")

    def identity_for_iteration(self, iteration: int) -> FADAOracleCheckpointIdentity:
        iteration = int(iteration)
        if iteration in self.intermediate_iterations:
            role = "idm_coverage"
        elif iteration == self.final_iteration:
            role = "final_oracle"
        else:
            raise ValueError(f"unsupported FADA Oracle checkpoint iteration: {iteration}")
        return FADAOracleCheckpointIdentity(self, iteration, role)

    @property
    def intermediate_iterations(self) -> tuple[int, ...]:
        return tuple(range(self.save_interval, self.final_iteration, self.save_interval))

    @property
    def fingerprint(self) -> str:
        record = self.identity_for_iteration(self.final_iteration).to_record()
        record.pop("iteration")
        record.pop("role")
        return _canonical_json_sha256(record)


def seal_fada_oracle_checkpoint(
    state_dict: Mapping[str, Any],
    contract: FADAOracleCheckpointContract,
    *,
    iteration: int,
) -> dict[str, Any]:
    payload = dict(state_dict)
    payload["fada_privileged_oracle"] = contract.identity_for_iteration(iteration).to_record()
    return payload


def validate_fada_oracle_checkpoint_payload(
    payload: Mapping[str, Any],
    contract: FADAOracleCheckpointContract,
    *,
    expected_iteration: int,
) -> FADAOracleCheckpointIdentity:
    expected = contract.identity_for_iteration(expected_iteration)
    expected_record = expected.to_record()
    actual = payload.get("fada_privileged_oracle")
    if not isinstance(actual, Mapping):
        raise ValueError("checkpoint missing fada_privileged_oracle identity")
    for key, expected_value in expected_record.items():
        if key not in actual:
            raise ValueError(f"checkpoint identity missing {key}")
        if actual[key] != expected_value:
            raise ValueError(f"checkpoint identity {key} mismatch")
    return expected


def normalize_fada_oracle_checkpoint_identity(
    identity: Mapping[str, object],
) -> dict[str, object]:
    """Normalize legacy identities without granting them phase authority."""

    normalized = dict(identity)
    schema_version = normalized.get("schema_version")
    if schema_version == 2:
        legacy_profile = normalized.get("behavior_profile", FADA_ORACLE_PHASE_NEUTRAL_PROFILE)
        if legacy_profile != FADA_ORACLE_PHASE_NEUTRAL_PROFILE:
            raise ValueError("legacy FADA Oracle schema is phase-neutral only")
        normalized["behavior_profile"] = FADA_ORACLE_PHASE_NEUTRAL_PROFILE
    elif schema_version == FADA_ORACLE_CHECKPOINT_SCHEMA_VERSION:
        behavior_profile = normalized.get("behavior_profile")
        if behavior_profile not in FADA_ORACLE_BEHAVIOR_PROFILES:
            raise ValueError("FADA Oracle checkpoint behavior_profile is missing or unsupported")
    else:
        raise ValueError(f"unsupported FADA Oracle checkpoint schema: {schema_version!r}")
    return normalized


def validate_fada_oracle_behavior_environment(
    env_cfg: Any,
    behavior_profile: str,
) -> None:
    """Check observation semantics, leaving experiment values to their YAML owner."""

    if behavior_profile == FADA_ORACLE_CONFIGURED_PROFILE:
        # Cross-field dependencies are validated by the typed G1 environment.
        return
    spec = fada_oracle_behavior_spec(behavior_profile)
    if bool(getattr(env_cfg, "mode_observation", True)) != spec.mode_observation:
        raise ValueError(f"FADA Oracle {behavior_profile} mode_observation mismatch")
    if bool(getattr(env_cfg, "gait_phase_enabled", False)) != spec.gait_phase_enabled:
        raise ValueError(f"FADA Oracle {behavior_profile} gait phase enabled mismatch")
    if (
        spec.gait_phase_init_mode is not None
        and str(getattr(env_cfg, "gait_phase_init_mode", "")) != spec.gait_phase_init_mode
    ):
        raise ValueError(
            f"phase locomotion Oracle requires gait_phase_init_mode={spec.gait_phase_init_mode}"
        )
    command_gait = getattr(env_cfg, "command_gated_phase_contact", None)
    gated = bool(getattr(command_gait, "enabled", False))
    if gated != (spec.feet_phase_mode == "command_gated_contact"):
        raise ValueError("Oracle phase clock and command-gated environment owner disagree")


class FADAOracleCheckpointGateway:
    """Cold-path save/reload/finalize owner for one v012 Oracle lineage."""

    def __init__(self, contract: FADAOracleCheckpointContract) -> None:
        self.contract = contract

    @staticmethod
    def _checkpoint_name(iteration: int) -> str:
        return f"model_{int(iteration)}.pt"

    def save(self, learner: Any, path: str | Path, iteration: int) -> None:
        target = Path(path)
        iteration = int(iteration)
        expected_name = self._checkpoint_name(iteration)
        if target.name != expected_name:
            raise ValueError(
                f"checkpoint filename/iteration mismatch: expected {expected_name}, got {target.name}"
            )
        self.contract.identity_for_iteration(iteration)
        if iteration == self.contract.final_iteration:
            missing = [
                self._checkpoint_name(value)
                for value in self.contract.intermediate_iterations
                if not (target.parent / self._checkpoint_name(value)).is_file()
            ]
            if missing:
                raise ValueError(f"missing FADA Oracle intermediate checkpoints: {missing}")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = seal_fada_oracle_checkpoint(
            learner.get_state_dict(), self.contract, iteration=iteration
        )
        torch.save(payload, target)
        reloaded = torch.load(target, map_location="cpu", weights_only=True)
        if not isinstance(reloaded, Mapping):
            raise ValueError("saved FADA Oracle checkpoint must be a mapping")
        validate_fada_oracle_checkpoint_payload(
            reloaded, self.contract, expected_iteration=iteration
        )
        if iteration == self.contract.final_iteration:
            self.finalize(target.parent)

    def finalize(self, directory: str | Path) -> Path:
        root = Path(directory)
        expected_iterations = (
            *self.contract.intermediate_iterations,
            self.contract.final_iteration,
        )
        expected_names = {self._checkpoint_name(value) for value in expected_iterations}
        observed_names = {path.name for path in root.glob("model_*.pt")}
        missing = sorted(expected_names - observed_names)
        extra = sorted(observed_names - expected_names)
        if missing:
            raise ValueError(f"missing FADA Oracle checkpoints: {missing}")
        if extra:
            raise ValueError(f"extra FADA Oracle checkpoints: {extra}")

        records: list[dict[str, Any]] = []
        checkpoint_hashes: dict[str, str] = {}
        for iteration in expected_iterations:
            path = root / self._checkpoint_name(iteration)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, Mapping):
                raise ValueError(f"checkpoint {path.name} must be a mapping")
            validate_fada_oracle_checkpoint_payload(
                payload, self.contract, expected_iteration=iteration
            )
            records.append(dict(payload["fada_privileged_oracle"]))
            checkpoint_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

        admitted = validate_fada_oracle_lineage(records)
        manifest = {
            "schema_version": FADA_ORACLE_CHECKPOINT_SCHEMA_VERSION,
            "oracle_lineage_id": admitted.oracle_lineage_id,
            "contract_fingerprint": self.contract.fingerprint,
            "intermediate_iterations": list(admitted.intermediate_iterations),
            "final_iteration": admitted.final_iteration,
            "checkpoint_sha256": checkpoint_hashes,
        }
        target = root / "fada_oracle_lineage.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(target)
        return target


def validate_no_gait_reward(reward_scales: Mapping[str, object]) -> None:
    forbidden_tokens = ("gait", "phase", "footfall")
    for name, raw_scale in reward_scales.items():
        normalized = str(name).lower()
        if not any(token in normalized for token in forbidden_tokens):
            continue
        try:
            scale = _numeric(raw_scale, name=f"gait reward {name!r}")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"gait reward {name!r} must be numeric and zero") from exc
        if scale != 0.0:
            raise ValueError(f"gait reward {name!r} must be disabled, got {scale}")


def _reject_stand_reward_authority(value: object, *, path: str = "reward") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}"
            if key.startswith("stand_"):
                raise ValueError(f"FADA Oracle single Reward forbids {child_path}")
            _reject_stand_reward_authority(child, path=child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_stand_reward_authority(child, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and value.startswith("stand_"):
        raise ValueError(f"FADA Oracle single Reward forbids term {value!r} at {path}")


def validate_fada_single_reward(
    *,
    reward_scales: Mapping[str, object],
    reward_config: Mapping[str, object],
    behavior_profile: str = FADA_ORACLE_PHASE_NEUTRAL_PROFILE,
) -> None:
    """Check the single-reward interface, not a second copy of experiment settings."""

    from unilab.envs.locomotion.g1.walk_config import G1RewardConfig, normalize_g1_gait_reward

    if behavior_profile == FADA_ORACLE_CONFIGURED_PROFILE:
        G1RewardConfig(**dict(reward_config, scales=dict(reward_scales)))
        return
    spec = fada_oracle_behavior_spec(behavior_profile)
    for name, scale in reward_scales.items():
        if not math.isfinite(_numeric(scale, name=f"reward.scales.{name}")):
            raise ValueError(f"reward.scales.{name} must be finite")
    normalized = normalize_g1_gait_reward(dict(reward_config, scales=dict(reward_scales)))
    if normalized["feet_phase_mode"] != spec.feet_phase_mode:
        raise ValueError(f"{behavior_profile} requires feet_phase_mode={spec.feet_phase_mode}")
    if behavior_profile == FADA_ORACLE_PHASE_NEUTRAL_PROFILE:
        validate_no_gait_reward(reward_scales)
    reward_mode = reward_config.get("mode")
    if isinstance(reward_mode, Mapping) and reward_mode:
        raise ValueError("FADA Oracle single Reward forbids reward.mode dispatcher")
    _reject_stand_reward_authority(reward_config)

    gait_constraint = reward_config.get("gait_constraint", {})
    if not isinstance(gait_constraint, Mapping):
        raise ValueError("FADA Oracle gait_constraint must be a mapping")
    if float(gait_constraint.get("penalty_scale", 0.0)) != 0.0:
        raise ValueError("FADA Oracle gait constraint penalty_scale must be zero")
    if bool(gait_constraint.get("enabled", False)):
        raise ValueError("FADA Oracle gait constraint mode must be disabled")


@dataclass(frozen=True)
class AdmittedFADAOracleLineage:
    oracle_lineage_id: str
    intermediate_iterations: tuple[int, ...]
    final_iteration: int
    behavior_profile: str


def validate_fada_oracle_lineage(
    records: Sequence[Mapping[str, object]],
) -> AdmittedFADAOracleLineage:
    final_iteration, save_interval = FADA_ORACLE_FINAL_ITERATION, 240
    configured = bool(records) and records[0].get("behavior_profile") == FADA_ORACLE_CONFIGURED_PROFILE
    if configured:
        schedule = records[0].get("checkpoint_schedule")
        if not isinstance(schedule, (list, tuple)) or len(schedule) != 2:
            raise ValueError("configured Oracle requires checkpoint_schedule")
        final_iteration, save_interval = (
            _identity_int(value, name="checkpoint schedule") for value in schedule
        )
        if min(final_iteration, save_interval) <= 0:
            raise ValueError("checkpoint schedule must be positive")
        if any(record.get("checkpoint_schedule") != schedule for record in records):
            raise ValueError("oracle lineage checkpoint schedule mismatch")
    expected_intermediate = tuple(range(save_interval, final_iteration, save_interval))
    if len(records) != len(expected_intermediate) + 1:
        raise ValueError(
            "oracle lineage must contain every scheduled intermediate and final record"
        )
    schema_versions = {record.get("schema_version") for record in records}
    if len(schema_versions) != 1:
        raise ValueError("oracle lineage checkpoint schema mismatch")
    normalized_records = [normalize_fada_oracle_checkpoint_identity(record) for record in records]
    lineage_ids = {str(record.get("oracle_lineage_id", "")) for record in normalized_records}
    if len(lineage_ids) != 1 or "" in lineage_ids:
        raise ValueError("oracle lineage records must share one non-empty lineage id")
    if {record.get("task_name") for record in normalized_records} != {"G1WalkFlat"}:
        raise ValueError("oracle lineage task must be exactly G1WalkFlat")
    if {record.get("privileged_schema") for record in normalized_records} != {
        FADA_PRIVILEGED_SCHEMA
    }:
        raise ValueError("oracle lineage privileged schema mismatch")
    stable_identity_keys = (
        "backend",
        "action_scale",
        "seed",
        "dimensions",
        "body_names",
        "actuated_joint_names",
        "privileged_field_slices",
        "asset_sha256",
        "config_hashes",
        "actor_directly_privileged",
        "behavior_profile",
    )
    for key in stable_identity_keys:
        if any(key not in record for record in normalized_records):
            raise ValueError(f"oracle lineage missing {key}")
        if len({_canonical_json_sha256(record[key]) for record in normalized_records}) != 1:
            raise ValueError(f"oracle lineage {key} mismatch")
    intermediate = tuple(
        _identity_int(record.get("iteration"), name="oracle iteration")
        for record in normalized_records[:-1]
    )
    if intermediate != expected_intermediate:
        raise ValueError("oracle intermediate iterations disagree with checkpoint schedule")
    if any(record.get("role") != "idm_coverage" for record in normalized_records[:-1]):
        raise ValueError("oracle intermediate checkpoint role must be idm_coverage")
    final = normalized_records[-1]
    if (
        _identity_int(final.get("iteration"), name="oracle final iteration")
        != final_iteration
    ):
        raise ValueError("oracle final iteration disagrees with checkpoint schedule")
    if final.get("role") != "final_oracle":
        raise ValueError("oracle final checkpoint role must be final_oracle")
    return AdmittedFADAOracleLineage(
        oracle_lineage_id=next(iter(lineage_ids)),
        intermediate_iterations=intermediate,
        final_iteration=final_iteration,
        behavior_profile=str(final["behavior_profile"]),
    )


__all__ = [
    "FADA_ORACLE_CONFIGURED_PROFILE",
    "AdmittedFADAOracleLineage",
    "FADA_ORACLE_CHECKPOINT_SCHEMA_VERSION",
    "FADA_ORACLE_COMMAND_GATED_PHASE_CONTACT_PROFILE",
    "FADA_ORACLE_FIXED_PHASE_CONTACT_PROFILE",
    "FADA_ORACLE_FIXED_PHASE_CONTACT_V2_PROFILE",
    "FADA_ORACLE_PHASE_LOCOMOTION_PROFILE",
    "FADA_ORACLE_PHASE_NEUTRAL_PROFILE",
    "FADA_ORACLE_FINAL_ITERATION",
    "FADA_ORACLE_INTERMEDIATE_ITERATIONS",
    "FADA_PRIVILEGED_SCHEMA",
    "FADAOracleCheckpointContract",
    "FADAOracleCheckpointGateway",
    "FADAOracleCheckpointIdentity",
    "FADAOracleBehaviorSpec",
    "G1FADAPrivilegedLayout",
    "G1FADAPrivilegedObservation",
    "build_g1_fada_privileged_layout",
    "canonical_fada_config_sha256",
    "fada_oracle_behavior_spec",
    "pack_g1_fada_privileged_observation",
    "normalize_fada_oracle_checkpoint_identity",
    "validate_fada_oracle_behavior_environment",
    "seal_fada_oracle_checkpoint",
    "validate_fada_single_reward",
    "validate_fada_oracle_checkpoint_payload",
    "validate_fada_oracle_lineage",
    "validate_no_gait_reward",
]
