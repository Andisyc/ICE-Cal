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
FADA_ORACLE_CHECKPOINT_SCHEMA_VERSION = 2
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


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

    def __post_init__(self) -> None:
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

    def identity_for_iteration(self, iteration: int) -> FADAOracleCheckpointIdentity:
        iteration = int(iteration)
        if iteration in FADA_ORACLE_INTERMEDIATE_ITERATIONS:
            role = "idm_coverage"
        elif iteration == FADA_ORACLE_FINAL_ITERATION:
            role = "final_oracle"
        else:
            raise ValueError(f"unsupported FADA Oracle checkpoint iteration: {iteration}")
        return FADAOracleCheckpointIdentity(self, iteration, role)

    @property
    def fingerprint(self) -> str:
        record = self.identity_for_iteration(FADA_ORACLE_FINAL_ITERATION).to_record()
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
        if iteration == FADA_ORACLE_FINAL_ITERATION:
            missing = [
                self._checkpoint_name(value)
                for value in FADA_ORACLE_INTERMEDIATE_ITERATIONS
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
        if iteration == FADA_ORACLE_FINAL_ITERATION:
            self.finalize(target.parent)

    def finalize(self, directory: str | Path) -> Path:
        root = Path(directory)
        expected_iterations = (
            *FADA_ORACLE_INTERMEDIATE_ITERATIONS,
            FADA_ORACLE_FINAL_ITERATION,
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
            scale = float(raw_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"gait reward {name!r} must be numeric and zero") from exc
        if scale != 0.0:
            raise ValueError(f"gait reward {name!r} must be disabled, got {scale}")


@dataclass(frozen=True)
class AdmittedFADAOracleLineage:
    oracle_lineage_id: str
    intermediate_iterations: tuple[int, ...]
    final_iteration: int


def validate_fada_oracle_lineage(
    records: Sequence[Mapping[str, object]],
) -> AdmittedFADAOracleLineage:
    if len(records) != len(FADA_ORACLE_INTERMEDIATE_ITERATIONS) + 1:
        raise ValueError(
            "oracle lineage iterations must contain exactly 20 intermediate + 1 final record"
        )
    lineage_ids = {str(record.get("oracle_lineage_id", "")) for record in records}
    if len(lineage_ids) != 1 or "" in lineage_ids:
        raise ValueError("oracle lineage records must share one non-empty lineage id")
    if {record.get("task_name") for record in records} != {"G1WalkFlat"}:
        raise ValueError("oracle lineage task must be exactly G1WalkFlat")
    if {record.get("privileged_schema") for record in records} != {FADA_PRIVILEGED_SCHEMA}:
        raise ValueError("oracle lineage privileged schema mismatch")
    if {record.get("schema_version") for record in records} != {
        FADA_ORACLE_CHECKPOINT_SCHEMA_VERSION
    }:
        raise ValueError("oracle lineage checkpoint schema mismatch")
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
    )
    for key in stable_identity_keys:
        if any(key not in record for record in records):
            raise ValueError(f"oracle lineage missing {key}")
        if len({_canonical_json_sha256(record[key]) for record in records}) != 1:
            raise ValueError(f"oracle lineage {key} mismatch")
    intermediate = tuple(int(record.get("iteration", -1)) for record in records[:-1])
    if intermediate != FADA_ORACLE_INTERMEDIATE_ITERATIONS:
        raise ValueError("oracle intermediate iterations must be exactly 240..4800 by 240")
    if any(record.get("role") != "idm_coverage" for record in records[:-1]):
        raise ValueError("oracle intermediate checkpoint role must be idm_coverage")
    final = records[-1]
    if int(final.get("iteration", -1)) != FADA_ORACLE_FINAL_ITERATION:
        raise ValueError("oracle final iteration must be 5000")
    if final.get("role") != "final_oracle":
        raise ValueError("oracle final checkpoint role must be final_oracle")
    return AdmittedFADAOracleLineage(
        oracle_lineage_id=next(iter(lineage_ids)),
        intermediate_iterations=intermediate,
        final_iteration=FADA_ORACLE_FINAL_ITERATION,
    )


__all__ = [
    "AdmittedFADAOracleLineage",
    "FADA_ORACLE_CHECKPOINT_SCHEMA_VERSION",
    "FADA_ORACLE_FINAL_ITERATION",
    "FADA_ORACLE_INTERMEDIATE_ITERATIONS",
    "FADA_PRIVILEGED_SCHEMA",
    "FADAOracleCheckpointContract",
    "FADAOracleCheckpointGateway",
    "FADAOracleCheckpointIdentity",
    "G1FADAPrivilegedLayout",
    "G1FADAPrivilegedObservation",
    "build_g1_fada_privileged_layout",
    "canonical_fada_config_sha256",
    "pack_g1_fada_privileged_observation",
    "seal_fada_oracle_checkpoint",
    "validate_fada_oracle_checkpoint_payload",
    "validate_fada_oracle_lineage",
    "validate_no_gait_reward",
]
