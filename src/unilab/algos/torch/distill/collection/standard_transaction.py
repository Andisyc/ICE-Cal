"""Explicit transaction owner for standard role distillation collection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from unilab.algos.torch.distill.collection.common import (
    _attach_collector_performance,
    _command_sample_mask,
    _info_array,
    _obs_array,
    _performance_span,
    _policy_actions,
    _reset_done_rows_after_step,
    _resolve_collection_reset,
    _target_height_array,
    command_active_mask,
    project_student_obs,
    project_teacher_obs,
)
from unilab.algos.torch.distill.datasets.dataset import (
    DistillationTensorDataset,
    build_distillation_dataset,
)
from unilab.algos.torch.distill.observability.performance import (
    DistillationStageObservationAccumulator,
)


@dataclass(frozen=True)
class StandardCollectionSpec:
    num_samples: int
    expected_student_obs_dim: int
    expected_teacher_obs_dim: int
    teacher_obs_key: str = "obs"
    teacher_projection: str = "identity"
    student_projection: str = "identity"
    student_drop_index: int | None = None
    action_mode: str = "zero"
    action_seed: int | None = None
    command_sample_filter: str = "none"
    command_info_key: str = "commands"
    target_height_info_key: str | None = None
    command_xy_threshold: float = 0.05
    command_yaw_threshold: float = 0.05
    max_env_steps: int | None = None
    role_label: str | None = None
    metadata: Mapping[str, Any] | None = None


class StandardCollectionTransaction:
    """Own one validation/reset/collect/finalize transaction.

    Environment and policies are borrowed resources. Mutable buffers and
    counters belong exclusively to this transaction.
    """

    def __init__(
        self,
        env: Any,
        spec: StandardCollectionSpec,
        *,
        teacher_policy: torch.nn.Module | None,
        rollout_policy: torch.nn.Module | None,
        initial_reset: tuple[Any, Any] | None,
        performance_clock: Callable[[], float] | None,
    ) -> None:
        self.env = env
        self.spec = spec
        self.teacher_policy = teacher_policy
        self.rollout_policy = rollout_policy
        self.initial_reset = initial_reset
        self.performance = (
            None
            if performance_clock is None
            else DistillationStageObservationAccumulator(clock=performance_clock)
        )
        self.student_chunks: list[torch.Tensor] = []
        self.teacher_chunks: list[torch.Tensor] = []
        self.teacher_action_chunks: list[torch.Tensor] = []
        self.command_chunks: list[torch.Tensor] = []
        self.target_height_chunks: list[torch.Tensor] = []
        self.command_intent_chunks: list[str] = []
        self.env_steps = 0
        self.collected_count = 0
        self.command_seen_samples = 0
        self.command_selected_samples = 0
        self.synthetic_teacher_tail = False
        self.done_seen_samples = 0
        self.autoreset_done_count = 0
        self.manual_done_reset_count = 0
        self.teacher_inference_rows = 0
        self.student_inference_rows = 0
        self.action_abs_max = 0.0

    def run(self) -> DistillationTensorDataset:
        self._prepare()
        while self.collected_count < self.spec.num_samples:
            actions = self._collect_current_rows()
            if self.collected_count >= self.spec.num_samples:
                break
            self._step_environment(actions)
        return self._finalize()

    def _prepare(self) -> None:
        spec = self.spec
        if spec.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {spec.num_samples}")
        if spec.command_sample_filter not in {"none", "active", "inactive"}:
            raise ValueError(
                f"Unsupported command_sample_filter: {spec.command_sample_filter!r}"
            )
        if spec.action_mode not in {"zero", "random", "teacher_policy", "student_policy"}:
            raise ValueError(f"Unsupported collect action_mode: {spec.action_mode!r}")
        policy_mode = spec.action_mode in {"teacher_policy", "student_policy"}
        if policy_mode and self.teacher_policy is None:
            raise ValueError(
                f"teacher_policy is required when action_mode={spec.action_mode!r} "
                "to cache teacher target actions"
            )
        if not policy_mode and self.teacher_policy is not None:
            raise ValueError(
                "teacher_policy can only be set when action_mode='teacher_policy' "
                "or action_mode='student_policy'"
            )
        if spec.action_mode == "student_policy" and self.rollout_policy is None:
            raise ValueError("rollout_policy is required when action_mode='student_policy'")
        if spec.action_mode != "student_policy" and self.rollout_policy is not None:
            raise ValueError("rollout_policy can only be set when action_mode='student_policy'")
        if policy_mode and spec.action_seed is not None:
            raise ValueError("action_seed is only supported when action_mode='random'")
        action_shape = getattr(getattr(self.env, "action_space", None), "shape", None)
        if action_shape is None:
            raise ValueError("env.action_space.shape must be defined for distillation collection")
        if spec.max_env_steps is not None and spec.max_env_steps < 0:
            raise ValueError(
                f"max_env_steps must be non-negative when set, got {spec.max_env_steps}"
            )
        self.num_envs = int(getattr(self.env, "num_envs"))
        self.action_dim = int(action_shape[0])
        self.effective_max_env_steps = (
            spec.max_env_steps
            if spec.max_env_steps is not None
            else (
                max(int(np.ceil(spec.num_samples / max(self.num_envs, 1))) * 100, 1)
                if spec.command_sample_filter != "none"
                else None
            )
        )
        self.rng = np.random.default_rng(spec.action_seed)
        if getattr(self.env, "state", None) is None and callable(
            getattr(self.env, "init_state", None)
        ):
            self.env.init_state()
        self.obs, self.current_info = _resolve_collection_reset(
            self.env,
            num_envs=self.num_envs,
            initial_reset=self.initial_reset,
        )

    def _collect_current_rows(self) -> np.ndarray | None:
        spec = self.spec
        source_np = _obs_array(self.obs, spec.teacher_obs_key)
        teacher_np, self.synthetic_teacher_tail = project_teacher_obs(
            source_np,
            projection=spec.teacher_projection,
            expected_teacher_obs_dim=spec.expected_teacher_obs_dim,
        )
        student_np = project_student_obs(
            source_np,
            projection=spec.student_projection,
            expected_student_obs_dim=spec.expected_student_obs_dim,
            student_drop_index=spec.student_drop_index,
        )
        commands_np = (
            _info_array(
                self.current_info,
                spec.command_info_key,
                expected_rows=teacher_np.shape[0],
            )
            if spec.command_sample_filter != "none"
            else None
        )
        target_height_np = (
            None
            if spec.target_height_info_key in (None, "")
            else _target_height_array(
                self.current_info,
                str(spec.target_height_info_key),
                expected_rows=teacher_np.shape[0],
            )
        )
        command_active = (
            command_active_mask(
                commands_np,
                xy_threshold=spec.command_xy_threshold,
                yaw_threshold=spec.command_yaw_threshold,
            )
            if commands_np is not None
            else None
        )
        row_mask = _command_sample_mask(
            self.current_info,
            sample_filter=spec.command_sample_filter,
            command_info_key=spec.command_info_key,
            expected_rows=teacher_np.shape[0],
            xy_threshold=spec.command_xy_threshold,
            yaw_threshold=spec.command_yaw_threshold,
        )
        if row_mask.shape[0] != teacher_np.shape[0]:
            raise ValueError(
                "command sample mask row mismatch: "
                f"expected {teacher_np.shape[0]}, got {row_mask.shape[0]}"
            )
        if spec.command_sample_filter != "none":
            self.command_seen_samples += int(row_mask.shape[0])
            self.command_selected_samples += int(np.count_nonzero(row_mask))
        label_actions = self._teacher_actions(teacher_np)
        actions = self._rollout_actions(student_np, label_actions)
        self._admit_rows(
            teacher_np=teacher_np,
            student_np=student_np,
            label_actions=label_actions,
            commands_np=commands_np,
            target_height_np=target_height_np,
            command_active=command_active,
            row_mask=row_mask,
        )
        if actions is not None:
            self.action_abs_max = max(self.action_abs_max, float(np.max(np.abs(actions))))
        return actions

    def _teacher_actions(self, teacher_np: np.ndarray) -> np.ndarray | None:
        if self.teacher_policy is None:
            return None
        with _performance_span(self.performance, "teacher_inference"):
            actions = _policy_actions(
                self.teacher_policy,
                teacher_np,
                action_dim=self.action_dim,
                policy_name="teacher_policy",
            )
        self.teacher_inference_rows += int(teacher_np.shape[0])
        if not np.all(np.isfinite(actions)):
            raise ValueError("teacher_policy produced non-finite target actions")
        return actions

    def _rollout_actions(
        self, student_np: np.ndarray, label_actions: np.ndarray | None
    ) -> np.ndarray | None:
        if self.spec.action_mode == "teacher_policy":
            return label_actions
        if self.spec.action_mode != "student_policy":
            return None
        if self.rollout_policy is None:
            raise RuntimeError("student rollout policy contract was not materialized")
        with _performance_span(self.performance, "student_inference"):
            actions = _policy_actions(
                self.rollout_policy,
                student_np,
                action_dim=self.action_dim,
                policy_name="rollout_policy",
            )
        self.student_inference_rows += int(student_np.shape[0])
        if not np.all(np.isfinite(actions)):
            raise ValueError("rollout_policy produced non-finite rollout actions")
        return actions

    def _admit_rows(
        self,
        *,
        teacher_np: np.ndarray,
        student_np: np.ndarray,
        label_actions: np.ndarray | None,
        commands_np: np.ndarray | None,
        target_height_np: np.ndarray | None,
        command_active: np.ndarray | None,
        row_mask: np.ndarray,
    ) -> None:
        with _performance_span(self.performance, "tensor_pack"):
            selected_teacher = teacher_np[row_mask]
            selected_student = student_np[row_mask]
            selected_actions = label_actions[row_mask] if label_actions is not None else None
            selected_commands = commands_np[row_mask] if commands_np is not None else None
            selected_height = target_height_np[row_mask] if target_height_np is not None else None
            selected_active = command_active[row_mask] if command_active is not None else None
            take = min(self.spec.num_samples - self.collected_count, selected_teacher.shape[0])
            if take <= 0:
                return
            self.teacher_chunks.append(torch.as_tensor(selected_teacher[:take], dtype=torch.float32))
            self.student_chunks.append(torch.as_tensor(selected_student[:take], dtype=torch.float32))
            if selected_actions is not None:
                self.teacher_action_chunks.append(
                    torch.as_tensor(selected_actions[:take], dtype=torch.float32)
                )
            if selected_commands is not None and selected_active is not None:
                self.command_chunks.append(
                    torch.as_tensor(selected_commands[:take], dtype=torch.float32)
                )
                self.command_intent_chunks.extend(
                    "active" if bool(value) else "inactive" for value in selected_active[:take]
                )
            if selected_height is not None:
                self.target_height_chunks.append(
                    torch.as_tensor(selected_height[:take], dtype=torch.float32)
                )
            self.collected_count += int(take)

    def _step_environment(self, actions: np.ndarray | None) -> None:
        spec = self.spec
        if self.effective_max_env_steps is not None and self.env_steps >= self.effective_max_env_steps:
            raise RuntimeError(
                f"command_sample_filter={spec.command_sample_filter!r} selected "
                f"{self.command_selected_samples}/{self.command_seen_samples} samples after "
                f"{self.env_steps} env steps; increase max_env_steps or relax command thresholds"
            )
        if spec.action_mode == "zero":
            actions = np.zeros((self.num_envs, self.action_dim), dtype=np.float32)
        elif spec.action_mode == "random":
            actions = self.rng.uniform(-1.0, 1.0, size=(self.num_envs, self.action_dim)).astype(
                np.float32
            )
        if actions is None:
            raise RuntimeError(f"collect action_mode={spec.action_mode!r} did not materialize actions")
        if not np.all(np.isfinite(actions)):
            raise ValueError(f"collect action_mode={spec.action_mode!r} produced non-finite actions")
        self.action_abs_max = max(self.action_abs_max, float(np.max(np.abs(actions))))
        with _performance_span(self.performance, "env_step"):
            state = self.env.step(actions)
        self.env_steps += 1
        self.obs, self.current_info, done_count, autoreset_count, manual_reset_count = (
            _reset_done_rows_after_step(self.env, state, num_envs=self.num_envs)
        )
        self.done_seen_samples += done_count
        self.autoreset_done_count += autoreset_count
        self.manual_done_reset_count += manual_reset_count

    def _finalize(self) -> DistillationTensorDataset:
        spec = self.spec
        payload = dict(spec.metadata or {})
        role_label = None if spec.role_label in (None, "") else str(spec.role_label)
        if role_label is not None:
            payload["role_label"] = role_label
        payload.update(
            {
                "source": "live_env_rollout",
                "teacher_obs_key": spec.teacher_obs_key,
                "teacher_projection": spec.teacher_projection,
                "student_projection": spec.student_projection,
                "student_drop_index": spec.student_drop_index,
                "target_height_info_key": (
                    None if spec.target_height_info_key in (None, "") else str(spec.target_height_info_key)
                ),
                "action_mode": spec.action_mode,
                "action_seed": spec.action_seed,
                "action_abs_max": float(self.action_abs_max),
                "num_envs": self.num_envs,
                "env_steps": self.env_steps,
                "done_seen_samples": int(self.done_seen_samples),
                "autoreset_done_count": int(self.autoreset_done_count),
                "manual_done_reset_count": int(self.manual_done_reset_count),
                "synthetic_teacher_tail": bool(self.synthetic_teacher_tail),
            }
        )
        if spec.command_sample_filter != "none":
            payload.update(
                {
                    "command_sample_filter": spec.command_sample_filter,
                    "command_info_key": spec.command_info_key,
                    "command_xy_threshold": spec.command_xy_threshold,
                    "command_yaw_threshold": spec.command_yaw_threshold,
                    "command_seen_samples": int(self.command_seen_samples),
                    "command_selected_samples": int(self.command_selected_samples),
                    "max_env_steps": self.effective_max_env_steps,
                }
            )
        if spec.action_mode == "student_policy":
            payload["rollout_policy"] = "distillation_student"
        with _performance_span(self.performance, "tensor_pack"):
            dataset = build_distillation_dataset(
                torch.cat(self.student_chunks, dim=0),
                torch.cat(self.teacher_chunks, dim=0),
                expected_student_obs_dim=spec.expected_student_obs_dim,
                expected_teacher_obs_dim=spec.expected_teacher_obs_dim,
                expected_teacher_action_dim=self.action_dim,
                metadata=payload,
                teacher_actions=(
                    torch.cat(self.teacher_action_chunks, dim=0)
                    if self.teacher_action_chunks
                    else None
                ),
                commands=torch.cat(self.command_chunks, dim=0) if self.command_chunks else None,
                target_height=(
                    torch.cat(self.target_height_chunks, dim=0)
                    if self.target_height_chunks
                    else None
                ),
                command_intents=(tuple(self.command_intent_chunks) if self.command_intent_chunks else None),
                role_labels=(role_label,) * spec.num_samples if role_label is not None else None,
            )
        return _attach_collector_performance(
            dataset,
            accumulator=self.performance,
            teacher_inference_rows=self.teacher_inference_rows,
            student_inference_rows=self.student_inference_rows,
            env_steps=self.env_steps,
        )
