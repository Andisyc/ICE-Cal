"""Framework bindings extracted from :mod:`joystick` by responsibility.

The concrete environment remains the sole owner of mutable state.
"""

from __future__ import annotations

import numpy as np

from unilab.base.augmentation import SymmetryObsLayout
from unilab.dtype_config import get_global_dtype
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
from unilab.envs.locomotion.g1.walk_observation import (
    assemble_walk_observation,
    build_obs_groups_spec,
)


class G1WalkObservationBindings:
    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return build_obs_groups_spec(
            mode_observation=bool(self._cfg.mode_observation),
            height_observation=self._uses_height_command_observation(),
            privileged_strength=self._includes_privileged_actuator_strength_obs(),
            fada_privileged=self._fada_privileged_enabled(),
            fada_body_count=len(getattr(self, "_fada_body_names", ())),
        )

    def get_fada_privileged_checkpoint_identity(
        self,
    ) -> G1FADAPrivilegedCheckpointLayoutIdentity:
        """Return the immutable checkpoint layout sealed during environment initialization."""

        identity = self._fada_checkpoint_layout_identity
        if identity is None:
            raise ValueError("FADA privileged checkpoint identity is unavailable")
        return identity

    def _compute_obs(
        self,
        info: dict,
        linvel,
        gyro,
        gravity,
        dof_pos,
        dof_vel,
        *,
        row_ids: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        noise_cfg = self._cfg.noise_config
        diff = dof_pos - self.default_angles
        command = info["commands"]
        command_obs = self._command_observation(info, command.shape[0])
        last_actions = info.get("current_actions", np.zeros_like(diff))
        gait_phase = self._gait_phase_for_observation(info)
        mode_obs = self._mode_observation(info)
        walk_profile = self._uses_walk_observation_profile()
        dtype = np.dtype(get_global_dtype())

        privileged_strength = None
        if self._includes_privileged_actuator_strength_obs():
            privileged_strength = np.asarray(
                info.get("privileged_actuator_strength"),
                dtype=dtype,
            )
            expected_shape = (diff.shape[0], self._num_action)
            if privileged_strength.shape != expected_shape:
                raise ValueError(
                    "critic actuator-strength observation requires "
                    f"info['privileged_actuator_strength'] shape {expected_shape}, "
                    f"got {privileged_strength.shape}"
                )
            if not np.isfinite(privileged_strength).all():
                raise ValueError("critic actuator-strength observation must be finite")

        fada_privileged_obs = (
            self._materialize_fada_privileged_observation(
                info,
                linvel,
                row_ids=row_ids,
            )
            if self._fada_privileged_enabled()
            else None
        )
        return assemble_walk_observation(
            noisy_gyro=self._obs_noise(gyro, noise_cfg.scale_gyro),
            noisy_gravity=self._obs_noise(gravity, noise_cfg.scale_gravity),
            noisy_diff=self._obs_noise(diff, noise_cfg.scale_joint_angle),
            noisy_dof_vel=self._obs_noise(dof_vel, noise_cfg.scale_joint_vel),
            gyro=gyro,
            gravity=gravity,
            diff=diff,
            dof_vel=dof_vel,
            last_actions=last_actions,
            command_obs=command_obs,
            gait_phase=gait_phase,
            mode_obs=mode_obs,
            linvel=linvel,
            mode_observation=bool(self._cfg.mode_observation),
            walk_profile=walk_profile,
            fada_privileged=self._fada_privileged_enabled(),
            privileged_strength=privileged_strength,
            fada_privileged_obs=fada_privileged_obs,
            dtype=dtype,
        )

    def _fada_privileged_enabled(self) -> bool:
        return bool(
            getattr(getattr(self._cfg, "fada_privileged_observation", None), "enabled", False)
        )

    def _materialize_fada_privileged_observation(
        self,
        info: dict,
        linvel: np.ndarray,
        *,
        row_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        rows = int(np.asarray(linvel).shape[0])
        if self._fada_tau_max is None:
            raise ValueError("FADA privileged observation requires cached actuator force limits")
        row_indices = None
        if row_ids is not None:
            row_indices = np.asarray(row_ids, dtype=np.int64).reshape(-1)
            if row_indices.size != rows:
                raise ValueError(
                    "FADA privileged observation row_ids must match the observation rows"
                )

        def select_backend_rows(values: np.ndarray) -> np.ndarray:
            array = np.asarray(values)
            return array if row_indices is None else array[row_indices]

        return pack_fada_runtime_observation(
            body_names=self._fada_body_names,
            tau_max=self._fada_tau_max,
            linvel=linvel,
            left_contact_sensor=select_backend_rows(
                self._backend.get_sensor_data("left_foot_net_contact")
            ),
            right_contact_sensor=select_backend_rows(
                self._backend.get_sensor_data("right_foot_net_contact")
            ),
            root_clearance=select_backend_rows(self._terrain_relative_base_height()),
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

    def _gait_phase_for_observation(self, info: dict) -> np.ndarray:
        gait_phase = np.asarray(
            info.get("gait_phase", np.zeros((self._num_envs, 2), dtype=get_global_dtype())),
            dtype=get_global_dtype(),
        )
        if not bool(getattr(self._cfg, "gait_phase_enabled", True)):
            return np.zeros_like(gait_phase, dtype=get_global_dtype())
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
