from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import torch

from unilab.algos.torch.distill.fada import FADAPlannerIDMPolicy
from unilab.algos.torch.distill.fada_playback import FADAPlaybackController
from unilab.algos.torch.fada_context.calibration import (
    CalibratedFADAPolicy,
    CalibrationReadout,
    CalibrationReadoutState,
    CoefficientEncoder,
    DirectionBank,
    MonotoneScaleCurve,
    load_calibration_artifact,
)


class CalibratedFADAPlaybackController(FADAPlaybackController):
    """Track real history availability before enabling calibration readout."""

    def __init__(
        self,
        policy: CalibratedFADAPolicy,
        *,
        device: str | torch.device,
        jump_threshold: torch.Tensor,
    ) -> None:
        super().__init__(policy, device=device)
        self.calibrated_policy = policy
        self.readout_state = CalibrationReadoutState(
            axis_count=policy.direction_bank.axis_count,
            jump_threshold=jump_threshold,
        )
        self._history_count: torch.Tensor | None = None
        self.last_readout: CalibrationReadout | None = None

    def reset(self, done: object | None = None) -> None:
        super().reset(done)
        if done is None:
            self._history_count = None
            self.readout_state.reset()
            self.last_readout = None
            return
        mask = torch.as_tensor(done, dtype=torch.bool, device=self.device).reshape(-1)
        if self._history_count is not None:
            if mask.shape != self._history_count.shape:
                raise ValueError("calibrated playback reset mask batch mismatch")
            self._history_count[mask] = 0
            self.readout_state.reset(mask)

    @torch.no_grad()
    def act(self, observation: object, command: object) -> torch.Tensor:
        obs = self._observation_tensor(observation)
        cmd = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        if cmd.ndim == 1:
            cmd = cmd.unsqueeze(0)
        expected_command_shape = (obs.shape[0], self.config.command_dim)
        if tuple(cmd.shape) != expected_command_shape:
            raise ValueError(
                "FADA playback command shape mismatch: "
                f"expected={expected_command_shape} observed={tuple(cmd.shape)}"
            )
        if not bool(torch.isfinite(cmd).all()):
            raise ValueError("FADA playback command must contain only finite values")

        pending_reset = None if self._pending_reset is None else self._pending_reset.clone()
        first_observation = self._observation_history is None
        self._advance_observation_history(obs)
        assert self._observation_history is not None
        assert self._action_history is not None
        if first_observation:
            self._history_count = torch.ones(obs.shape[0], dtype=torch.int64, device=self.device)
        else:
            assert self._history_count is not None
            self._history_count.clamp_max_(self.config.history_length - 1).add_(1)
            if pending_reset is not None:
                self._history_count[pending_reset] = 1
        assert self._history_count is not None
        output = self.calibrated_policy.forward_with_readout(
            self._observation_history,
            self._action_history,
            cmd,
            ready=self._history_count >= self.config.history_length,
            readout_state=self.readout_state,
        )
        self.last_readout = output.readout
        action = output.action.detach()
        expected_action_shape = (obs.shape[0], self.config.action_dim)
        if tuple(action.shape) != expected_action_shape:
            raise ValueError(
                "FADA playback action shape mismatch: "
                f"expected={expected_action_shape} observed={tuple(action.shape)}"
            )
        if not bool(torch.isfinite(action).all()):
            raise ValueError("FADA playback policy produced non-finite actions")
        self._action_history = torch.cat(
            (self._action_history[:, 1:], action.unsqueeze(1)),
            dim=1,
        )
        return action


def load_calibrated_policy(
    healthy_policy: FADAPlannerIDMPolicy,
    artifact_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_metadata: Mapping[str, str],
) -> CalibratedFADAPolicy:
    payload = load_calibration_artifact(artifact_path)
    artifact_metadata = payload["metadata"]
    if any(artifact_metadata.get(name) != value for name, value in expected_metadata.items()):
        raise ValueError("calibration artifact metadata identity mismatch")
    if payload["architecture"] != asdict(healthy_policy.config):
        raise ValueError("calibration artifact does not match the source Tracker architecture")
    direction_state = payload["direction_bank"]
    directions = direction_state.get("directions")
    expected_direction_shape = (
        len(payload["axis_names"]),
        healthy_policy.config.prediction_horizon,
        healthy_policy.config.hidden_dim,
    )
    if (
        not isinstance(directions, torch.Tensor)
        or tuple(directions.shape) != expected_direction_shape
    ):
        raise ValueError("calibration artifact direction bank is malformed")
    direction_bank = DirectionBank(
        axis_count=int(directions.shape[0]),
        prediction_horizon=int(directions.shape[1]),
        latent_dim=int(directions.shape[2]),
    )
    direction_bank.load_state_dict(direction_state, strict=True)
    encoder_config = payload.get("coefficient_encoder_config")
    encoder_state = payload.get("coefficient_encoder")
    if not isinstance(encoder_config, dict) or not isinstance(encoder_state, dict):
        raise ValueError("calibration artifact lacks Coefficient Encoder state")
    expected_encoder_config = {
        "state_dim": healthy_policy.config.obs_dim,
        "action_dim": healthy_policy.config.action_dim,
        "axis_count": len(payload["axis_names"]),
        "hidden_dim": 128,
        "layers": 2,
    }
    if encoder_config != expected_encoder_config:
        raise ValueError("calibration artifact Coefficient Encoder architecture mismatch")
    encoder = CoefficientEncoder(**encoder_config)
    encoder.load_state_dict(encoder_state, strict=True)
    curves = []
    for raw in payload["scale_curves"]:
        if not isinstance(raw, dict):
            raise ValueError("calibration artifact scale curve is malformed")
        curves.append(
            MonotoneScaleCurve(
                x=raw["x"],
                y=raw["y"],
                slopes=raw["slopes"],
                kind=raw.get("kind", "pchip"),
            )
        )
    policy = CalibratedFADAPolicy(
        healthy_policy.config,
        direction_bank=direction_bank.to(device),
        coefficient_encoder=encoder.to(device),
        scale_curves=tuple(curves),
        planner=healthy_policy.planner.to(device),
        idm=healthy_policy.idm.to(device),
    )
    policy.eval()
    return policy
