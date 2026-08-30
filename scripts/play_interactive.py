"""MuJoCo-only interactive playback composition root.

Use ``--task`` and ``--sim`` to select the task owner configuration. Policy,
checkpoint, viewer, overlay, control, and trace behavior live in production
owners under :mod:`unilab.visualization`.
"""

# pyright: reportAttributeAccessIssue=false, reportArgumentType=false

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import mujoco
import mujoco.viewer

from unilab.base import registry
from unilab.training import ensure_registries, resolve_task_checkpoint_path
from unilab.training.rsl_rl import RslRlVecEnvWrapper

ensure_registries()

from unilab.visualization.interactive_playback import (  # noqa: E402
    _apply_distill_playback_reset_contract,
)
from unilab.visualization.playback_checkpoint_contract import (
    _apply_missing_g1_height_command_contract,
    _apply_missing_g1_mode_observation_contract,
    _checkpoint_actor_input_dim,
    _g1_standing_contract_issues,
    _infer_checkpoint_actor_input_dim,
    _load_checkpoint_run_config,
    _nested_get,
    _resolve_play_checkpoint_path,
    _warn_if_g1_sac_checkpoint_lacks_standing_contract,
)
from unilab.visualization.playback_checkpoint_contract import (  # noqa: E402
    apply_checkpoint_env_contract as _apply_checkpoint_env_contract,
)
from unilab.visualization.playback_checkpoint_contract import (
    resolve_checkpoint as _resolve_checkpoint_owner,
)
from unilab.visualization.playback_cli import (  # noqa: E402
    SUPPORTED_INTERACTIVE_ALGOS,
    InteractiveCliArgs,
    PlayInteractiveArgs,
    _algo_config_dict,
    _build_play_args,
    _compose_interactive_config,
    _interactive_overrides_from_cli,
    _normalize_checkpoint_value,
    _normalize_interactive_overrides,
    _override_key,
    _parse_interactive_cli,
)
from unilab.visualization.playback_controls import (  # noqa: E402
    _COMMAND_OBS_VERIFY_COMMAND,
    _KEY_ENTER,
    _KEY_LEFT,
    _KEY_RIGHT,
    _KEY_UP,
    _apply_playback_command,
    _build_height_commander,
    _build_keyboard_commander,
    _build_playback_config,
    _handle_command_key,
    _handle_height_key,
    _policy_obs_contains_command,
    _should_render_velocity_arrows,
)
from unilab.visualization.playback_sessions import KeyboardCommander  # noqa: E402
from unilab.visualization.playback_viewer import (  # noqa: E402
    _backend_adapter,
    _build_interactive_env_factory,
    _InteractiveViewerRuntime,
    _load_resolved_visual_viewer_model,
    _load_viewer_model,
    _prepare_viewer_runtime,
    _render_interactive_frame,
    _run_interactive_viewer_loop,
    _select_playback_device,
    play_interactive,
)
from unilab.visualization.playback_viewer import (
    create_interactive_session as _create_interactive_session,
)


def resolve_checkpoint(
    task: str,
    load_run: str,
    checkpoint: str | None = None,
    algo_log_name: str = "rsl_rl_ppo",
    log_root: str | None = None,
) -> str | None:
    """Preserve the script-level resolver injection seam."""

    return _resolve_checkpoint_owner(
        task,
        load_run,
        checkpoint,
        algo_log_name,
        log_root,
        root_dir=ROOT_DIR,
        resolver=resolve_task_checkpoint_path,
    )


def main() -> None:
    cli_args = _parse_interactive_cli(sys.argv[1:])
    cfg = _compose_interactive_config(cli_args.algo, cli_args.overrides)
    args = _build_play_args(cfg, algo=cli_args.algo)
    play_interactive(args, cfg=cfg, algo=cli_args.algo)


if __name__ == "__main__":
    main()
