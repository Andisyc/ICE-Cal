import os
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def _dry_run_start_sh(*args: str, env: dict[str, str] | None = None) -> str:
    child_env = os.environ.copy()
    child_env.pop("UNILAB_G1_ACTION_TRACE", None)
    child_env.pop("UNILAB_G1_ACTION_TRACE_INTERVAL", None)
    if env:
        child_env.update(env)
    result = subprocess.run(
        ["bash", "start.sh", "--dry-run", *args],
        cwd=ROOT_DIR,
        capture_output=True,
        check=True,
        env=child_env,
        text=True,
    )
    return result.stdout


def test_start_sh_default_routes_to_g1_walk_flat() -> None:
    output = _dry_run_start_sh("2026-06-12_15-46-01_mujoco")

    assert "[start.sh] task=g1_walk_flat sim=mujoco algo=sac keyboard=true" in output
    assert "--task g1_walk_flat" in output
    assert "interactive.keyboard=true" in output
    assert "algo.load_run=2026-06-12_15-46-01_mujoco" in output
    assert "+algo.checkpoint=model_5000.pt" in output
    assert "[start.sh] action_trace=" not in output


def test_start_sh_explicit_stand_still_routes_to_stand_task() -> None:
    output = _dry_run_start_sh("--task", "g1_stand_still", "2026-07-09_00-00-00_mujoco")

    assert "[start.sh] task=g1_stand_still sim=mujoco algo=sac keyboard=false" in output
    assert "--task g1_stand_still" in output
    assert "interactive.keyboard=false" in output
    assert "algo.load_run=2026-07-09_00-00-00_mujoco" in output
    assert "+algo.checkpoint=model_5000.pt" in output
    assert "[start.sh] action_trace=" not in output


def test_start_sh_stand_still_shorthand_routes_to_stand_task() -> None:
    output = _dry_run_start_sh("g1_stand_still", "2026-07-09_00-00-00_mujoco")

    assert "[start.sh] task=g1_stand_still sim=mujoco algo=sac keyboard=false" in output
    assert "--task g1_stand_still" in output
    assert "algo.load_run=2026-07-09_00-00-00_mujoco" in output
    assert "[start.sh] action_trace=" not in output


def test_start_sh_preserves_extra_hydra_overrides_after_shortcut() -> None:
    output = _dry_run_start_sh(
        "--task",
        "g1_stand_still",
        "2026-07-09_00-00-00_mujoco",
        "training.device=cpu",
        "interactive.keyboard=true",
    )

    assert "algo.load_run=2026-07-09_00-00-00_mujoco" in output
    assert "training.device=cpu" in output
    assert "interactive.keyboard=false" in output
    assert "interactive.keyboard=true" in output


def test_start_sh_respects_explicit_action_trace_env() -> None:
    output = _dry_run_start_sh(
        "g1_stand_still",
        "2026-07-09_00-00-00_mujoco",
        env={"UNILAB_G1_ACTION_TRACE": "0", "UNILAB_G1_ACTION_TRACE_INTERVAL": "5"},
    )

    assert "[start.sh] action_trace=0 interval=5" in output
