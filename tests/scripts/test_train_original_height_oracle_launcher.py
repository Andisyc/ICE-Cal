from pathlib import Path


def test_launcher_uses_no_gait_baseline_task():
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "train_original_height_oracle.sh"
    ).read_text(encoding="utf-8")
    active_script = "\n".join(
        line for line in script.splitlines() if not line.lstrip().startswith("#")
    )

    assert (
        "task=sac/g1_walk_flat/mujoco_fada_privileged_oracle_grouped_dr_lineage"
        in active_script
    )
    assert "mujoco_fada_privileged_oracle_original_height_grouped_dr_lineage" not in active_script
    assert "FADAPrivilegedOracle_baseline_" in active_script
    assert 'ICE_CAL_ORACLE_LINEAGE_ID="baseline-${run_stamp}"' in active_script
    assert "original_height" not in active_script
