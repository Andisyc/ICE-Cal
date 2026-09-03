from __future__ import annotations

import importlib
import inspect
import math

import numpy as np
import pytest


def _owner():
    try:
        return importlib.import_module("unilab.algos.torch.distill.fada.path_deviation")
    except ModuleNotFoundError:
        pytest.fail("Stage C straight-line path-deviation owner is missing")


def test_report_compares_each_branch_with_the_same_straight_line() -> None:
    owner = _owner()
    report = owner.build_straight_line_deviation_report(
        nominal_xy_m=np.asarray([[0.0, 0.0], [1.0, 0.1], [2.0, -0.1]]),
        faulty_xy_m=np.asarray([[0.0, 0.0], [1.0, 0.4], [2.0, 0.6]]),
        nominal_yaw_rad=np.zeros(3),
        faulty_yaw_rad=np.zeros(3),
        origin_xy_m=np.asarray([0.0, 0.0]),
        heading_rad=0.0,
    )

    assert report["reference_line"]["num_samples"] == 3
    assert report["nominal"]["lateral_m"] == pytest.approx([0.0, 0.1, -0.1])
    assert report["nominal"]["rms_lateral_m"] == pytest.approx(math.sqrt(0.02 / 3.0))
    assert report["nominal"]["mean_abs_lateral_m"] == pytest.approx(0.2 / 3.0)
    assert report["nominal"]["max_abs_lateral_m"] == pytest.approx(0.1)
    assert report["nominal"]["final_lateral_m"] == pytest.approx(-0.1)
    assert report["faulty"]["rms_lateral_m"] == pytest.approx(math.sqrt(0.52 / 3.0))
    assert report["faulty"]["mean_abs_lateral_m"] == pytest.approx(1.0 / 3.0)
    assert report["faulty"]["max_abs_lateral_m"] == pytest.approx(0.6)
    assert report["faulty"]["final_lateral_m"] == pytest.approx(0.6)
    assert report["excess"]["max_abs_lateral_m"] == pytest.approx(0.5)
    assert report["excess"]["final_abs_lateral_m"] == pytest.approx(0.5)


def test_report_uses_initial_heading_instead_of_world_x_axis() -> None:
    owner = _owner()
    report = owner.build_straight_line_deviation_report(
        nominal_xy_m=np.asarray([[3.0, 4.0], [3.0, 5.0]]),
        faulty_xy_m=np.asarray([[3.0, 4.0], [3.2, 5.0]]),
        nominal_yaw_rad=np.full(2, math.pi / 2.0),
        faulty_yaw_rad=np.full(2, math.pi / 2.0),
        origin_xy_m=np.asarray([3.0, 4.0]),
        heading_rad=math.pi / 2.0,
    )

    assert report["nominal"]["max_abs_lateral_m"] == pytest.approx(0.0, abs=1e-12)
    assert report["faulty"]["final_lateral_m"] == pytest.approx(-0.2)


def test_report_records_wrapped_per_frame_yaw_drift() -> None:
    owner = _owner()
    parameters = inspect.signature(owner.build_straight_line_deviation_report).parameters
    assert {"nominal_yaw_rad", "faulty_yaw_rad"} <= set(parameters)

    heading = math.pi - 0.1
    report = owner.build_straight_line_deviation_report(
        nominal_xy_m=np.zeros((2, 2)),
        faulty_xy_m=np.zeros((2, 2)),
        nominal_yaw_rad=np.asarray([heading, -math.pi + 0.1]),
        faulty_yaw_rad=np.asarray([heading, -math.pi + 0.3]),
        origin_xy_m=np.zeros(2),
        heading_rad=heading,
    )

    assert report["schema_version"] == "fada-path-deviation/v2"
    assert report["nominal"]["yaw_rad"] == pytest.approx([heading, -math.pi + 0.1])
    assert report["nominal"]["yaw_drift_rad"] == pytest.approx([0.0, 0.2])
    assert report["faulty"]["yaw_drift_rad"] == pytest.approx([0.0, 0.4])
    assert report["nominal"]["rms_yaw_drift_rad"] == pytest.approx(math.sqrt(0.02))
    assert report["faulty"]["mean_abs_yaw_drift_rad"] == pytest.approx(0.2)
    assert report["excess"]["max_abs_yaw_drift_rad"] == pytest.approx(0.2)


def test_report_rejects_position_yaw_frame_mismatch() -> None:
    owner = _owner()
    with pytest.raises(ValueError, match="nominal position/yaw frame count mismatch"):
        owner.build_straight_line_deviation_report(
            nominal_xy_m=np.zeros((2, 2)),
            faulty_xy_m=np.zeros((2, 2)),
            nominal_yaw_rad=np.zeros(1),
            faulty_yaw_rad=np.zeros(2),
            origin_xy_m=np.zeros(2),
            heading_rad=0.0,
        )


@pytest.mark.parametrize(
    "nominal,faulty",
    [
        (np.zeros((0, 2)), np.zeros((1, 2))),
        (np.zeros((1, 3)), np.zeros((1, 2))),
        (np.asarray([[0.0, np.nan]]), np.zeros((1, 2))),
    ],
)
def test_report_rejects_unusable_position_traces(
    nominal: np.ndarray, faulty: np.ndarray
) -> None:
    owner = _owner()
    with pytest.raises(ValueError, match="position trace"):
        owner.build_straight_line_deviation_report(
            nominal_xy_m=nominal,
            faulty_xy_m=faulty,
            nominal_yaw_rad=np.zeros(max(len(nominal), 1)),
            faulty_yaw_rad=np.zeros(max(len(faulty), 1)),
            origin_xy_m=np.asarray([0.0, 0.0]),
            heading_rad=0.0,
        )
