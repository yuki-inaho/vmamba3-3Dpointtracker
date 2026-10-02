import numpy as np
import pytest

from mamba3_tracker.viz.comparison import (
    masked_path,
    gt_detail_cube,
    project_xyz,
    select_tracks,
    set_point_3d,
    shared_cube,
)


def test_project_uses_original_intrinsics_without_scale_alignment():
    k = np.array([[100, 0, 50], [0, 200, 60], [0, 0, 1]])
    xyz = np.array([[1, 2, 10], [-2, -1, 2]], dtype=float)
    np.testing.assert_allclose(project_xyz(xyz, k), [[60, 100], [-50, -40]])


def test_project_rejects_nonpositive_depth_and_nonfinite_points():
    xyz = np.array([[0, 0, 0], [1, 2, -1], [np.nan, 1, 1], [0, 0, 1]])
    uv = project_xyz(xyz, np.eye(3))
    assert np.isnan(uv[:3]).all()
    np.testing.assert_equal(uv[-1], [0, 0])


def test_selection_depends_only_on_visibility_with_stable_id_ties():
    visibility = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]])
    np.testing.assert_array_equal(select_tracks(visibility, 3), [3, 0, 1])


def test_selection_rejects_empty_and_bad_limits():
    with pytest.raises(ValueError):
        select_tracks(np.zeros((0, 4)), 32)
    with pytest.raises(ValueError):
        select_tracks(np.ones((3, 4)), 0)


def test_shared_cube_contains_all_methods_with_equal_metres():
    arrays = [np.array([[[-2, 0, 3], [0, 1, 4]]]), np.array([[[8, 2, 20]]])]
    limits = shared_cube(arrays)
    assert limits.shape == (3, 2)
    np.testing.assert_allclose(
        np.diff(limits, axis=1), np.full((3, 1), limits[0, 1] - limits[0, 0])
    )
    for xyz in arrays:
        assert np.all(xyz >= limits[:, 0])
        assert np.all(xyz <= limits[:, 1])


def test_masked_path_does_not_join_over_occlusion():
    xyz = np.arange(15, dtype=float).reshape(5, 3)
    result = masked_path(xyz, np.array([1, 1, 0, 1, 1]), 4, 5)
    assert np.isnan(result[2]).all()
    np.testing.assert_equal(result[[0, 1, 3, 4]], xyz[[0, 1, 3, 4]])


def test_masked_path_uses_only_present_and_past_frames():
    xyz = np.arange(15, dtype=float).reshape(5, 3)
    result = masked_path(xyz, np.ones(5), 2, 2)
    np.testing.assert_equal(result, xyz[1:3])


def test_occluded_3d_point_can_be_drawn_without_list_shape_error():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axes = figure.add_subplot(projection="3d")
    (marker,) = axes.plot([], [], [], "o")
    set_point_3d(marker, np.full(3, np.nan))
    figure.canvas.draw()
    plt.close(figure)


def test_gt_detail_cube_ignores_invisible_outliers_and_is_equal_scale():
    gt = np.array([[[0, 0, 1], [1, 2, 3], [999, 999, 999]]], dtype=float)
    visibility = np.array([[1, 1, 0]])
    limits = gt_detail_cube(gt, visibility)
    np.testing.assert_allclose(limits.mean(1), [0.5, 1, 2])
    np.testing.assert_allclose(np.diff(limits, axis=1), np.full((3, 1), 1.76))
