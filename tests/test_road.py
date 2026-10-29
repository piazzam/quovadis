import numpy as np
import pytest

from quovadis_repro.metrics import _selected_mask


def test_road_morf_selects_highest_scores():
    heatmap = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = _selected_mask(heatmap, 0.5, "morf")
    expected = np.array([[False, False], [True, True]])
    np.testing.assert_array_equal(mask, expected)


def test_road_lerf_selects_lowest_scores():
    heatmap = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = _selected_mask(heatmap, 0.5, "lerf")
    expected = np.array([[True, True], [False, False]])
    np.testing.assert_array_equal(mask, expected)


def test_road_rejects_unknown_direction():
    with pytest.raises(ValueError):
        _selected_mask(np.ones((2, 2)), 0.5, "unknown")
