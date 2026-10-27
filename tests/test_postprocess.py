"""Tests for the post-processing module."""
import numpy as np
import pytest


from radwet_l.postprocess import combine_classifications, apply_mask


def test_combine_water_only():
    """Water=2 with no flood should stay 2 in forest areas."""
    water = np.array([[2, 0]], dtype=np.uint8)
    flood = np.array([[0, 0]], dtype=np.uint8)
    lcc = np.array([[0, 0]], dtype=np.uint8)  # 0=forest

    result = combine_classifications(water, flood, lcc)
    assert result[0, 0] == 2  # water
    assert result[0, 1] == 4  # neither -> other land


def test_combine_flood_only():
    """Flood=1 with no water should stay 1 in forest areas."""
    water = np.array([[0, 0]], dtype=np.uint8)
    flood = np.array([[1, 0]], dtype=np.uint8)
    lcc = np.array([[0, 0]], dtype=np.uint8)

    result = combine_classifications(water, flood, lcc)
    assert result[0, 0] == 1  # flood forest
    assert result[0, 1] == 4  # other


def test_combine_overlap_becomes_other():
    """Both water=2 and flood=1 -> 4 (other, ambiguous)."""
    water = np.array([[2]], dtype=np.uint8)
    flood = np.array([[1]], dtype=np.uint8)
    lcc = np.array([[0]], dtype=np.uint8)

    result = combine_classifications(water, flood, lcc)
    assert result[0, 0] == 4


def test_combine_non_forest_mask():
    """Non-forest pixels (lcc=1) should get class 0 (background).

    Class 5 (non-forest) is applied later by mask_image_with_class after sieve filtering,
    matching the original pipeline (applyXGBoostClassificationImgs_Bins.py:795-803).
    """
    water = np.array([[2, 0]], dtype=np.uint8)
    flood = np.array([[0, 1]], dtype=np.uint8)
    lcc = np.array([[1, 1]], dtype=np.uint8)  # non-forest

    result = combine_classifications(water, flood, lcc)
    assert result[0, 0] == 0
    assert result[0, 1] == 0


def test_combine_full_scenario():
    """Full test matching original logic."""
    water = np.array([[2, 0, 2, 0], [0, 0, 0, 0]], dtype=np.uint8)
    flood = np.array([[0, 1, 1, 0], [0, 0, 1, 0]], dtype=np.uint8)
    lcc = np.array([[0, 0, 0, 0], [0, 1, 0, 1]], dtype=np.uint8)

    result = combine_classifications(water, flood, lcc)
    assert result[0, 0] == 2  # water
    assert result[0, 1] == 1  # flood
    assert result[0, 2] == 4  # both -> other
    assert result[0, 3] == 4  # neither in forest -> other
    assert result[1, 0] == 4  # neither in forest -> other
    assert result[1, 1] == 0  # non-forest -> 0 (class 5 applied after sieve)
    assert result[1, 2] == 1  # flood in forest
    assert result[1, 3] == 0  # non-forest -> 0 (class 5 applied after sieve)


def test_apply_mask():
    """Masked pixels should get the mask_value."""
    data = np.array([[1, 2, 4], [1, 2, 4]], dtype=np.uint8)
    mask = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.uint8)

    result = apply_mask(data, mask, mask_value=6)
    assert result[0, 2] == 6
    assert result[1, 1] == 6
    assert result[0, 0] == 1  # unchanged
    assert result[0, 1] == 2  # unchanged
