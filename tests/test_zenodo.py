"""Tests for the Zenodo comparison helpers."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from radwet_l.zenodo import class_percentages, compare, find_product


def write_raster(path, array):
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype="uint8", crs="EPSG:4326",
        transform=from_origin(-74.0, -4.0, 0.001, 0.001),
    ) as ds:
        ds.write(array, 1)


def test_find_product_matches_cycle_number_exactly(tmp_path):
    for name in (
        "Classified_Output_Orbit-Cycle_22-Dated-x_COG.tif",
        "Classified_Output_Orbit-Cycle_226-Dated-06-03-23_19-03-23_COG.tif",
    ):
        (tmp_path / name).touch()
    assert find_product(tmp_path, 226).name.startswith("Classified_Output_Orbit-Cycle_226-")
    assert find_product(tmp_path, 2) is None


def test_class_percentages_ignore_nodata():
    arr = np.array([[0, 1], [4, 4]], dtype=np.uint8)
    pct = class_percentages(arr)
    assert pct[1] == pytest.approx(100 / 3)
    assert pct[4] == pytest.approx(200 / 3)


def test_compare_agreement_counts_pixels_valid_in_both(tmp_path):
    ours = np.array([[1, 1, 4, 0]], dtype=np.uint8)
    zen = np.array([[1, 4, 4, 1]], dtype=np.uint8)
    write_raster(tmp_path / "ours.tif", ours)
    write_raster(tmp_path / "zen.tif", zen)
    agreement = compare(tmp_path / "ours.tif", tmp_path / "zen.tif")
    assert agreement == pytest.approx(200 / 3)
