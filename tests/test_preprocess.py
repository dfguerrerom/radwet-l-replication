"""Tests for the preprocessing module."""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin


from radwet_l.preprocess import (
    dn_to_db,
    compute_ndpi,
    find_scene_group_paths,
    mosaic_scene_group,
)


def test_dn_to_db_basic():
    """DN=1000 should produce a finite dB value, DN=0 -> NaN."""
    dn = np.array([[1000, 2000], [0, 500]], dtype=np.float32)
    db = dn_to_db(dn)
    assert np.isfinite(db[0, 0])
    assert np.isfinite(db[0, 1])
    assert np.isnan(db[1, 0])  # DN=0 -> NaN
    assert np.isfinite(db[1, 1])


def test_dn_to_db_formula():
    """Verify: dB = 10 * log10(DN^2) - 83."""
    dn = np.array([[1000]], dtype=np.float32)
    db = dn_to_db(dn)
    expected = 10 * np.log10(1000.0**2) - 83  # = 60 - 83 = -23
    np.testing.assert_allclose(db[0, 0], expected, atol=0.01)


def test_dn_to_db_all_zero():
    """All-zero input should produce all-NaN output."""
    dn = np.zeros((3, 3), dtype=np.float32)
    db = dn_to_db(dn)
    assert np.all(np.isnan(db))


def test_compute_ndpi():
    """NDPI = (HH - HV) / (HH + HV)."""
    hh = np.array([[-10.0, -5.0]], dtype=np.float32)
    hv = np.array([[-15.0, -20.0]], dtype=np.float32)
    ndpi = compute_ndpi(hh, hv)
    expected = (hh - hv) / (hh + hv)
    np.testing.assert_allclose(ndpi, expected)


def test_compute_ndpi_handles_zero_denominator():
    """When HH + HV = 0, NDPI should be NaN."""
    hh = np.array([[5.0]], dtype=np.float32)
    hv = np.array([[-5.0]], dtype=np.float32)
    ndpi = compute_ndpi(hh, hv)
    assert np.isnan(ndpi[0, 0])


def test_dn_to_db_dn_one_becomes_nan():
    """DN=1 produces dB=-83 which should be NaN (matches original line 475)."""
    arr = np.array([[1, 100]], dtype=np.float32)
    result = dn_to_db(arr)
    assert np.isnan(result[0, 0]), "DN=1 should produce NaN (dB=-83 edge case)"
    assert not np.isnan(result[0, 1]), "DN=100 should produce valid dB"


def write_test_raster(path, array):
    """Write a tiny single-band GeoTIFF for preprocessing tests."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs="EPSG:4326",
        transform=from_origin(-76.0, -3.0, 0.1, 0.1),
        nodata=0,
    ) as ds:
        ds.write(array, 1)


def test_find_scene_group_paths_prefers_lin_family_for_incidence(tmp_path):
    """Angle aliases should resolve onto the full LIN scene family when present."""
    for name in [
        "L22_pass_2021-03-11_LIN_0.tif",
        "L22_pass_2021-03-11_LIN_1.tif",
        "L22_pass_2021-03-11_angle_0.tif",
    ]:
        (tmp_path / name).touch()

    result = find_scene_group_paths(tmp_path / "L22_pass_2021-03-11_angle_0.tif")

    assert result == [
        str(tmp_path / "L22_pass_2021-03-11_LIN_0.tif"),
        str(tmp_path / "L22_pass_2021-03-11_LIN_1.tif"),
    ]


def test_mosaic_scene_group_averages_overlap_and_keeps_single_scene_pixels(tmp_path):
    """Overlapping valid pixels should be averaged while non-overlap stays intact."""
    scene0 = tmp_path / "L22_pass_2021-03-11_HH_0.tif"
    scene1 = tmp_path / "L22_pass_2021-03-11_HH_1.tif"
    output = tmp_path / "hh_scene_mosaic.tif"

    write_test_raster(scene0, np.array([[10, 10], [10, 0]], dtype=np.uint16))
    write_test_raster(scene1, np.array([[0, 30], [30, 30]], dtype=np.uint16))

    mosaic_scene_group([str(scene0), str(scene1)], str(output))

    with rasterio.open(output) as ds:
        result = ds.read(1)
        assert ds.dtypes[0] == "float32"

    expected = np.array([[10.0, 20.0], [20.0, 30.0]], dtype=np.float32)
    np.testing.assert_allclose(result, expected)
