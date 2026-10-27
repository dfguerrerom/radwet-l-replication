"""Preprocessing functions for ALOS-2 SAR data.

Handles:
  - DN to dB conversion (for Level 1.1/1.5 raw data)
  - NDPI (Normalized Difference Polarization Index) calculation
  - Lee speckle filter
  - Band stacking
  - Working-grid resampling
  - Ancillary data resampling
"""
from pathlib import Path
import re

import numpy as np
import rsgislib
import rsgislib.imagefilter
import rsgislib.imageutils
from osgeo import gdal


SCENE_FILE_RE = re.compile(
    r"^(?P<prefix>.+?)_(?P<token>HH|HV|LIN|angle)_(?P<scene>\d+)(?P<suffix>(?:_[0-9]+m)?)\.tif$"
)
INCIDENCE_SCENE_TOKENS = ("LIN", "angle")


def dn_to_db(dn_array):
    """Convert raw DN values to sigma-naught dB.

    Formula: dB = 10 * log10(DN^2) - 83
    DN=0 pixels become NaN.
    DN=1 produces dB=-83, which is treated as NoData and also set to NaN.
    This matches the original pipeline (applyXGBoostClassificationImgs_Bins.py line 475):
        hhArraydBNanCor = np.where(hhArraydB == -83, np.nan, hhArraydB)

    Args:
        dn_array: 2D numpy array of uint16/float DN values.

    Returns:
        2D numpy array of float32 dB values (NaN where DN=0 or DN=1).
    """
    arr = dn_array.astype(np.float64)
    arr[arr == 0] = np.nan
    db = 10.0 * np.log10(arr * arr) - 83.0
    db[db == -83.0] = np.nan
    return db.astype(np.float32)


def parse_scene_raster(path):
    """Parse a numbered scene raster filename."""
    match = SCENE_FILE_RE.match(Path(path).name)
    if match is None:
        return None

    info = match.groupdict()
    info["scene"] = int(info["scene"])
    return info


def find_scene_group_paths(source_path):
    """Resolve all sibling scene rasters that belong to the same input family."""
    source = Path(source_path)
    info = parse_scene_raster(source)
    if info is None:
        return [str(source)]

    token_priority = (
        list(INCIDENCE_SCENE_TOKENS)
        if info["token"] in INCIDENCE_SCENE_TOKENS
        else [info["token"]]
    )

    selected = {}
    for token in token_priority:
        pattern = f"{info['prefix']}_{token}_*{info['suffix']}.tif"
        for candidate in sorted(source.parent.glob(pattern)):
            candidate_info = parse_scene_raster(candidate)
            if candidate_info is None:
                continue
            if candidate_info["prefix"] != info["prefix"]:
                continue
            if candidate_info["suffix"] != info["suffix"]:
                continue
            selected.setdefault(candidate_info["scene"], str(candidate))

    if not selected:
        return [str(source)]

    return [selected[scene] for scene in sorted(selected)]


def mosaic_scene_group(scene_paths, output_path, nodata=0.0):
    """Average matching scene rasters into a single mosaic."""
    if not scene_paths:
        raise ValueError("scene_paths must contain at least one raster")
    if len(scene_paths) == 1:
        return scene_paths[0]

    datasets = []
    try:
        for scene_path in scene_paths:
            ds = gdal.Open(scene_path)
            if ds is None:
                raise FileNotFoundError(f"Cannot open: {scene_path}")
            datasets.append(ds)

        first = datasets[0]
        first_gt = first.GetGeoTransform()
        if abs(first_gt[2]) > 1e-12 or abs(first_gt[4]) > 1e-12:
            raise ValueError("Rotated rasters are not supported for scene mosaics")

        x_res = first_gt[1]
        y_res = first_gt[5]
        projection = first.GetProjection()

        min_x = first_gt[0]
        max_y = first_gt[3]
        max_x = first_gt[0] + first.RasterXSize * x_res
        min_y = first_gt[3] + first.RasterYSize * y_res

        for ds in datasets[1:]:
            gt = ds.GetGeoTransform()
            if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12:
                raise ValueError("Rotated rasters are not supported for scene mosaics")
            if ds.GetProjection() != projection:
                raise ValueError("Scene mosaic inputs must share the same projection")
            if abs(gt[1] - x_res) > 1e-12 or abs(gt[5] - y_res) > 1e-12:
                raise ValueError("Scene mosaic inputs must share the same pixel size")

            min_x = min(min_x, gt[0])
            max_y = max(max_y, gt[3])
            max_x = max(max_x, gt[0] + ds.RasterXSize * gt[1])
            min_y = min(min_y, gt[3] + ds.RasterYSize * gt[5])

        out_cols = int(round((max_x - min_x) / x_res))
        out_rows = int(round((max_y - min_y) / abs(y_res)))

        sum_arr = np.zeros((out_rows, out_cols), dtype=np.float64)
        count_arr = np.zeros((out_rows, out_cols), dtype=np.uint16)

        for ds in datasets:
            gt = ds.GetGeoTransform()
            x_off = int(round((gt[0] - min_x) / x_res))
            y_off = int(round((max_y - gt[3]) / abs(y_res)))

            arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            valid = np.isfinite(arr) & (arr != 0)

            row_slice = slice(y_off, y_off + ds.RasterYSize)
            col_slice = slice(x_off, x_off + ds.RasterXSize)
            sum_window = sum_arr[row_slice, col_slice]
            count_window = count_arr[row_slice, col_slice]
            sum_window[valid] += arr[valid]
            count_window[valid] += 1

        mosaic = np.full((out_rows, out_cols), float(nodata), dtype=np.float32)
        valid = count_arr > 0
        mosaic[valid] = (sum_arr[valid] / count_arr[valid]).astype(np.float32)

        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            output_path,
            out_cols,
            out_rows,
            1,
            gdal.GDT_Float32,
            options=["COMPRESS=LZW"],
        )
        out_ds.SetProjection(projection)
        out_ds.SetGeoTransform((min_x, x_res, 0.0, max_y, 0.0, y_res))
        band = out_ds.GetRasterBand(1)
        band.WriteArray(mosaic)
        band.SetNoDataValue(float(nodata))

        description = first.GetRasterBand(1).GetDescription()
        if description:
            band.SetDescription(description)

        out_ds.FlushCache()
        del out_ds
    finally:
        for ds in datasets:
            del ds

    return output_path


def compute_ndpi(hh_db, hv_db):
    """Compute Normalized Difference Polarization Index.

    NDPI = (HH - HV) / (HH + HV)

    Args:
        hh_db: 2D array, HH backscatter in dB.
        hv_db: 2D array, HV backscatter in dB.

    Returns:
        2D array of NDPI values.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = hh_db + hv_db
        ndpi = np.where(denom != 0, (hh_db - hv_db) / denom, np.nan)
    return ndpi.astype(np.float32)


def save_geotiff(ref_path, arrays, output_path, nodata=np.nan):
    """Save numpy arrays as a multi-band GeoTIFF using a reference file for georef.

    Args:
        ref_path: Path to reference GeoTIFF (for projection, transform, size).
        arrays: List of 2D numpy arrays (one per band).
        output_path: Output GeoTIFF path.
        nodata: NoData value.
    """
    ref_ds = gdal.Open(ref_path)
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path,
        ref_ds.RasterXSize,
        ref_ds.RasterYSize,
        len(arrays),
        gdal.GDT_Float32,
    )
    out_ds.SetProjection(ref_ds.GetProjection())
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    for i, arr in enumerate(arrays):
        band = out_ds.GetRasterBand(i + 1)
        band.WriteArray(arr)
        band.SetNoDataValue(float(nodata))
    out_ds.FlushCache()
    del out_ds, ref_ds


def apply_lee_filter(input_path, output_path, window_size=3):
    """Apply Lee speckle filter using rsgislib.

    Args:
        input_path: Input multi-band image path.
        output_path: Output filtered image path.
        window_size: Filter window size (default 3x3).
    """
    rsgislib.imagefilter.apply_lee_filter(
        input_path, output_path, window_size, window_size,
        "GTIFF", rsgislib.TYPE_32FLOAT,
    )


def stack_bands(band_paths, output_path):
    """Stack multiple single-band or multi-band images into one image.

    Args:
        band_paths: List of paths to GeoTIFF/KEA images.
        output_path: Output stacked image path (KEA format).
    """
    rsgislib.imageutils.stack_img_bands(
        band_paths, None, output_path, 0, 0, "KEA", rsgislib.TYPE_32FLOAT,
    )


def get_pixel_size(path):
    """Return absolute pixel size (x_res, y_res) in source units."""
    ds = gdal.Open(path)
    if ds is None:
        raise FileNotFoundError(f"Cannot open: {path}")
    geo_transform = ds.GetGeoTransform()
    del ds
    return abs(geo_transform[1]), abs(geo_transform[5])


def resample_to_working_grid(source_path, output_path, target_x_res, target_y_res=None,
                             resample_alg="average", tol=1e-9):
    """Resample an input raster to the pipeline working grid.

    This is used to coarsen 25 m GEE inputs to the 50 m grid the models were
    trained on, using GDAL ``average`` resampling onto a 0.000449 degree grid.

    Args:
        source_path: Path to the source raster.
        output_path: Path for the resampled raster.
        target_x_res: Target x pixel size.
        target_y_res: Target y pixel size. Defaults to x size.
        resample_alg: GDAL resampling algorithm name.
        tol: Absolute tolerance for "already on target grid" detection.

    Returns:
        Path to the working-grid raster. If the source already matches the
        requested grid, the original path is returned.
    """
    if target_y_res is None:
        target_y_res = target_x_res

    x_res, y_res = get_pixel_size(source_path)
    if abs(x_res - target_x_res) <= tol and abs(y_res - target_y_res) <= tol:
        return source_path

    options = gdal.WarpOptions(
        format="GTiff",
        xRes=target_x_res,
        yRes=target_y_res,
        resampleAlg=resample_alg,
        creationOptions=["COMPRESS=LZW"],
    )
    out_ds = gdal.Warp(output_path, source_path, options=options)
    if out_ds is None:
        raise RuntimeError(f"Failed to resample {source_path} to {output_path}")
    del out_ds
    return output_path


def resample_to_match(source_path, reference_path, output_path):
    """Resample source image to match reference image grid.

    Args:
        source_path: Path to image to resample (slope, HAND, etc).
        reference_path: Path to reference image (ALOS tile).
        output_path: Output resampled image path.
    """
    rsgislib.imageutils.resample_img_to_match(
        reference_path, source_path, output_path,
        "GTIFF", rsgislib.INTERP_BILINEAR,
    )


def mask_image(input_img, mask_img, output_img, mask_value=1):
    """Mask a multi-band image using a single-band mask.

    Pixels where mask == mask_value are set to 0 in the output.
    Matches original pipeline lines 617-621 where LCC and slope masks
    are applied to the stacked image BEFORE segmentation.

    Args:
        input_img: Path to input multi-band image.
        mask_img: Path to single-band mask image.
        output_img: Path for output masked image.
        mask_value: Value(s) in mask that identify pixels to zero out.
    """
    rsgislib.imageutils.mask_img(
        input_img, mask_img, output_img, "GTIFF",
        rsgislib.TYPE_32FLOAT, 0, mask_value,
    )


def mask_image_with_class(input_img, mask_img, output_img, mask_value=1, class_value=5):
    """Mask image and assign a class value to masked pixels.

    Like mask_image but sets masked pixels to class_value instead of 0.
    Used for post-sieve masking: LCC -> class 5, slope -> class 6.

    Args:
        input_img: Path to input image.
        mask_img: Path to single-band mask image.
        output_img: Path for output image.
        mask_value: Value in mask identifying pixels to reclassify.
        class_value: Class code to assign to masked pixels.
    """
    rsgislib.imageutils.mask_img(
        input_img, mask_img, output_img, "GTIFF",
        rsgislib.TYPE_8UINT, class_value, mask_value,
    )


def gen_valid_mask(input_img, output_mask):
    """Generate a binary mask of finite (non-NaN, non-Inf) pixels.

    Matches original line 629: rsgislib.imageutils.gen_finite_mask().
    Pixels with all-finite values across bands get value 1; others get 0.

    Args:
        input_img: Path to input image.
        output_mask: Path for output binary mask image.
    """
    rsgislib.imageutils.gen_finite_mask(input_img, output_mask, "GTIFF")
