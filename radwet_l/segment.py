"""Segmentation functions using rsgislib Shepherd algorithm.

Two-stage approach:
  1. Segment low-backscatter areas (likely open water) with fewer clusters
  2. Segment full image with more clusters
  3. Union both segmentation results
"""
import shutil
import numpy as np
import rsgislib
import rsgislib.segmentation.shepherdseg
import rsgislib.segmentation
import rsgislib.rastergis
from .config import (
    LOW_BS_NUM_CLUSTERS, MAIN_NUM_CLUSTERS, OBJ_SIZE,
    LOW_BS_DIST_THRESH, MAIN_DIST_THRESH, HV_LOW_THRESH,
)


def create_low_backscatter_image(hh, hv, ndpi, threshold=HV_LOW_THRESH):
    """Mask image to keep only low-backscatter (likely water) pixels.

    Args:
        hh: 2D array, HH band in dB.
        hv: 2D array, HV band in dB.
        ndpi: 2D array, NDPI values.
        threshold: HV threshold in dB (default -15).

    Returns:
        Tuple of (hh_masked, hv_masked, ndpi_masked) — zeroed where HV >= threshold.
    """
    mask = hv < threshold
    hh_out = np.where(mask, hh, 0.0).astype(np.float32)
    hv_out = np.where(mask, hv, 0.0).astype(np.float32)
    ndpi_out = np.where(mask, ndpi, 0.0).astype(np.float32)
    return hh_out, hv_out, ndpi_out


def run_shepherd_segmentation(input_img, output_clumps, num_clusters, obj_size,
                               dist_thresh, sampling=100, bands=None):
    """Run Shepherd's segmentation on an image.

    Args:
        input_img: Path to input image (multi-band).
        output_clumps: Path for output clumps image (KEA format).
        num_clusters: Number of K-means clusters.
        obj_size: Minimum segment size in pixels.
        dist_thresh: Distance threshold for merging.
        sampling: Subsampling ratio for K-means (1=no subsampling).
        bands: List of band indices to use (1-indexed). None = [1, 2, 3].
    """
    if bands is None:
        bands = [1, 2, 3]

    rsgislib.segmentation.shepherdseg.run_shepherd_segmentation(
        input_img, output_clumps,
        num_clusters=num_clusters,
        min_n_pxls=obj_size,
        dist_thres=dist_thresh,
        bands=bands,
        sampling=sampling,
        km_max_iter=50,
    )


def union_clumps(clumps_list, output_path):
    """Merge multiple clump images into a single segmentation.

    Args:
        clumps_list: List of paths to clump images.
        output_path: Path for output merged clumps (KEA format).
    """
    rsgislib.segmentation.union_of_clumps(
        clumps_list, output_path, "KEA", 0, True,
    )
    rsgislib.rastergis.pop_rat_img_stats(output_path, True, True, True, 1)


def segment_tile(alos_img, low_bs_img, output_clumps, has_low_bs=True):
    """Run full two-stage segmentation pipeline.

    Args:
        alos_img: Path to ALOS stacked image (HH, HV, NDPI, IncAngle as KEA).
        low_bs_img: Path to low-backscatter image (HH, HV, NDPI masked).
        output_clumps: Path for output merged clumps.
        has_low_bs: Whether low-backscatter image has valid pixels.

    Returns:
        Path to final clumps image.
    """
    clumps_low = low_bs_img.replace(".tif", "_clumps.kea")
    clumps_main = alos_img.replace(".kea", "_clumps.kea")

    # Stage 1: Low backscatter segmentation (open water candidates)
    if has_low_bs:
        try:
            run_shepherd_segmentation(
                low_bs_img, clumps_low,
                num_clusters=LOW_BS_NUM_CLUSTERS,
                obj_size=OBJ_SIZE,
                dist_thresh=LOW_BS_DIST_THRESH,
                sampling=1,
            )
        except Exception as e:
            print(f"  Low backscatter segmentation failed: {e}")
            has_low_bs = False

    # Stage 2: Main segmentation (full image)
    # Adaptive sampling: reduce for smaller images (e.g., 50m grid) to ensure
    # enough unique data points for K-means clustering
    from osgeo import gdal as _gdal
    _ds = _gdal.Open(alos_img)
    _npx = _ds.RasterXSize * _ds.RasterYSize
    del _ds
    main_sampling = 100 if _npx > 50_000_000 else max(1, _npx // 500_000)

    run_shepherd_segmentation(
        alos_img, clumps_main,
        num_clusters=MAIN_NUM_CLUSTERS,
        obj_size=OBJ_SIZE,
        dist_thresh=MAIN_DIST_THRESH,
        sampling=main_sampling,
    )

    # Union of both segmentation results
    if has_low_bs:
        union_clumps([clumps_low, clumps_main], output_clumps)
    else:
        shutil.copy(clumps_main, output_clumps)
        rsgislib.rastergis.pop_rat_img_stats(output_clumps, True, True, True, 1)

    return output_clumps
