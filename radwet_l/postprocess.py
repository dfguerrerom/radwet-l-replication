"""Post-processing for classified output.

Handles:
  - Combining water + flood classifications
  - Mode filter (majority smoothing)
  - Sieve filter (remove small objects)
  - Applying LCC and slope masks
  - Color table definition
"""
import numpy as np
import subprocess
import rsgislib
import rsgislib.imagefilter
import rsgislib.imageutils
from .config import (
    CLASS_BACKGROUND, CLASS_FLOOD_FOREST, CLASS_OPEN_WATER,
    CLASS_OTHER_LAND, CLASS_NON_FOREST, CLASS_STEEP_SLOPE,
    MODE_FILTER_SIZE, SIEVE_THRESHOLD, COLOR_TABLE,
)


def combine_classifications(water_arr, flood_arr, lcc_mask):
    """Combine water and flood classifications.

    Matches original logic (applyXGBoostClassificationImgs_Bins.py:795-803):
      - In forest (lcc==0): water=2, flood=1, both(3)→4, neither→4
      - Non-forest (lcc!=0): set to 0 (reclassified to 5 after sieve by mask_image_with_class)

    Args:
        water_arr: 2D array with 0 or 2.
        flood_arr: 2D array with 0 or 1.
        lcc_mask: 2D array, 0=forest, 1=non-forest.

    Returns:
        2D array of combined class codes.
    """
    combined = water_arr + flood_arr  # 0, 1, 2, or 3

    result = np.where(
        lcc_mask == 0,
        np.where(combined == 0, CLASS_OTHER_LAND,
                 np.where(combined == 3, CLASS_OTHER_LAND, combined)),
        CLASS_BACKGROUND,  # non-forest → 0 (reclassified to 5 after sieve)
    )

    return result.astype(np.uint8)


def apply_mask(data, mask, mask_value):
    """Apply a binary mask to classification array.

    Args:
        data: 2D array of class codes.
        mask: 2D binary array (1=mask out).
        mask_value: Class code to assign to masked pixels.

    Returns:
        2D array with masked pixels set to mask_value.
    """
    return np.where(mask == 1, mask_value, data).astype(np.uint8)


def apply_mode_filter(input_path, output_path, window_size=MODE_FILTER_SIZE):
    """Apply mode (majority) filter to smooth classification.

    Args:
        input_path: Input classified image.
        output_path: Output filtered image.
        window_size: Filter window size.
    """
    rsgislib.imagefilter.apply_mode_filter(
        input_path, output_path, window_size, "GTIFF",
        rsgislib.TYPE_8UINT,
    )


def apply_sieve_filter(input_path, output_path, threshold=SIEVE_THRESHOLD):
    """Remove small isolated regions using GDAL sieve.

    Args:
        input_path: Input classified image.
        output_path: Output sieved image.
        threshold: Minimum number of pixels per region.
    """
    subprocess.run(
        ["gdal_sieve.py", "-st", str(threshold), "-8", "-of", "GTiff",
         input_path, output_path],
        check=True,
    )


def apply_color_table(image_path, color_lut=None):
    """Define color table on classified image.

    Args:
        image_path: Path to classified GeoTIFF.
        color_lut: Dict mapping class code -> hex color. Uses default if None.
    """
    if color_lut is None:
        color_lut = COLOR_TABLE
    rsgislib.imageutils.define_colour_table(image_path, color_lut)
