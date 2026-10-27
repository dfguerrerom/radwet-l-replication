"""Classification module using pre-trained XGBoost models.

Handles:
  - Per-segment feature extraction from raster attribute tables
  - StandardScaler normalization
  - XGBoost prediction
  - Thresholding to class labels

IMPORTANT: The shipped models expect fewer features than the original apply script claims:
  - Water model: 8 features (not 21)
  - Flood model: 4 features (not 9)
See trainXGBoostGPU_Water.py:48 and trainXGBoostGPU_Flood.py:56.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from pickle import load
import rios.rat
import rsgislib
import rsgislib.rastergis
from .config import (
    WATER_FEATURES, FLOOD_FEATURES,
    WATER_THRESHOLD, FLOOD_THRESHOLD,
    CLASS_OPEN_WATER, CLASS_FLOOD_FOREST, CLASS_BACKGROUND,
)


def extract_segment_stats(clumps_img, alos_img, slope_img, hand_img,
                          compute_percentiles=False):
    """Populate raster attribute table with per-segment statistics.

    Extracts Mean and StdDev for all bands. Optionally extracts percentiles
    for HH, HV, NDPI (bands 1-3) — disabled by default since the shipped
    models don't use percentile features and the computation is memory-heavy.

    Args:
        clumps_img: Path to segmented clumps image (KEA with RAT).
        alos_img: Path to 4-band stacked image (HH, HV, NDPI, IncAngle as KEA).
        slope_img: Path to resampled slope image.
        hand_img: Path to resampled HAND image.
        compute_percentiles: If True, compute Per25/50/75 (memory-intensive).

    Returns:
        clumps_img path (RAT is modified in-place).
    """
    # Mean + StdDev for ALOS bands
    for band_num, prefix in [(1, "HH"), (2, "HV"), (3, "NDPI"), (4, "Inc")]:
        bs = [
            rsgislib.rastergis.BandAttStats(
                band=band_num, mean_field=f"{prefix}Mean", std_dev_field=f"{prefix}stdDev",
            ),
        ]
        rsgislib.rastergis.populate_rat_with_stats(alos_img, clumps_img, bs)

    # Mean + StdDev for ancillary data
    for img_path, prefix in [(slope_img, "Slope"), (hand_img, "Hand")]:
        bs = [
            rsgislib.rastergis.BandAttStats(
                band=1, mean_field=f"{prefix}Mean", std_dev_field=f"{prefix}stdDev",
            ),
        ]
        rsgislib.rastergis.populate_rat_with_stats(img_path, clumps_img, bs)

    # Percentiles for HH, HV, NDPI (bands 1-3 of alos_img)
    # Skipped by default: models only use Mean/StdDev features, and
    # populate_rat_with_percentiles is very memory-intensive.
    if compute_percentiles:
        for band_num, prefix in [(1, "HH"), (2, "HV"), (3, "NDPI")]:
            percs = [
                rsgislib.rastergis.BandAttPercentiles(percentile=25.0, field_name=f"{prefix}Per25"),
                rsgislib.rastergis.BandAttPercentiles(percentile=50.0, field_name=f"{prefix}Per50"),
                rsgislib.rastergis.BandAttPercentiles(percentile=75.0, field_name=f"{prefix}Per75"),
            ]
            rsgislib.rastergis.populate_rat_with_percentiles(
                alos_img, clumps_img, band_num, percs,
            )

    return clumps_img


def classify_array(data, model_path, scaler_path):
    """Run XGBoost classification on pre-extracted feature array.

    Args:
        data: 2D numpy array (n_samples, n_features). Raw (unscaled) values.
        model_path: Path to XGBoost model (.ubj or .json).
        scaler_path: Path to sklearn StandardScaler pickle.

    Returns:
        1D numpy array of prediction probabilities.
    """
    with open(scaler_path, "rb") as f:
        scaler = load(f)

    scaled = scaler.transform(data)

    model = xgb.Booster()
    model.load_model(model_path)

    dmatrix = xgb.DMatrix(scaled)
    predictions = model.predict(dmatrix)
    return predictions


def read_features_from_rat(clumps_img, feature_names):
    """Read per-segment features from raster attribute table.

    Args:
        clumps_img: Path to clumps image with populated RAT.
        feature_names: List of column names to read.

    Returns:
        pandas DataFrame with one row per segment.
    """
    df = pd.DataFrame()
    for name in feature_names:
        df[name] = rios.rat.readColumn(clumps_img, name, bandNumber=1)
    return df


def classify_water(clumps_img, model_path, scaler_path):
    """Classify open water using 8-feature water model.

    Args:
        clumps_img: Path to clumps image with populated RAT.
        model_path: Path to water XGBoost model.
        scaler_path: Path to water scaler.

    Returns:
        1D array of class labels (2=water, 0=not water) per segment.
    """
    df = read_features_from_rat(clumps_img, WATER_FEATURES)
    probs = classify_array(df.values, model_path, scaler_path)
    return np.where(probs > WATER_THRESHOLD, CLASS_OPEN_WATER, CLASS_BACKGROUND)


def classify_flood(clumps_img, model_path, scaler_path):
    """Classify inundated forest using 4-feature flood model.

    Args:
        clumps_img: Path to clumps image with populated RAT.
        model_path: Path to flood XGBoost model.
        scaler_path: Path to flood scaler.

    Returns:
        1D array of class labels (1=flood, 0=not flood) per segment.
    """
    df = read_features_from_rat(clumps_img, FLOOD_FEATURES)
    probs = classify_array(df.values, model_path, scaler_path)
    return np.where(probs > FLOOD_THRESHOLD, CLASS_FLOOD_FOREST, CLASS_BACKGROUND)
