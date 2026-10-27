"""Shared configuration for the RadWet-L replication pipeline."""
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_DIR / "models"
CYCLE_DATES_CSV = REPO_DIR / "data" / "cycle_dates.csv"

# Working grid. The shipped models were trained on ~50 m tiles; running
# inference on 25 m inputs changes per-segment statistics and roughly doubles
# the flood-forest fraction, so inputs are coarsened to 50 m by default.
DEFAULT_WORKING_GRID = "50m"
TARGET_PIXEL_SIZE_50M_DEGREES = 0.000449
WORKING_GRID_RESAMPLE_ALG = "average"

# ScanSAR nominal incidence angle used for angle-optimal mosaicking, and the
# constant fallback when an L2.2 run has no incidence-angle raster.
NOMINAL_INCIDENCE_DEG = 37.0
FALLBACK_INCIDENCE_DEG = 38.5

# Classification thresholds
WATER_THRESHOLD = 0.99
FLOOD_THRESHOLD = 0.99

# Water model features (8) — from the original trainXGBoostGPU_Water.py:48
WATER_FEATURES = [
    "HHMean", "HVMean", "NDPIMean", "NDPIstdDev",
    "IncstdDev",
    "SlopeMean", "SlopestdDev",
    "HandstdDev",
]

# Flood model features (4) — from the original trainXGBoostGPU_Flood.py:56
FLOOD_FEATURES = ["HHMean", "HVMean", "NDPIMean", "IncMean"]

# Segmentation parameters (original defaults)
LOW_BS_NUM_CLUSTERS = 20
MAIN_NUM_CLUSTERS = 250
OBJ_SIZE = 5
LOW_BS_DIST_THRESH = 10
MAIN_DIST_THRESH = 10
HV_LOW_THRESH = -15  # dB threshold for low backscatter detection

# Post-processing
MODE_FILTER_SIZE = 3
SIEVE_THRESHOLD = 50  # minimum pixels per object
SLOPE_MASK_THRESH = 10  # degrees

# Class codes
CLASS_BACKGROUND = 0
CLASS_FLOOD_FOREST = 1
CLASS_OPEN_WATER = 2
CLASS_OTHER_LAND = 4
CLASS_NON_FOREST = 5
CLASS_STEEP_SLOPE = 6

CLASS_NAMES = {
    0: "Background",
    1: "Flood Forest",
    2: "Open Water",
    4: "Other Land",
    5: "Non-Forest",
    6: "Slope > 10°",
}

COLOR_TABLE = {
    0: "#000000",
    1: "#6CABDD",
    2: "#000080",
    4: "#004225",
    5: "#a7a7a7",
    6: "#d21255",
}
