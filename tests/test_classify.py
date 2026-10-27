"""Tests for the classification module."""
import os
import numpy as np
import pytest


from radwet_l.config import WATER_FEATURES, FLOOD_FEATURES, MODEL_DIR
from radwet_l.classify import classify_array


def test_water_features_count():
    """Water model expects exactly 8 features."""
    assert len(WATER_FEATURES) == 8


def test_flood_features_count():
    """Flood model expects exactly 4 features."""
    assert len(FLOOD_FEATURES) == 4


def test_water_features_match_scaler():
    """Water features must match the 8 features from trainXGBoostGPU_Water.py:48."""
    expected = [
        "HHMean", "HVMean", "NDPIMean", "NDPIstdDev",
        "IncstdDev", "SlopeMean", "SlopestdDev", "HandstdDev",
    ]
    assert WATER_FEATURES == expected


def test_flood_features_match_scaler():
    """Flood features must match the 4 features from trainXGBoostGPU_Flood.py:56."""
    expected = ["HHMean", "HVMean", "NDPIMean", "IncMean"]
    assert FLOOD_FEATURES == expected


def test_classify_array_water():
    """classify_array should return probabilities between 0 and 1."""
    model_path = os.path.join(MODEL_DIR, "XGBoost_Water.ubj")
    scaler_path = os.path.join(MODEL_DIR, "Scaler_Water.pkl")


    # Fake data matching 8 features
    fake_data = np.random.randn(10, 8).astype(np.float32)
    predictions = classify_array(fake_data, model_path, scaler_path)
    assert predictions.shape == (10,)
    assert np.all(predictions >= 0) and np.all(predictions <= 1)


def test_classify_array_flood():
    """classify_array should return probabilities between 0 and 1."""
    model_path = os.path.join(MODEL_DIR, "XGBoost_Flood.ubj")
    scaler_path = os.path.join(MODEL_DIR, "Scaler_Flood.pkl")


    # Fake data matching 4 features
    fake_data = np.random.randn(10, 4).astype(np.float32)
    predictions = classify_array(fake_data, model_path, scaler_path)
    assert predictions.shape == (10,)
    assert np.all(predictions >= 0) and np.all(predictions <= 1)
