"""Tests for the segmentation module."""
import numpy as np
import pytest


from radwet_l.segment import create_low_backscatter_image


def test_low_backscatter_mask():
    """Pixels with HV < -15 dB should be kept, others zeroed."""
    hh = np.array([[-8.0, -10.0, -5.0]], dtype=np.float32)
    hv = np.array([[-12.0, -18.0, -20.0]], dtype=np.float32)
    ndpi = np.array([[0.2, 0.3, 0.6]], dtype=np.float32)

    hh_out, hv_out, ndpi_out = create_low_backscatter_image(hh, hv, ndpi, threshold=-15)

    # HV=-12 > -15, so zeroed
    assert hh_out[0, 0] == 0.0
    # HV=-18 < -15, so kept
    assert hh_out[0, 1] == -10.0
    # HV=-20 < -15, so kept
    assert hh_out[0, 2] == -5.0


def test_low_backscatter_all_above_threshold():
    """If all HV values are above threshold, output should be all zeros."""
    hh = np.array([[-5.0, -8.0]], dtype=np.float32)
    hv = np.array([[-10.0, -12.0]], dtype=np.float32)
    ndpi = np.array([[0.3, 0.2]], dtype=np.float32)

    hh_out, hv_out, ndpi_out = create_low_backscatter_image(hh, hv, ndpi, threshold=-15)

    assert np.all(hh_out == 0.0)
    assert np.all(hv_out == 0.0)
    assert np.all(ndpi_out == 0.0)


def test_low_backscatter_preserves_shape():
    """Output shape should match input shape."""
    shape = (100, 50)
    hh = np.random.randn(*shape).astype(np.float32) - 10
    hv = np.random.randn(*shape).astype(np.float32) - 15
    ndpi = np.random.randn(*shape).astype(np.float32) * 0.3

    hh_out, hv_out, ndpi_out = create_low_backscatter_image(hh, hv, ndpi)

    assert hh_out.shape == shape
    assert hv_out.shape == shape
    assert ndpi_out.shape == shape
