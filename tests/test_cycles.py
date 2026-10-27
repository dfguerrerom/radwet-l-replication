"""Tests for the orbit-cycle calendar."""
import datetime as dt

import pytest

from radwet_l.cycles import cycle_dates, cycle_for_date


def test_cycle_dates_matches_zenodo_product_names():
    """Zenodo names cycle 226 'Dated-06-03-23_19-03-23'."""
    assert cycle_dates(226) == (dt.date(2023, 3, 6), dt.date(2023, 3, 19))


def test_cycles_span_14_days():
    start, end = cycle_dates(174)
    assert (end - start).days == 13


def test_cycle_for_date():
    assert cycle_for_date("2021-03-11") == 174
    assert cycle_for_date(dt.date(2019, 2, 4)) == 119
    assert cycle_for_date("2010-01-01") is None


def test_unknown_cycle_raises():
    with pytest.raises(ValueError):
        cycle_dates(100000)
