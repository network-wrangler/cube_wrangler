"""Tests for cube_wrangler.util."""

from __future__ import annotations

import datetime

import numpy as np
import pytest

from cube_wrangler.util import (
    column_name_to_parts,
    get_shared_streets_intersection_hash,
    hhmmss_to_datetime,
    secs_to_datetime,
    shorten_name,
)

# ---------------------------------------------------------------------------
# get_shared_streets_intersection_hash
# ---------------------------------------------------------------------------


@pytest.mark.roadway
def test_shared_streets_hash_returns_string():
    """Hash is a non-empty hex string."""
    result = get_shared_streets_intersection_hash(44.952112, -93.096598)
    assert isinstance(result, str)
    assert len(result) == 32  # md5 hex digest


@pytest.mark.roadway
def test_shared_streets_hash_with_osm_node_id():
    """Hash differs when osm_node_id is provided."""
    h1 = get_shared_streets_intersection_hash(44.952112, -93.096598)
    h2 = get_shared_streets_intersection_hash(44.952112, -93.096598, osm_node_id=954734870)
    assert isinstance(h2, str)
    assert len(h2) == 32
    assert h1 != h2


@pytest.mark.roadway
def test_shared_streets_hash_deterministic():
    """Same inputs produce the same hash."""
    h1 = get_shared_streets_intersection_hash(44.952112, -93.096598, osm_node_id=12345)
    h2 = get_shared_streets_intersection_hash(44.952112, -93.096598, osm_node_id=12345)
    assert h1 == h2


# ---------------------------------------------------------------------------
# hhmmss_to_datetime
# ---------------------------------------------------------------------------


@pytest.mark.roadway
def test_hhmmss_to_datetime_basic():
    """Parses HH:MM:SS into a datetime.time."""
    result = hhmmss_to_datetime("06:00:00")
    assert result == datetime.time(6, 0, 0)


@pytest.mark.roadway
def test_hhmmss_to_datetime_hhmm():
    """Parses HH:MM (no seconds) into a datetime.time."""
    result = hhmmss_to_datetime("6:00")
    assert result == datetime.time(6, 0)


@pytest.mark.roadway
def test_hhmmss_to_datetime_returns_time_type():
    """Return type is datetime.time."""
    result = hhmmss_to_datetime("10:30:00")
    assert isinstance(result, datetime.time)


# ---------------------------------------------------------------------------
# secs_to_datetime
# ---------------------------------------------------------------------------


@pytest.mark.roadway
def test_secs_to_datetime_midnight():
    """0 seconds = midnight."""
    result = secs_to_datetime(0)
    assert result == datetime.time(0, 0, 0)


@pytest.mark.roadway
def test_secs_to_datetime_noon():
    """43200 seconds = 12:00:00."""
    result = secs_to_datetime(43200)
    assert result == datetime.time(12, 0, 0)


@pytest.mark.roadway
def test_secs_to_datetime_returns_time_type():
    """Return type is datetime.time."""
    result = secs_to_datetime(3600)
    assert isinstance(result, datetime.time)


# ---------------------------------------------------------------------------
# shorten_name
# ---------------------------------------------------------------------------


@pytest.mark.roadway
def test_shorten_name_string_input():
    """Comma-separated string is deduplicated and cleaned."""
    result = shorten_name("Main St,Main St")
    assert isinstance(result, str)
    assert "Main St" in result


@pytest.mark.roadway
def test_shorten_name_float_input():
    """Float input is converted to string."""
    result = shorten_name(1.0)
    assert isinstance(result, str)


@pytest.mark.roadway
def test_shorten_name_numpy_int():
    """Numpy int input is converted to string."""
    result = shorten_name(np.int64(42))
    assert isinstance(result, str)


@pytest.mark.roadway
def test_shorten_name_removes_special_chars():
    """Non-word characters are replaced with spaces."""
    result = shorten_name("Héllo & Wörld!")
    assert isinstance(result, str)
    assert "&" not in result


# ---------------------------------------------------------------------------
# column_name_to_parts
# ---------------------------------------------------------------------------


@pytest.mark.roadway
def test_column_name_to_parts_unscoped(cube_parameters):
    """Column not in properties_to_split returns (col, None, None, 0)."""
    base, tp, cat, managed = column_name_to_parts("model_link_id", cube_parameters)
    assert base == "model_link_id"
    assert tp is None
    assert cat is None
    assert managed == 0


@pytest.mark.roadway
def test_column_name_to_parts_lanes_with_timeperiod(cube_parameters):
    """lanes_AM splits into (lanes, AM, None, 0)."""
    base, tp, cat, managed = column_name_to_parts("lanes_AM", cube_parameters)
    assert base == "lanes"
    assert tp == "AM"
    assert cat is None
    assert managed == 0


@pytest.mark.roadway
def test_column_name_to_parts_managed_lane(cube_parameters):
    """ML_lanes_PM has managed=1."""
    base, tp, cat, managed = column_name_to_parts("ML_lanes_PM", cube_parameters)
    assert managed == 1
    assert tp == "PM"


@pytest.mark.roadway
def test_column_name_to_parts_price_with_category(cube_parameters):
    """price_sov returns category=sov."""
    base, tp, cat, managed = column_name_to_parts("price_sov", cube_parameters)
    assert base == "price"
    assert cat == "sov"
