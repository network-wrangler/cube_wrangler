"""Tests for cube_wrangler.transit.StandardTransit."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cube_wrangler.parameters import Parameters
from cube_wrangler.transit import StandardTransit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transit(parameters=None):
    """Return a StandardTransit instance backed by a mock partridge feed."""
    fake_feed = MagicMock()
    return StandardTransit(fake_feed, parameters=parameters or Parameters())


# ---------------------------------------------------------------------------
# StandardTransit instantiation
# ---------------------------------------------------------------------------


@pytest.mark.roadway
def test_standard_transit_instantiates_with_mock_feed():
    """StandardTransit can be constructed with a mock feed."""
    st = _make_transit()
    assert isinstance(st, StandardTransit)


@pytest.mark.roadway
def test_standard_transit_stores_parameters():
    """Parameters attribute is a Parameters instance."""
    p = Parameters()
    st = _make_transit(parameters=p)
    assert st.parameters is p


@pytest.mark.roadway
def test_standard_transit_accepts_dict_parameters():
    """Passing an empty dict creates a default Parameters instance."""
    fake_feed = MagicMock()
    st = StandardTransit(fake_feed, parameters={})
    assert isinstance(st.parameters, Parameters)


@pytest.mark.roadway
def test_standard_transit_rejects_invalid_parameters():
    """Non-dict, non-Parameters value raises ValueError."""
    fake_feed = MagicMock()
    with pytest.raises(ValueError, match="Parameters should be a dict or instance"):
        StandardTransit(fake_feed, parameters=42)


# ---------------------------------------------------------------------------
# time_to_cube_time_period
# ---------------------------------------------------------------------------


@pytest.mark.roadway
def test_time_to_cube_time_period_am(cube_parameters):
    """7:00 AM (25200 seconds) maps to AM period."""
    st = _make_transit(parameters=cube_parameters)
    result = st.time_to_cube_time_period(25200)  # 7 * 3600
    assert result == "AM"


@pytest.mark.roadway
def test_time_to_cube_time_period_md(cube_parameters):
    """11:00 AM (39600 seconds) maps to MD period (10:00-15:00)."""
    st = _make_transit(parameters=cube_parameters)
    result = st.time_to_cube_time_period(39600)  # 11 * 3600
    assert result == "MD"


@pytest.mark.roadway
def test_time_to_cube_time_period_pm(cube_parameters):
    """4:00 PM (57600 seconds) maps to PM period."""
    st = _make_transit(parameters=cube_parameters)
    result = st.time_to_cube_time_period(57600)  # 16 * 3600
    assert result == "PM"


@pytest.mark.roadway
def test_time_to_cube_time_period_nt(cube_parameters):
    """8:00 PM (72000 seconds) maps to NT period (19:00-03:00)."""
    st = _make_transit(parameters=cube_parameters)
    result = st.time_to_cube_time_period(72000)  # 20 * 3600
    assert result == "NT"


@pytest.mark.roadway
def test_time_to_cube_time_period_returns_string(cube_parameters):
    """Default (as_str=True) returns a string."""
    st = _make_transit(parameters=cube_parameters)
    result = st.time_to_cube_time_period(25200)
    assert isinstance(result, str)


@pytest.mark.roadway
def test_time_to_cube_time_period_verbose_does_not_raise(cube_parameters):
    """verbose=True should not raise."""
    st = _make_transit(parameters=cube_parameters)
    result = st.time_to_cube_time_period(25200, verbose=True)
    assert isinstance(result, str)
