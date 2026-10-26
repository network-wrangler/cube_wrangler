"""Tests for cube_wrangler roadway utilities."""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"

STPAUL_SHAPE_FILE = DATA_DIR / "st_paul_shape.geojson"
STPAUL_LINK_FILE = DATA_DIR / "st_paul_link.json"
STPAUL_NODE_FILE = DATA_DIR / "st_paul_node.geojson"


@pytest.mark.roadway
def test_parameter_read(cube_parameters):
    """Parameters instance is created with expected keys."""
    assert hasattr(cube_parameters, "time_period_to_time")
    assert set(cube_parameters.time_period_to_time.keys()) == {"EA", "AM", "MD", "PM", "NT"}
    assert hasattr(cube_parameters, "categories")
    assert "sov" in cube_parameters.categories.as_dict()


@pytest.mark.roadway
def test_parameter_settings_location_is_set(cube_parameters):
    """Parameters exposes a settings_location Path attribute."""
    from pathlib import Path

    assert hasattr(cube_parameters, "settings_location")
    assert isinstance(cube_parameters.settings_location, Path)


@pytest.mark.roadway
def test_stpaul_data_files_present():
    """Required stpaul test data files are present in tests/data/."""
    for path in [STPAUL_LINK_FILE, STPAUL_NODE_FILE, STPAUL_SHAPE_FILE]:
        assert path.exists(), f"Missing test data file: {path}"
