"""Tests for cube_wrangler.parameters."""

from pathlib import Path

import pytest

from cube_wrangler.parameters import CategoriesConfig, Parameters, TimePeriodsConfig


@pytest.mark.roadway
def test_parameters_default_instantiation():
    """Parameters instantiate with no arguments."""
    p = Parameters()
    assert isinstance(p, Parameters)


@pytest.mark.roadway
def test_parameters_time_period_to_time_keys(cube_parameters):
    """time_period_to_time returns all five expected period codes."""
    tpt = cube_parameters.time_period_to_time
    assert set(tpt.keys()) == {"EA", "AM", "MD", "PM", "NT"}


@pytest.mark.roadway
def test_parameters_time_period_to_time_values(cube_parameters):
    """Each time period value is a (start, end) tuple of strings."""
    for code, span in cube_parameters.time_period_to_time.items():
        assert isinstance(span, tuple), f"{code} value should be a tuple"
        assert len(span) == 2, f"{code} value should have two elements"
        assert all(isinstance(t, str) for t in span), f"{code} times should be strings"


@pytest.mark.roadway
def test_parameters_categories_as_dict(cube_parameters):
    """categories.as_dict() contains all expected vehicle types."""
    cats = cube_parameters.categories.as_dict()
    assert set(cats.keys()) == {"sov", "hov2", "hov3", "truck"}


@pytest.mark.roadway
def test_parameters_categories_fallback_lists(cube_parameters):
    """Each category has a non-empty fallback list."""
    for name, fallbacks in cube_parameters.categories.as_dict().items():
        assert isinstance(fallbacks, list), f"{name} should be a list"
        assert len(fallbacks) > 0, f"{name} fallback list should not be empty"


@pytest.mark.roadway
def test_parameters_properties_to_split_keys(cube_parameters):
    """properties_to_split contains all expected property keys."""
    expected = {"trn_priority", "ttime_assert", "lanes", "ML_lanes", "price", "access"}
    assert set(cube_parameters.properties_to_split.keys()) == expected


@pytest.mark.roadway
def test_parameters_output_variables_nonempty(cube_parameters):
    """output_variables is a non-empty list of strings."""
    ov = cube_parameters.output_variables
    assert isinstance(ov, list)
    assert len(ov) > 0
    assert all(isinstance(v, str) for v in ov)


@pytest.mark.roadway
def test_parameters_settings_location_is_path(cube_parameters):
    """settings_location is a Path object."""
    assert isinstance(cube_parameters.settings_location, Path)


@pytest.mark.roadway
def test_parameters_scratch_location_is_path(cube_parameters):
    """scratch_location is a Path object."""
    assert isinstance(cube_parameters.scratch_location, Path)


@pytest.mark.roadway
def test_parameters_crosswalk_paths_are_paths(cube_parameters):
    """Crosswalk file shortcuts are Path objects."""
    assert isinstance(cube_parameters.net_to_dbf_crosswalk, Path)
    assert isinstance(cube_parameters.log_to_net_crosswalk, Path)


@pytest.mark.roadway
def test_parameters_output_file_paths_are_paths(cube_parameters):
    """All output file shortcuts set in __post_init__ are Path objects."""
    path_attrs = [
        "output_link_shp",
        "output_node_shp",
        "output_link_csv",
        "output_node_csv",
        "output_link_txt",
        "output_node_txt",
        "output_link_header_width_txt",
        "output_node_header_width_txt",
        "output_cube_network_script",
    ]
    for attr in path_attrs:
        val = getattr(cube_parameters, attr)
        assert isinstance(val, Path), f"{attr} should be a Path, got {type(val)}"


@pytest.mark.roadway
def test_parameters_custom_base_dir(tmp_path):
    """Custom base_dir is reflected in derived path attributes."""
    p = Parameters(base_dir=tmp_path)
    assert p.base_dir == tmp_path
    assert p.settings_location == tmp_path / "examples" / "settings"
    assert p.scratch_location == tmp_path / "tests" / "scratch"


@pytest.mark.roadway
def test_parameters_custom_time_periods():
    """Custom time period overrides are respected."""
    custom = TimePeriodsConfig(AM=("7:00", "9:00"))
    p = Parameters(time_periods=custom)
    assert p.time_period_to_time["AM"] == ("7:00", "9:00")
    # Other periods unchanged
    assert p.time_period_to_time["EA"] == ("3:00", "6:00")


@pytest.mark.roadway
def test_parameters_zones_default(cube_parameters):
    """Default zones value is set."""
    assert cube_parameters.zones == 3061


@pytest.mark.roadway
def test_time_periods_config_as_dict():
    """TimePeriodsConfig.as_dict() returns all five periods."""
    tp = TimePeriodsConfig()
    d = tp.as_dict()
    assert set(d.keys()) == {"EA", "AM", "MD", "PM", "NT"}
    assert d["AM"] == ("6:00", "10:00")


@pytest.mark.roadway
def test_categories_config_as_dict():
    """CategoriesConfig.as_dict() returns correct fallback structure."""
    cats = CategoriesConfig()
    d = cats.as_dict()
    assert "default" in d["sov"]
    assert "sov" in d["hov2"]
    assert "hov2" in d["hov3"]
