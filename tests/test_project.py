"""Tests for cube_wrangler.project.Project (static / data-only methods)."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from cube_wrangler.project import Project

# ---------------------------------------------------------------------------
# Project.read_logfile
# ---------------------------------------------------------------------------


@pytest.mark.roadway
def test_read_logfile_returns_dataframe(log_file_10):
    """read_logfile returns a non-empty DataFrame."""
    df = Project.read_logfile(str(log_file_10))
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


@pytest.mark.roadway
def test_read_logfile_has_expected_columns(log_file_10):
    """Log file DataFrame contains OBJECT, OPERATION, GROUP and link columns."""
    df = Project.read_logfile(str(log_file_10))
    for col in ("OBJECT", "OPERATION", "GROUP"):
        assert col in df.columns, f"Expected column {col!r} in logfile DataFrame"


@pytest.mark.roadway
def test_read_logfile_link_count(log_file_10):
    """10-row log file yields 10 link rows."""
    df = Project.read_logfile(str(log_file_10))
    link_rows = df[df["OBJECT"] == "L"]
    assert len(link_rows) == 10


@pytest.mark.roadway
def test_read_logfile_accepts_path_object(log_file_10):
    """read_logfile accepts a Path object (via str coercion)."""
    df = Project.read_logfile(str(log_file_10))
    assert isinstance(df, pd.DataFrame)


@pytest.mark.roadway
def test_read_logfile_accepts_list(log_file_10, log_file_50):
    """read_logfile concatenates multiple log files when given a list."""
    df = Project.read_logfile([str(log_file_10), str(log_file_50)])
    link_rows = df[df["OBJECT"] == "L"]
    assert len(link_rows) == 60  # 10 + 50


@pytest.mark.roadway
def test_read_logfile_strips_bracket_suffixes(log_file_10):
    """Column names like NAME[111] are shortened to NAME."""
    df = Project.read_logfile(str(log_file_10))
    for col in df.columns:
        assert "[" not in col, f"Column {col!r} still has bracket suffix"


@pytest.mark.roadway
def test_read_logfile_larger_file(log_file_100):
    """100-row log file parses without error."""
    df = Project.read_logfile(str(log_file_100))
    link_rows = df[df["OBJECT"] == "L"]
    assert len(link_rows) == 100


# ---------------------------------------------------------------------------
# Project instantiation (minimal, no network required)
# ---------------------------------------------------------------------------


@pytest.mark.roadway
def test_project_default_name():
    """Project without project_name uses DEFAULT_PROJECT_NAME."""
    p = Project()
    assert p.project_name == Project.DEFAULT_PROJECT_NAME


@pytest.mark.roadway
def test_project_custom_name():
    """Project stores the supplied project_name."""
    p = Project(project_name="MyProject")
    assert p.project_name == "MyProject"


@pytest.mark.roadway
def test_project_with_parameters_dict(cube_parameters):
    """Project accepts a Parameters instance."""
    p = Project(parameters=cube_parameters)
    assert p.parameters is cube_parameters


@pytest.mark.roadway
def test_project_roadway_changes_stored(log_file_10):
    """Project stores roadway_changes DataFrame when provided."""
    df = Project.read_logfile(str(log_file_10))
    p = Project(roadway_changes=df)
    assert p.roadway_changes is df


@pytest.mark.roadway
def test_add_highway_changes_processes_boolean_link_change(stpaul_net, cube_parameters):
    """Boolean link changes use schema types instead of Parameters.bool_col."""
    base_links = stpaul_net.links_df
    link = base_links.loc[base_links["drive_access"].notna()].iloc[0]
    original_access = bool(link["drive_access"])
    changed_access = not original_access
    changes = pd.DataFrame(
        [
            {
                "OBJECT": "L",
                "OPERATION": "C",
                "GROUP": 0,
                "A": link["A"],
                "B": link["B"],
                "drive_access": changed_access,
            }
        ]
    )

    base_network = copy.copy(stpaul_net)
    base_network.links_df = base_links.copy()
    project = Project(roadway_changes=changes, parameters=cube_parameters)
    project.base_roadway_network = base_network

    result = project.add_highway_changes()

    assert len(result) == 1
    property_change = result[0]["roadway_property_change"]
    change = property_change["property_changes"]["drive_access"]
    assert change == {"existing": original_access, "set": changed_access}
    # 1 == True, so equality alone would not catch an int/bool mismatch
    assert type(change["existing"]) is bool
    assert type(change["set"]) is bool


def _link_change_project(stpaul_net, cube_parameters, base_links, changes):
    base_network = copy.copy(stpaul_net)
    base_network.links_df = base_links
    project = Project(roadway_changes=changes, parameters=cube_parameters)
    project.base_roadway_network = base_network
    return project


@pytest.mark.roadway
def test_compatibility_check_returns_renamed_changes(stpaul_net, cube_parameters, tmp_path):
    """Crosswalk renames are returned so Project.__init__ can keep them."""
    from cube_wrangler.parameters import Parameters

    pd.DataFrame({"log": ["LOGCOL"], "net": ["lanes_AM"]}).to_csv(
        tmp_path / "log_to_net.csv", index=False
    )
    pd.DataFrame({"dbf": ["DBFCOL"], "net": ["drive_access"]}).to_csv(
        tmp_path / "net_to_dbf.csv", index=False
    )
    parameters = Parameters(settings_location=tmp_path)
    changes = pd.DataFrame(
        {
            "OBJECT": ["L"],
            "OPERATION": ["A"],
            "A": [1],
            "B": [2],
            "LOGCOL": [3],
            "model_node_id": [0],
        }
    )

    result = Project.determine_roadway_network_changes_compatibility(
        stpaul_net, changes, parameters
    )

    assert "lanes_AM" in result.columns
    assert "LOGCOL" in changes.columns


@pytest.mark.roadway
def test_add_highway_changes_uses_first_duplicate_link(stpaul_net, cube_parameters):
    """With duplicate (A, B) links, the first one is the base link, as before."""
    links = stpaul_net.links_df.copy()
    first = links.loc[links["drive_access"].notna()].iloc[[0]]
    duplicate = first.copy()
    duplicate["drive_access"] = not bool(first["drive_access"].iloc[0])
    links = pd.concat([links, duplicate], ignore_index=True)
    row = first.iloc[0]
    changes = pd.DataFrame(
        [
            {
                "OBJECT": "L",
                "OPERATION": "C",
                "GROUP": 0,
                "A": row["A"],
                "B": row["B"],
                "drive_access": bool(duplicate["drive_access"].iloc[0]),
            }
        ]
    )
    project = _link_change_project(stpaul_net, cube_parameters, links, changes)

    result = project.add_highway_changes()

    change = result[0]["roadway_property_change"]["property_changes"]["drive_access"]
    assert change["existing"] is bool(row["drive_access"])


@pytest.mark.roadway
def test_add_highway_changes_fills_string_dtype_columns(stpaul_net, cube_parameters):
    """Missing values in StringDtype columns are filled with '' rather than 0."""
    links = stpaul_net.links_df.copy()
    link = links.iloc[0]
    links["name"] = links["name"].astype("string")
    links.loc[links.index[0], "name"] = pd.NA
    changes = pd.DataFrame(
        [
            {
                "OBJECT": "L",
                "OPERATION": "C",
                "GROUP": 0,
                "A": link["A"],
                "B": link["B"],
                "name": "New Street",
            }
        ]
    )
    project = _link_change_project(stpaul_net, cube_parameters, links, changes)

    result = project.add_highway_changes()

    change = result[0]["roadway_property_change"]["property_changes"]["name"]
    assert change == {"existing": "", "set": "New Street"}
