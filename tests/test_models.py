"""Tests for cube_wrangler.utils.models and cube_wrangler.models.tables."""

from __future__ import annotations

import pandas as pd
import pytest

from cube_wrangler.models.tables import CubeLinksTable, CubeNodesTable
from cube_wrangler.utils.models import coerce_df_to_model

# ---------------------------------------------------------------------------
# CubeLinksTable coercion
# ---------------------------------------------------------------------------


@pytest.mark.roadway
def test_coerce_links_int_columns():
    """Integer columns are coerced from string/float."""
    df = pd.DataFrame({"A": ["1", "2"], "B": ["3", "4"], "lanes_AM": [2.0, 3.0]})
    result = coerce_df_to_model(df, CubeLinksTable)
    assert result["A"].dtype == int
    assert result["B"].dtype == int
    assert result["lanes_AM"].dtype == int


@pytest.mark.roadway
def test_coerce_links_bool_columns():
    """Boolean columns are coerced from int/string."""
    df = pd.DataFrame({"drive_access": [1, 0], "walk_access": ["True", "False"]})
    result = coerce_df_to_model(df, CubeLinksTable)
    assert result["drive_access"].dtype == bool
    assert result["walk_access"].dtype == bool


@pytest.mark.roadway
def test_coerce_links_float_columns():
    """Float columns are coerced from int."""
    df = pd.DataFrame({"distance": [100, 200]})
    result = coerce_df_to_model(df, CubeLinksTable)
    assert result["distance"].dtype == float


@pytest.mark.roadway
def test_coerce_links_preserves_extra_columns():
    """Columns not declared in the schema are preserved as-is."""
    df = pd.DataFrame({"A": [1], "my_custom_col": ["hello"]})
    result = coerce_df_to_model(df, CubeLinksTable)
    assert "my_custom_col" in result.columns
    assert result["my_custom_col"].iloc[0] == "hello"


@pytest.mark.roadway
def test_coerce_links_preserves_attrs():
    """df.attrs is preserved through coercion."""
    df = pd.DataFrame({"A": [1, 2]})
    df.attrs["source"] = "test"
    result = coerce_df_to_model(df, CubeLinksTable)
    assert result.attrs.get("source") == "test"


@pytest.mark.roadway
def test_coerce_links_missing_columns_ok():
    """DataFrame with only a subset of schema columns validates without error."""
    df = pd.DataFrame({"model_link_id": [1, 2, 3]})
    result = coerce_df_to_model(df, CubeLinksTable)
    assert list(result.columns) == ["model_link_id"]


@pytest.mark.roadway
def test_coerce_links_invalid_raises():
    """Non-coercible values raise ValueError."""
    df = pd.DataFrame({"A": ["not_an_int", "also_bad"]})
    with pytest.raises(ValueError, match="Coercion to CubeLinksTable failed"):
        coerce_df_to_model(df, CubeLinksTable)


# ---------------------------------------------------------------------------
# CubeNodesTable coercion
# ---------------------------------------------------------------------------


@pytest.mark.roadway
def test_coerce_nodes_int_columns():
    """model_node_id is coerced to int."""
    df = pd.DataFrame({"model_node_id": ["10", "20"], "X": [1.0, 2.0]})
    result = coerce_df_to_model(df, CubeNodesTable)
    assert result["model_node_id"].dtype == int
    assert result["X"].dtype == float


@pytest.mark.roadway
def test_coerce_nodes_missing_columns_ok():
    """Partial node DataFrame validates cleanly."""
    df = pd.DataFrame({"N": [1, 2]})
    result = coerce_df_to_model(df, CubeNodesTable)
    assert "N" in result.columns


# ---------------------------------------------------------------------------
# Schema import smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.roadway
def test_cube_links_table_importable():
    """CubeLinksTable can be imported and is a DataFrameModel."""
    from pandera.pandas import DataFrameModel

    assert issubclass(CubeLinksTable, DataFrameModel)


@pytest.mark.roadway
def test_cube_nodes_table_importable():
    """CubeNodesTable can be imported and is a DataFrameModel."""
    from pandera.pandas import DataFrameModel

    assert issubclass(CubeNodesTable, DataFrameModel)
