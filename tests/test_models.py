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
    assert pd.api.types.is_integer_dtype(result["lanes_AM"])


@pytest.mark.roadway
def test_coerce_links_bool_columns():
    """Boolean columns are coerced from int/string."""
    df = pd.DataFrame({"drive_access": [1, 0], "walk_access": ["True", "False"]})
    result = coerce_df_to_model(df, CubeLinksTable)
    assert pd.api.types.is_bool_dtype(result["drive_access"])
    assert pd.api.types.is_bool_dtype(result["walk_access"])
    assert result["walk_access"].tolist() == [True, False]


@pytest.mark.roadway
def test_coerce_links_numeric_string_booleans():
    """Numeric strings "0"/"1" map to False/True instead of both becoming True."""
    df = pd.DataFrame({"drive_access": ["0", "1", " 1 "]})
    result = coerce_df_to_model(df, CubeLinksTable)
    assert result["drive_access"].tolist() == [False, True, True]


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


@pytest.mark.roadway
def test_convert_types_fills_blank_and_nan_like_legacy_functions():
    """NaN/blank integer, float and bool values become 0/False before coercion."""
    from types import SimpleNamespace

    from cube_wrangler.roadway import convert_bool, convert_int, convert_types

    def make_net(lanes=("", float("nan")), access=("", "1")):
        links = pd.DataFrame(
            {
                "A": [1, 2],
                "B": [2, 3],
                "lanes_AM": list(lanes),
                "distance": [float("nan"), 2.5],
                "drive_access": list(access),
            }
        )
        return SimpleNamespace(links_df=links, nodes_df=pd.DataFrame({"model_node_id": [1]}))

    net = convert_types(make_net())
    assert net.links_df["lanes_AM"].tolist() == [0, 0]
    assert net.links_df["distance"].tolist() == [0.0, 2.5]
    assert net.links_df["drive_access"].tolist() == [False, True]

    # the narrower legacy wrappers only fill the columns they are responsible for
    net = convert_int(make_net(access=("1", "1")), int_col_names=["lanes_AM"])
    assert net.links_df["lanes_AM"].tolist() == [0, 0]
    assert net.links_df["distance"].isna().tolist() == [True, False]
    net = convert_bool(make_net(lanes=(1, 2)), bool_col_names=["drive_access"])
    assert net.links_df["drive_access"].tolist() == [False, True]
    assert net.links_df["distance"].isna().tolist() == [True, False]


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
