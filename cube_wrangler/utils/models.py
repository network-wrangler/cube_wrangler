"""Utilities for coercing DataFrames to Pandera DataFrameModel schemas."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
from pandera.errors import SchemaError, SchemaErrors
from pandera.pandas import DataFrameModel

from ..logger import WranglerLogger

_SMALL_RECS = 20


def _convert_string_dtype_to_object(df: pd.DataFrame) -> pd.DataFrame:
    """Convert StringDtype columns to object dtype for pandera compatibility.

    Fixes compatibility issues with pandas 2.2+ StringDtype and
    numpy.issubdtype in Python 3.11+.
    """
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype(object)
    return df


def coerce_df_to_model(
    df: pd.DataFrame,
    model: type[DataFrameModel],
    output_file: Path = Path("coercion_failure_cases.csv"),
) -> pd.DataFrame:
    """Coerce DataFrame column types to match a Pandera DataFrameModel schema.

    Only columns present in the DataFrame are coerced; columns absent from the
    DataFrame are left absent (all schema columns should be declared
    ``Optional`` for this to work correctly).  ``df.attrs`` is preserved
    across the operation.

    Args:
        df: DataFrame whose columns should be coerced.
        model: Pandera DataFrameModel with ``Config.coerce = True``.
        output_file: Path to write detailed failure cases when many errors
            occur. Defaults to ``coercion_failure_cases.csv``.

    Returns:
        DataFrame with columns coerced to their declared schema types.

    Raises:
        ValueError: When coercion fails and cannot be recovered.
    """
    attrs = copy.deepcopy(df.attrs)
    df = _convert_string_dtype_to_object(df)

    try:
        result = model.validate(df, lazy=True)
        result.attrs = attrs
        return result
    except SchemaErrors as exc:
        WranglerLogger.error(
            "Coercion to %s failed with %d error(s):\n%s",
            model.__name__,
            len(exc.failure_cases),
            exc.failure_cases,
        )
        if len(exc.failure_cases) > _SMALL_RECS:
            exc.failure_cases.to_csv(output_file)
            WranglerLogger.info("Detailed failure cases written to %s", output_file)
        msg = f"Coercion to {model.__name__} failed."
        raise ValueError(msg) from exc
    except SchemaError as exc:
        WranglerLogger.error(
            "Coercion to %s failed: %s\n%s",
            model.__name__,
            exc,
            exc.failure_cases,
        )
        msg = f"Coercion to {model.__name__} failed."
        raise ValueError(msg) from exc
