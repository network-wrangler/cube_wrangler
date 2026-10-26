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


def _parse_string_booleans(df: pd.DataFrame, model: type[DataFrameModel]) -> pd.DataFrame:
    """Convert "True"/"False"/"1"/"0" strings to booleans in columns the model declares as bool.

    Pandas cannot cast strings to nullable booleans, and ``astype(bool)`` would
    treat every non-empty string (including "False") as True.
    """
    mapping = {"true": True, "false": False, "1": True, "0": False}
    for col, column in model.to_schema().columns.items():
        if col not in df.columns or not pd.api.types.is_bool_dtype(column.dtype.type):
            continue
        if df[col].dtype == object:
            parsed = df[col].map(
                lambda v: mapping.get(v.strip().lower(), v) if isinstance(v, str) else v
            )
            df[col] = parsed
    return df


def to_python_scalar(model: type[DataFrameModel], column: str, value: object) -> object:
    """Convert a value read from a coerced DataFrame to a plain Python scalar.

    Args:
        model: Pandera DataFrameModel declaring the column.
        column: Column name the value came from.
        value: Scalar read from a DataFrame already coerced to ``model``.

    Returns:
        The value as a plain Python scalar (so it is JSON/YAML serializable), or the
        column's declared default (possibly ``None``) when the value is missing.
    """
    if value is None or (pd.api.types.is_scalar(value) and bool(pd.isna(value))):
        schema_column = model.to_schema().columns.get(column)
        return None if schema_column is None else schema_column.default
    return value.item() if hasattr(value, "item") else value


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
    df = _parse_string_booleans(_convert_string_dtype_to_object(df), model)

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
