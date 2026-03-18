"""Reference implementations of _process_link_changes for benchmark comparison.

These mirror the logic in cube_wrangler/project.py so that the benchmark can
compare the current O(N_network × N_changes) pattern against the improved
O(N_network + N_changes) pattern without modifying production code.
"""

from __future__ import annotations

import numbers
from typing import Any

import pandas as pd

# Columns that are never treated as changed properties (mirrors Project.STATIC_VALUES).
STATIC_VALUES: set[str] = {"model_link_id", "area_type", "county", "centroidconnect"}


def changeable_cols(log_df: pd.DataFrame, base_links_df: pd.DataFrame) -> list[str]:
    """Return columns that appear in both *log_df* and *base_links_df* and are not static."""
    return list(
        (set(log_df.columns) & set(base_links_df.columns))
        - STATIC_VALUES
        - {"OBJECT", "OPERATION", "GROUP", "OPERATION_final", "OPERATION_history"}
    )


def _build_property_dict(row: Any, base_row: Any, changed: list[str]) -> list[dict]:
    """Build the property change list for a single link."""
    result = []
    for c in changed:
        _d: dict = {"property": c, "set": row[c]}
        val = base_row[c]
        if val is not None:
            if isinstance(val, numbers.Integral):
                _d["existing"] = int(val)
            elif isinstance(val, numbers.Real):
                _d["existing"] = float(val)
            else:
                _d["existing"] = val
        result.append(_d)
    return result


def _detect_changes(row: Any, base_row: Any, cols: list[str]) -> list[str]:
    """Return columns in *cols* whose value differs between *row* and *base_row*."""
    changed = []
    for col in cols:
        if col not in row.index or col not in base_row.index:
            continue
        if str(row[col]).strip("\"'").replace(".0", "") == str(base_row[col]).strip(
            "\"'"
        ).replace(".0", ""):
            continue
        changed.append(col)
    return changed


def current_process_link_changes(
    cube_change_df: pd.DataFrame,
    base_links_df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """Current implementation: iterrows + full-network boolean mask per row.

    O(N_network × N_changes) — this is the bottleneck identified in the
    performance analysis.
    """
    result_df = pd.DataFrame(columns=["properties", "model_link_id"])

    for _idx, row in cube_change_df.iterrows():
        # Full-network scan per row — BOTTLENECK 1
        base_df = base_links_df[
            (base_links_df["A"] == row["A"]) & (base_links_df["B"] == row["B"])
        ].copy()
        if base_df.empty:
            continue
        base_row = base_df.iloc[0]

        changed = _detect_changes(row, base_row, cols)
        if not changed:
            card_df = pd.DataFrame()
        else:
            card_df = pd.DataFrame(
                {
                    "properties": [_build_property_dict(row, base_row, changed)],
                    "model_link_id": [base_row["model_link_id"]],
                }
            )
        # Growing concat per row — BOTTLENECK 2
        result_df = pd.concat([result_df, card_df], ignore_index=True, sort=False)

    return result_df


def improved_process_link_changes(
    cube_change_df: pd.DataFrame,
    base_links_df: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """Improved implementation: pre-built MultiIndex lookup + collect-then-concat.

    O(N_network + N_changes) — the set_index cost is paid once before the loop.
    """
    ab_index = base_links_df.set_index(["A", "B"])
    frames: list[pd.DataFrame] = []

    for _idx, row in cube_change_df.iterrows():
        try:
            base_row = ab_index.loc[(row["A"], row["B"])]
        except KeyError:
            continue
        if isinstance(base_row, pd.DataFrame):
            base_row = base_row.iloc[0]

        changed = _detect_changes(row, base_row, cols)
        if not changed:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "properties": [_build_property_dict(row, base_row, changed)],
                    "model_link_id": [base_row["model_link_id"]],
                }
            )
        )

    if not frames:
        return pd.DataFrame(columns=["properties", "model_link_id"])
    return pd.concat(frames, ignore_index=True, sort=False)
