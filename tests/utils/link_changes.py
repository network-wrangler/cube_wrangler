"""Reference implementation of _process_link_changes for benchmarking.

Mirrors the logic in cube_wrangler/project.py so the benchmark can measure
the link-change processing loop in isolation, without a fully-wired Project.

To compare performance across branches, save a baseline on one branch and
compare on another:

    # on feature/9-modernize-design-patterns
    pytest tests/test_benchmark.py -m benchmark --benchmark-save=baseline

    # on feature/perf-link-changes
    pytest tests/test_benchmark.py -m benchmark --benchmark-compare=baseline
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
    """Mirror of _process_link_changes from cube_wrangler/project.py.

    Uses a pre-built (A, B) dict for O(1) per-row lookup and collects result
    frames in a list before a single final concat — O(N_network + N_changes).
    """
    # Build (A, B) → positional index once — O(N_network)
    ab_lookup: dict[tuple, int] = {
        (int(a), int(b)): i
        for i, (a, b) in enumerate(
            zip(base_links_df["A"], base_links_df["B"], strict=True)
        )
    }

    card_frames: list[pd.DataFrame] = []
    for _idx, row in cube_change_df.iterrows():
        link_idx = ab_lookup.get((int(row["A"]), int(row["B"])))
        if link_idx is None:
            continue
        base_row = base_links_df.iloc[link_idx]

        changed = _detect_changes(row, base_row, cols)
        if changed:
            card_frames.append(
                pd.DataFrame(
                    {
                        "properties": [_build_property_dict(row, base_row, changed)],
                        "model_link_id": [base_row["model_link_id"]],
                    }
                )
            )

    if not card_frames:
        return pd.DataFrame(columns=["properties", "model_link_id"])
    return pd.concat(card_frames, ignore_index=True, sort=False)
