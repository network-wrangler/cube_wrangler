"""Benchmark tests for the Cube log → project card pipeline.

These tests are marked with `benchmark` and are skipped in the default CI run.
To run benchmarks explicitly:

    uv run pytest tests/test_benchmark.py -m benchmark
    uv run pytest tests/test_benchmark.py -m benchmark --benchmark-histogram

To compare branches:

    uv run pytest tests/test_benchmark.py -m benchmark --benchmark-save=branch_name
    pytest-benchmark compare branch_a branch_b

Bottlenecks identified in PERFORMANCE_ANALYSIS.md:
  1. Full-network boolean mask scan per change row   (O(N_net × N_changes))
  2. pd.concat inside iterrows loop                  (O(N_changes²))
  3. prop_for_scope called once per (prop, timeperiod, category) combination
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
from utils.link_changes import (
    changeable_cols,
    current_process_link_changes,
    improved_process_link_changes,
)

pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_logfile(log_path: Path):
    """Thin wrapper around Project.read_logfile for use as a benchmark callable."""
    from cube_wrangler.project import Project

    return Project.read_logfile(str(log_path))


def _parse_change_df(log_path: Path, base_links_df):
    """Read log, consolidate actions, return only Change rows + changeable_col."""
    import pandas as pd

    from cube_wrangler.project import Project

    log_df = Project.read_logfile(str(log_path))
    link_df = log_df[log_df["OBJECT"] == "L"].copy()

    cols = [c for c in link_df.columns if c in base_links_df.columns]
    for c in cols:
        with contextlib.suppress(Exception):
            link_df[c] = link_df[c].astype(base_links_df[c].dtype)

    history = (
        link_df.groupby(["A", "B"])["OPERATION"]
        .agg(lambda x: x.tolist())
        .rename("OPERATION_history")
        .reset_index()
    )
    link_df = link_df.merge(history, on=["A", "B"], how="left")
    link_df.drop_duplicates(subset=["A", "B"], keep="last", inplace=True)

    def _final_op(x):
        if x.OPERATION_history[-1] == "D":
            return "N" if "A" in x.OPERATION_history[:-1] else "D"
        if x.OPERATION_history[-1] == "A":
            return "C" if "D" in x.OPERATION_history[:-1] else "A"
        return "A" if "A" in x.OPERATION_history[:-1] else "C"

    link_df["OPERATION_final"] = link_df.apply(_final_op, axis=1)
    change_df = link_df[link_df["OPERATION_final"] == "C"].copy()
    cols = changeable_cols(link_df, base_links_df)
    return change_df, cols


# ---------------------------------------------------------------------------
# Stage 1: read_logfile
# ---------------------------------------------------------------------------


class TestBenchmarkReadLogfile:
    """Benchmark Project.read_logfile() at different log sizes."""

    def test_read_logfile_50(self, benchmark, log_file_50):
        result = benchmark(_read_logfile, log_file_50)
        assert len(result) >= 50

    def test_read_logfile_100(self, benchmark, log_file_100):
        result = benchmark(_read_logfile, log_file_100)
        assert len(result) >= 100

    def test_read_logfile_500(self, benchmark, log_file_500):
        result = benchmark(_read_logfile, log_file_500)
        assert len(result) >= 500


# ---------------------------------------------------------------------------
# Stage 2: _process_link_changes — current vs improved
# ---------------------------------------------------------------------------


class TestBenchmarkLinkChanges:
    """Compare current O(N_net × N_changes) vs improved O(N_net + N_changes)."""

    @pytest.fixture(scope="class")
    def change_100(self, log_file_100, stpaul_links_df):
        return _parse_change_df(log_file_100, stpaul_links_df)

    @pytest.fixture(scope="class")
    def change_500(self, log_file_500, stpaul_links_df):
        return _parse_change_df(log_file_500, stpaul_links_df)

    # --- 100 changes ---

    def test_current_100(self, benchmark, change_100, stpaul_links_df):
        """Current implementation with 100 change rows."""
        df, cols = change_100
        result = benchmark(current_process_link_changes, df, stpaul_links_df, cols)
        assert len(result) >= 0

    def test_improved_100(self, benchmark, change_100, stpaul_links_df):
        """Improved implementation with 100 change rows."""
        df, cols = change_100
        result = benchmark(improved_process_link_changes, df, stpaul_links_df, cols)
        assert len(result) >= 0

    # --- 500 changes ---

    def test_current_500(self, benchmark, change_500, stpaul_links_df):
        """Current implementation with 500 change rows."""
        df, cols = change_500
        result = benchmark(current_process_link_changes, df, stpaul_links_df, cols)
        assert len(result) >= 0

    def test_improved_500(self, benchmark, change_500, stpaul_links_df):
        """Improved implementation with 500 change rows."""
        df, cols = change_500
        result = benchmark(improved_process_link_changes, df, stpaul_links_df, cols)
        assert len(result) >= 0


# ---------------------------------------------------------------------------
# Stage 3: prop_for_scope (the split_properties bottleneck)
# ---------------------------------------------------------------------------


class TestBenchmarkPropForScope:
    """Benchmark prop_for_scope, which is called 35+ times during split_properties."""

    @pytest.fixture(scope="class")
    def links_df(self, stpaul_links_df):
        return stpaul_links_df

    def test_prop_for_scope_lanes_am(self, benchmark, links_df):
        """Single prop_for_scope call — representative of the 35 calls in split_properties."""
        from network_wrangler.roadway.links.scopes import prop_for_scope

        result = benchmark(
            prop_for_scope, links_df, "lanes", timespan=["6:00", "10:00"]
        )
        assert result is not None
