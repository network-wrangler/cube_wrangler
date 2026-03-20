"""Benchmark tests for the Cube log to project card pipeline.

These tests are marked with ``benchmark`` and skipped in the default CI run.
To run benchmarks explicitly::

    pytest tests/test_benchmark.py -m benchmark
    pytest tests/test_benchmark.py -m benchmark --benchmark-histogram

To compare against main or a saved baseline::

    pytest tests/test_benchmark.py -m benchmark --benchmark-save=my_branch
    pytest-benchmark compare main my_branch

Bottlenecks to address in a future improvement issue:
  see PERFORMANCE_ANALYSIS.md
"""

from __future__ import annotations

from pathlib import Path

import pytest
from utils.link_changes import (
    changeable_cols,
    current_process_link_changes,
)

pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_logfile(log_path: Path):
    """Thin wrapper around Project.read_logfile for use as a benchmark callable."""
    from cube_wrangler.project import Project

    return Project.read_logfile(str(log_path))


# ---------------------------------------------------------------------------
# Stage 1: read_logfile
# ---------------------------------------------------------------------------


class TestBenchmarkReadLogfile:
    """Benchmark Project.read_logfile() at different log sizes."""

    def test_read_logfile_50(self, benchmark, log_file_50):
        """Benchmark reading a 50-row log file."""
        result = benchmark(_read_logfile, log_file_50)
        assert len(result) >= 50

    def test_read_logfile_100(self, benchmark, log_file_100):
        """Benchmark reading a 100-row log file."""
        result = benchmark(_read_logfile, log_file_100)
        assert len(result) >= 100

    def test_read_logfile_500(self, benchmark, log_file_500):
        """Benchmark reading a 500-row log file."""
        result = benchmark(_read_logfile, log_file_500)
        assert len(result) >= 500


# ---------------------------------------------------------------------------
# Stage 2: _process_link_changes — current baseline
# ---------------------------------------------------------------------------


def _link_rows_from_log(log_path: Path, base_links_df):
    """Return link-change rows from a log file, cast to match base network dtypes."""
    import contextlib

    from cube_wrangler.project import Project

    log_df = Project.read_logfile(str(log_path))
    link_df = log_df[log_df["OBJECT"] == "L"].copy()

    for c in [col for col in link_df.columns if col in base_links_df.columns]:
        with contextlib.suppress(Exception):
            link_df[c] = link_df[c].astype(base_links_df[c].dtype)

    return link_df


class TestBenchmarkLinkChanges:
    """Benchmark the current O(N_net x N_changes) link-change processing loop."""

    @pytest.fixture(scope="class")
    def change_100(self, log_file_100, stpaul_links_df):
        """Prepared link-change rows and changeable columns for 100-row log."""
        link_df = _link_rows_from_log(log_file_100, stpaul_links_df)
        cols = changeable_cols(link_df, stpaul_links_df)
        return link_df, cols

    @pytest.fixture(scope="class")
    def change_500(self, log_file_500, stpaul_links_df):
        """Prepared link-change rows and changeable columns for 500-row log."""
        link_df = _link_rows_from_log(log_file_500, stpaul_links_df)
        cols = changeable_cols(link_df, stpaul_links_df)
        return link_df, cols

    def test_current_100(self, benchmark, change_100, stpaul_links_df):
        """Current implementation with 100 change rows."""
        df, cols = change_100
        result = benchmark(current_process_link_changes, df, stpaul_links_df, cols)
        assert len(result) >= 0

    def test_current_500(self, benchmark, change_500, stpaul_links_df):
        """Current implementation with 500 change rows."""
        df, cols = change_500
        result = benchmark(current_process_link_changes, df, stpaul_links_df, cols)
        assert len(result) >= 0


