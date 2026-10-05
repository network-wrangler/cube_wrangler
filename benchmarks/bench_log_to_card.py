"""Benchmark: Cube log file → project card conversion.

Instruments each stage of the pipeline individually so that bottlenecks
can be measured without noise from unrelated steps.

Usage:
    # 1. Generate test log files first (if tests/data/changes_*.log don't exist):
    python benchmarks/generate_test_logfile.py

    # 2. Run the benchmark:
    python benchmarks/bench_log_to_card.py

    # 3. For a cProfile flamegraph on the slowest size:
    python -m cProfile -o benchmarks/profile_changes_500.prof \
        benchmarks/bench_log_to_card.py --sizes 500
    snakeviz benchmarks/profile_changes_500.prof

Optional env vars:
    LINK_JSON   path to link.json  (default: stpaul example)
    SIZES       comma-separated list of log sizes to benchmark (default: 10,50,100,500)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from csv import reader
from pathlib import Path
from typing import List

import pandas as pd

# Make tests/utils importable without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

import contextlib

from tests.utils.link_changes import (
    changeable_cols,
    current_process_link_changes,
    improved_process_link_changes,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
STPAUL_DIR = REPO_ROOT.parent / "met_council_wrangler" / "examples" / "stpaul"
LINK_JSON = Path(os.environ.get("LINK_JSON", STPAUL_DIR / "link.json"))
TEST_DATA_DIR = REPO_ROOT / "tests" / "data"
DEFAULT_SIZES = [10, 50, 100, 500]


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


class Timer:
    """Simple context-manager timer."""

    def __init__(self, label: str, results: dict):
        self.label = label
        self.results = results

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        self.results[self.label] = elapsed
        print(f"  {self.label:<55} {elapsed:7.3f}s")


# ---------------------------------------------------------------------------
# Stage 1 – read_logfile  (replica of Project.read_logfile)
# ---------------------------------------------------------------------------


def benchmark_read_logfile(log_path: Path) -> pd.DataFrame:
    """Parse a Cube log file into a DataFrame."""
    with open(log_path) as f:
        content = f.readlines()

    link_lines = [x.strip().replace(";", ",") for x in content if x.startswith("L")]
    node_lines = [x.strip().replace(";", ",") for x in content if x.startswith("N")]

    def split_log(x):
        return next(iter(reader([x], delimiter=",", quotechar='"')))

    nodecol = ["OBJECT", "OPERATION", "GROUP", *node_lines[0].split(",")[1:]]
    linkcol = ["OBJECT", "OPERATION", "GROUP", *link_lines[0].split(",")[1:]]

    node_df = pd.DataFrame([split_log(x) for x in node_lines[1:]], columns=nodecol)
    link_df = pd.DataFrame([split_log(x) for x in link_lines[1:]], columns=linkcol)
    log_df = pd.concat([link_df, node_df], ignore_index=True, sort=False)
    log_df.columns = [c.split("[")[0] for c in log_df.columns]
    return log_df


# ---------------------------------------------------------------------------
# Stage 2 – consolidate_actions  (replica of _consolidate_actions)
# ---------------------------------------------------------------------------


def benchmark_consolidate_actions(
    log_df: pd.DataFrame, base_links_df: pd.DataFrame
) -> pd.DataFrame:
    """Group log rows by (A,B), determine final operation, return consolidated df."""
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
    link_df = pd.merge(link_df, history, on=["A", "B"], how="left")
    link_df.drop_duplicates(subset=["A", "B"], keep="last", inplace=True)

    def _final_op(x):
        if x.OPERATION_history[-1] == "D":
            return "N" if "A" in x.OPERATION_history[:-1] else "D"
        if x.OPERATION_history[-1] == "A":
            return "C" if "D" in x.OPERATION_history[:-1] else "A"
        return "A" if "A" in x.OPERATION_history[:-1] else "C"

    link_df["OPERATION_final"] = link_df.apply(lambda x: _final_op(x), axis=1)
    return link_df[[*cols, "OPERATION_final"]]


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(sizes: List[int]):
    print(f"\nLoading base network from {LINK_JSON} …")
    t0 = time.perf_counter()
    base_links_df = pd.read_json(LINK_JSON)
    load_time = time.perf_counter() - t0
    print(f"  Loaded {len(base_links_df):,} links in {load_time:.2f}s\n")

    summary_rows = []

    for n in sizes:
        log_path = TEST_DATA_DIR / f"changes_{n}.log"
        if not log_path.exists():
            print(f"  SKIP {log_path} (run generate_test_logfile.py first)")
            continue

        print(f"{'=' * 70}")
        print(f"  Log size: {n} changes  ({log_path.name})")
        print(f"{'=' * 70}")

        results: dict = {}

        with Timer("1. read_logfile()", results):
            log_df = benchmark_read_logfile(log_path)

        with Timer("2. consolidate_actions()", results):
            consolidated_df = benchmark_consolidate_actions(log_df, base_links_df)

        change_df = consolidated_df[consolidated_df["OPERATION_final"] == "C"].copy()
        print(f"     → {len(change_df)} change rows after consolidation")

        cols = changeable_cols(consolidated_df, base_links_df)

        with Timer("3a. CURRENT  _process_link_changes (iterrows + full scan)", results):
            result_current = current_process_link_changes(change_df, base_links_df, cols)

        with Timer(
            "3b. IMPROVED _process_link_changes (set_index + collect-concat)", results
        ):
            result_improved = improved_process_link_changes(change_df, base_links_df, cols)

        key_current = "3a. CURRENT  _process_link_changes (iterrows + full scan)"
        key_improved = "3b. IMPROVED _process_link_changes (set_index + collect-concat)"
        speedup = results[key_current] / max(results[key_improved], 1e-6)
        print(f"\n  Speedup (3a/3b): {speedup:.1f}x")
        print(
            f"  Result rows — current: {len(result_current)}, improved: {len(result_improved)}\n"
        )

        summary_rows.append(
            {
                "n_changes": n,
                **{k: round(v, 4) for k, v in results.items()},
                "speedup_3a_vs_3b": round(speedup, 1),
            }
        )

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    summary_df = pd.DataFrame(summary_rows).set_index("n_changes")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(summary_df.to_string())
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default=",".join(str(s) for s in DEFAULT_SIZES),
        help="Comma-separated list of log sizes to benchmark",
    )
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    run_benchmark(sizes)
