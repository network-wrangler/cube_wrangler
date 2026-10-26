# Performance Analysis

This document records bottlenecks identified by the benchmark suite and the
improvements made (or still proposed) to address them.  See `PERFORMANCE_ANALYSIS.md`
at the repo root for full timings, implementation details, and status.

---

## Status summary

| # | Bottleneck | Status | Branch |
|---|-----------|--------|--------|
| 1 | Full-network boolean mask scan per change row (`_process_single_link_change`) | ✅ Done | `feature/perf-link-changes` |
| 2 | Growing `pd.concat` inside `iterrows` loop (`_process_link_changes`) | ✅ Done | `feature/perf-link-changes` |
| 3 | Debug logging string formatting in inner loop | ✅ Done | `feature/perf-link-changes` |
| 4 | `prop_for_scope` called once per (property × timeperiod × category) | ✅ Done | PR #12, merged from `feature/perf-props-for-scopes` (uses `network_wrangler.props_for_scopes`) |
| 5 | Network loaded from geojson instead of parquet | 🟡 Operational | no code change needed |
| 6 | `validate_df_to_model` called on every `prop_for_scope` invocation | 🟡 Follow-on | `network_wrangler` |
| 7 | `fill_na` uses `apply(lambda)` over full links_df | 🟢 Low | deprecated path; not on the critical path |

---

## Completed: `prop_for_scope` batching (Issue 4) ★★★

`split_properties_by_time_period_and_category` previously called `prop_for_scope` once per
(property × time period × category) combination — 35+ calls on a MetCouncil network.
Each call re-validated the full DataFrame and re-exploded the scoped column from scratch.

**Fix:** `network_wrangler` now provides `props_for_scopes(links_df, prop_name, scopes)`,
which validates and explodes once, then filters for each scope. `cube_wrangler/roadway.py`
calls it once per property. See `PERFORMANCE_ANALYSIS.md` for the original API sketch.

**Estimated speedup:** ~10–30× for the `split_properties` step (10–45s → ~1–3s on production).
Re-run the benchmarks to confirm measured numbers.

---

## Benchmark baseline

Items 1–4 are already implemented, so a true "before" baseline requires checking out
`feature/9-modernize-design-patterns` (see the root `PERFORMANCE_ANALYSIS.md`).  Run
`pytest tests/test_benchmark.py -m benchmark --benchmark-save=baseline` there, then re-run
on the current branch with `--benchmark-compare=baseline` to measure improvement.  For any
remaining item (5, 6), save a new baseline before making the change.
