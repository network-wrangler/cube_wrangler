# Performance Analysis

This document records bottlenecks identified by the benchmark suite and proposed
improvements to address them.  See `PERFORMANCE_ANALYSIS.md` at the repo root for
full timings, implementation details, and status.

---

## Status summary

| # | Bottleneck | Status | Branch |
|---|-----------|--------|--------|
| 1 | Full-network boolean mask scan per change row (`_process_single_link_change`) | ✅ Done | `feature/perf-link-changes` |
| 2 | Growing `pd.concat` inside `iterrows` loop (`_process_link_changes`) | ✅ Done | `feature/perf-link-changes` |
| 3 | Debug logging string formatting in inner loop | ✅ Done | `feature/perf-link-changes` |
| 4 | `prop_for_scope` called once per (property × timeperiod × category) | 🔴 Next | needs `network_wrangler` PR |
| 5 | Network loaded from geojson instead of parquet | 🟡 Operational | no code change needed |
| 6 | `validate_df_to_model` called on every `prop_for_scope` invocation | 🟡 Follow-on | `network_wrangler` |

---

## Next tier: `prop_for_scope` batching (Issue 4) ★★★

`split_properties_by_time_period_and_category` calls `prop_for_scope` once per
(property × time period × category) combination — 35+ calls on a MetCouncil network.
Each call re-validates the full DataFrame and re-explodes the scoped column from scratch,
even though `_create_exploded_df_for_scoped_prop` returns the same result for every call
on the same property.

**Fix:** Add `props_for_scopes(links_df, prop_name, scopes)` to `network_wrangler` that
validates and explodes once, then filters for each scope. See `PERFORMANCE_ANALYSIS.md`
for the full API sketch and implementation plan.

**Estimated speedup:** ~10–30× for the `split_properties` step (10–45s → ~1–3s on production).

---

## Benchmark baseline

Run `pytest tests/test_benchmark.py -m benchmark --benchmark-save=baseline` to
capture a baseline before implementing any of the above fixes.  Then re-run with
`--benchmark-compare=baseline` to measure improvement.
