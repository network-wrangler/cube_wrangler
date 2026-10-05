# Performance Analysis

This document records bottlenecks identified by the benchmark suite and proposed
improvements to address them.  Improvements should be implemented in a dedicated
issue/branch, not in the benchmark branch itself.

---

## Identified Bottlenecks

### 1. Full-network boolean mask scan per change row (`project.py: _process_link_changes`)

**Location:** `evaluate_changes` → nested `_process_link_changes` → `_process_single_link_change`

**Pattern:**
```python
for _idx, row in cube_change_df.iterrows():
    base_df = base_links_df[
        (base_links_df["A"] == row["A"]) & (base_links_df["B"] == row["B"])
    ].copy()
```

**Complexity:** O(N_network × N_changes) — scans the full network once per change row.

**Proposed fix:** Build a `MultiIndex` on `(A, B)` once before the loop:
```python
ab_index = base_links_df.set_index(["A", "B"])
for _idx, row in cube_change_df.iterrows():
    base_row = ab_index.loc[(row["A"], row["B"])]
```
Reduces to O(N_network + N_changes).

---

### 2. Growing `pd.concat` inside `iterrows` loop (`project.py: _process_link_changes`)

**Pattern:**
```python
result_df = pd.DataFrame(...)
for ...:
    result_df = pd.concat([result_df, card_df], ...)
```

**Complexity:** O(N_changes²) in time and memory — each concat copies all prior rows.

**Proposed fix:** Collect frames in a list, concat once after the loop:
```python
frames = []
for ...:
    frames.append(card_df)
return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(...)
```

---

### 3. `prop_for_scope` called once per (property, timeperiod, category) combination

**Location:** `roadway.split_properties_by_time_period_and_category` calls
`network_wrangler.roadway.links.scopes.prop_for_scope` ~35 times for a typical
network (7 properties × 5 time periods).

**Complexity:** Each call scans the full links DataFrame to resolve scoped values.

**Proposed fix:** Profile whether batching or caching scope resolution reduces
the per-call overhead.  This may require a change in `network_wrangler` itself.

**Note:** Benchmarking `prop_for_scope` requires a properly-loaded
`RoadwayNetwork.links_df`; it cannot be isolated using raw JSON test data.
If a benchmark for this bottleneck is needed, add it to the `network_wrangler`
test suite or provide a pre-loaded fixture via `load_roadway_from_dir`.

---

## Benchmark Baseline

Run `pytest tests/test_benchmark.py -m benchmark --benchmark-save=baseline` to
capture a baseline before implementing any of the above fixes.  Then re-run with
`--benchmark-compare=baseline` to measure improvement.
