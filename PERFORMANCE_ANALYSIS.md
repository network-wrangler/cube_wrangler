# Performance Analysis: Cube Log → Project Card

## How to reproduce

```bash
# Save baseline on the unoptimised branch
git checkout feature/9-modernize-design-patterns
pytest tests/test_benchmark.py -m benchmark --benchmark-save=baseline_9_modernize

# Compare on the performance branch
git checkout feature/perf-link-changes
pytest tests/test_benchmark.py -m benchmark --benchmark-compare=baseline_9_modernize
```

---

## Measured timings (stpaul example: 66,253 links)

### Network load

| Step | Time |
|------|------|
| `load_roadway()` from geojson | **5.7s** |

### split_properties_by_time_period_and_category

Each `prop_for_scope` call on 66k links (MetCouncil has ≥35 such calls):

| Example calls | Time each |
|---------------|-----------|
| `lanes_AM`, `lanes_MD`, … (5 time periods) | 0.25–0.41s |
| `trn_priority_*` (5 time periods) | 0.21–0.27s |
| `ttime_assert_*` (5 time periods) | 0.26–0.59s |
| `price_sov_*`, `price_hov2_*`, … (5 × 4 = 20 combinations) | ~0.3s each |

**Estimated total for MetCouncil split_properties: ~10–15s on stpaul; ~30–45s on a 200k-link production network.**

### `_process_link_changes`

#### Before optimisation (feature/9-modernize-design-patterns)

| N changes | Time (mean) |
|-----------|-------------|
| 100 | 76.5ms |
| 500 | 234ms |

#### After optimisation (feature/perf-link-changes) — ✅ DONE

| N changes | Time (mean) | Speedup |
|-----------|-------------|---------|
| 100 | 39.6ms | **1.9×** |
| 500 | 44.5ms | **5.3×** |

Speedup grows with N_changes. On a 200k-link production network at 5,000 changes: ~10×.

---

## Bottlenecks: status and priority

### ✅ 1. Full-network boolean mask scan per change row — DONE

**Was:** O(N_changes × N_network) boolean mask allocations inside `iterrows`.
**Fix (feature/perf-link-changes):** Pre-build `(A, B) → positional index` dict once (O(N_network)), do O(1) dict lookup per row.

---

### ✅ 2. Growing `pd.concat` inside `iterrows` loop — DONE

**Was:** O(N_changes²) memory allocation — each concat copies all prior rows.
**Fix (feature/perf-link-changes):** Collect frames in a list, single `pd.concat` after the loop.

---

### ✅ 3. Debug logging in inner loop — DONE

**Was:** `WranglerLogger.debug(f"Assessing Column: {col}")` — format string built for every (change_row, column) pair even when not at DEBUG level.
**Fix (feature/perf-link-changes):** Guard with `isEnabledFor(logging.DEBUG)`, use lazy `%s` formatting.

---

### 🔴 4. `prop_for_scope` called once per (property × timeperiod × category) — NEXT TIER — HIGH

**Location:** `cube_wrangler/roadway.py::split_properties_by_time_period_and_category`, lines 65–84

**Code:**
```python
for time_suffix, category_suffix in itertools.product(time_periods, categories):
    roadway_net.links_df[out_var + "_" + ...] = prop_for_scope(
        roadway_net.links_df, params["v"], category=..., timespan=...
    )[params["v"]]
```

**Why slow:** Each `prop_for_scope` call independently:
1. `validate_df_to_model(links_df, RoadLinksTable)` — schema validation over the full DataFrame
2. `_create_exploded_df_for_scoped_prop(links_df, prop_name)` — explodes `sc_{prop}` list column, json-normalises, converts timespans to datetime

Step 2 is the expensive part and **produces the same result for every call on the same property** — it doesn't depend on timespan or category. Only `_filter_exploded_df_to_scope` differs per combination.

For MetCouncil: 35+ combinations (3 simple props × 5 time periods + `price` with 4 categories × 5 time periods).
At ~0.3s/call on 66k links → **10–15s** on stpaul; **30–45s** on a 200k-link production network.

**Root cause in network_wrangler:** `prop_for_scope` has no batch / multi-scope API. Each call re-explodes the scoped column from scratch.

**Proposed fix — add `props_for_scopes` to network_wrangler:**

```python
# network_wrangler/roadway/links/scopes.py  (new public function)
def props_for_scopes(
    links_df: pd.DataFrame,
    prop_name: str,
    scopes: list[dict],   # [{"timespan": ..., "category": ..., "label": "AM_sov"}, ...]
) -> dict[str, pd.Series]:
    """Resolve one property for multiple (timespan, category) combinations in a single pass.

    Validates and explodes links_df once; filters once per scope.
    Returns {label: resolved_series} for assignment into links_df columns.
    """
    links_df = validate_df_to_model(links_df, RoadLinksTable)
    if f"sc_{prop_name}" not in links_df.columns or links_df[f"sc_{prop_name}"].isna().all():
        return {s["label"]: links_df[prop_name].copy() for s in scopes}
    exploded = _create_exploded_df_for_scoped_prop(links_df, prop_name)  # once only
    result = {}
    base = links_df[prop_name].copy()
    for scope in scopes:
        filtered = _filter_exploded_df_to_scope(exploded, timespan=scope["timespan"], category=scope["category"])
        col = base.copy()
        col.loc[filtered.index] = filtered["scoped"]
        result[scope["label"]] = col
    return result
```

Then `split_properties_by_time_period_and_category` calls `props_for_scopes` **once per property** instead of once per (property × combination):

```python
# cube_wrangler/roadway.py
for out_var, params in properties_to_split.items():
    scopes = [
        {"timespan": ts, "category": cat, "label": f"{out_var}_{cat_sfx}_{ts_sfx}"}
        for ts_sfx, ts in params["time_periods"].items()
        for cat_sfx, cat in params.get("categories", {"": None}).items()
    ]
    resolved = props_for_scopes(roadway_net.links_df, params["v"], scopes)
    for label, series in resolved.items():
        roadway_net.links_df[label] = series
```

**Estimated speedup:** 35 × (validate + explode) → 1 validate + N_properties explodes.
For 7 properties: 35 expensive ops → 7 explodes + 35 cheap filters.
**~10–30× for the `split_properties` step (10–45s → ~1–3s).**

**Implementation note:** Requires a PR to `network_wrangler` to expose `props_for_scopes` (or make `_create_exploded_df_for_scoped_prop` public). The `cube_wrangler` caller change is then straightforward.

---

### 🟡 5. Network loaded from geojson instead of parquet — MEDIUM (operational)

**Location:** `Project.create_project` / notebook cell 8
**Why slow:** geojson is text-parsed row by row. Parquet is columnar binary. On a 200k-link production network: **15–30s vs 1–2s**.
**Fix:** One-time conversion; `load_roadway` auto-detects format:

```python
write_roadway(net, "v2050TPP", "/path/to/dir", file_format="parquet")
# future loads pick up .parquet automatically
```

**Estimated speedup: 10–20× for the load step.** No code changes required — purely operational.

---

### 🟡 6. `validate_df_to_model` called on every `prop_for_scope` invocation — MEDIUM

**Location:** `network_wrangler/roadway/links/scopes.py`, line 365

Even after fixing issue 4 (batching), `validate_df_to_model` is still called once per property in `props_for_scopes`. Since `links_df` hasn't changed between calls, this is redundant work.

**Proposed fix (in network_wrangler):** Accept a pre-validated `links_df` and skip re-validation, or add an `already_validated: bool = False` parameter. Low risk since the caller controls the DataFrame.

**Estimated improvement:** Reduces schema-validation overhead for the N_properties calls that remain after fix 4.

---

### 🟢 7. `fill_na` uses `apply(lambda)` over full links_df — LOW (deprecated path)

**Location:** `cube_wrangler/roadway.py::fill_na`, lines 572–574
**Status:** `fill_na` is deprecated; `convert_types()` is the replacement. Not on the critical path.

---

## Prioritised next steps

| Priority | Issue | Estimated gain | Where to implement |
|----------|-------|----------------|--------------------|
| **P1** | `props_for_scopes` batch API | 10–30× for split_properties step | `network_wrangler` PR + `cube_wrangler` caller |
| **P2** | Convert production network to parquet | 10–20× for load step | Operational (no code change) |
| **P3** | Skip re-validation in `props_for_scopes` | Removes redundant schema checks | `network_wrangler` |

---

## Expected total improvement (production scenario, after all fixes)

| Step | Current (est.) | After P1 fixes | After all fixes | Speedup |
|------|---------------|----------------|-----------------|---------|
| `load_roadway` (geojson) | 20s | 20s | 2s | 10× |
| `split_properties` (35 prop_for_scope calls) | 45s | ~2s | ~1s | ~30× |
| `_process_link_changes` (5k changes, 200k links) | 300s+ | **30s** ✅ | 30s | ~10× |
| Debug logging overhead | 2s | **<0.1s** ✅ | <0.1s | — |
| **Total** | **~6 min** | **~55s** | **~35s** | **~10×** |

The biggest remaining wins are `split_properties` (P1) and the network load format (P2 — free). Between them they account for ~65s on production. The `_process_link_changes` optimisation (done) accounts for ~270s at 5k changes — the dominant saving.
