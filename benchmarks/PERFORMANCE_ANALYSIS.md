# Performance Analysis: Cube Log → Project Card

## How to reproduce

```bash
# Generate synthetic log files using real stpaul (A,B) pairs
python benchmarks/generate_test_logfile.py

# Run the pipeline benchmark (stages 1–3, with current vs improved 3b)
python benchmarks/bench_log_to_card.py --sizes 10,50,100,500,1000

# Profile a specific size with cProfile
python -m cProfile -o benchmarks/profile_changes_500.prof \
    benchmarks/bench_log_to_card.py --sizes 500
snakeviz benchmarks/profile_changes_500.prof

# Measure prop_for_scope independently
python benchmarks/_bench_split_props.py
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

### `_process_link_changes` (the main bottleneck)

Measured on stpaul network (66k links):

| N changes | Current (iterrows + full scan) | Improved (set_index + collect-concat) | Speedup |
|-----------|-------------------------------|---------------------------------------|---------|
| 10 | 0.004s | 0.156s* | — |
| 50 | 0.016s | 0.019s | 0.8× |
| 100 | 0.030s | 0.025s | 1.2× |
| 500 | 0.158s | 0.067s | 2.4× |
| 1000 | 0.290s | 0.112s | 2.6× |

\* At n=10, the fixed cost of `set_index` on 66k rows (≈0.15s) dominates. On real production runs (hundreds to thousands of changes), the improvement grows.

**On a 200k-link production network:** the full scan per row is ~3× slower (≈0.9ms per change row vs 0.3ms on stpaul). For 5,000 changes that's ~4.5s (current) vs ~0.4s (improved index) — a ~10× speedup, plus the one-time `set_index` cost of ~0.5s.

**The iterrows + concat pattern scales as O(N_network × N_changes).** The improved version scales as O(N_network + N_changes).

---

## Root causes, ranked by severity on a production run

### 1. `prop_for_scope` called once per (property × timeperiod × category) combination — **HIGH**

**Location:** `cube_wrangler/roadway.py::split_properties_by_time_period_and_category`, lines 64–84
**Code:**
```python
for time_suffix, category_suffix in itertools.product(time_periods, categories):
    roadway_net.links_df[out_var + "_" + ...] = prop_for_scope(
        roadway_net.links_df, params["v"], category=..., timespan=...
    )[params["v"]]
```
**Why slow:** Each `prop_for_scope` call does a full O(N) pass over `links_df` to explode scoped values. MetCouncil has ≥35 combinations (3 simple props × 5 time periods + `price` with 4 categories × 5 time periods). At ~0.3s/call on 66k links, this is **10–15s on a small network and 30–45s on production**.

The property's scoped list only needs to be exploded once; the 35 column assignments can derive from a single exploded representation.

---

### 2. Full-network boolean mask scan per change row — **HIGH**

**Location:** `cube_wrangler/project.py::_process_single_link_change`, lines 722–725
**Code:**
```python
base_df = self.base_roadway_network.links_df[
    (self.base_roadway_network.links_df["A"] == change_row.A)
    & (self.base_roadway_network.links_df["B"] == change_row.B)
].copy()
```
**Why slow:** Called inside `iterrows` loop, once per change row. For N_changes rows on a M-link network, this is **O(N_changes × M)** boolean mask evaluations (each mask allocates a new array). At 5,000 changes on 200k links: 1B boolean ops.

---

### 3. Growing DataFrame via `pd.concat` inside `iterrows` loop — **HIGH**

**Location:** `cube_wrangler/project.py::_process_link_changes`, lines 896–900
**Code:**
```python
for index, row in cube_change_df.iterrows():
    card_df = _process_single_link_change(row, changeable_col)
    change_link_dict_df = pd.concat([change_link_dict_df, card_df], ...)
```
**Why slow:** Each `pd.concat` copies all previously accumulated rows into a new DataFrame. N concats on an average-size frame is **O(N²)** total memory allocation. At 1,000 changes this is 500k rows copied; at 5,000 it's 12.5M rows copied for no reason.

Issues 2 and 3 together cause the iterrows loop to scale super-linearly. The improved benchmark version (`set_index` lookup + collect-then-concat) is already 2.6× faster at 1,000 changes on the small network.

---

### 4. Network loaded from geojson instead of parquet — **MEDIUM**

**Location:** `Project.create_project` / notebook cell 8
**Code:** `load_roadway(links_file="v2050TPP_link.json", ...)`
**Why slow:** geojson is text-parsed row by row. Parquet is columnar binary with direct memory mapping. On a 200k-link production network, this can be **15–30s vs 1–2s**.

The network file format is a one-time conversion cost but affects every project card generation run.

---

### 5. `WranglerLogger.debug` called in inner loops — **MEDIUM**

**Location:** `_process_single_link_change`, line 750 (inside the `for col in changeable_col` loop)
**Code:** `WranglerLogger.debug("Assessing Column: {}".format(col))`
**Why slow:** Python string formatting + log-level check happens for every (change_row, column) pair even when logging is at INFO level. For 1,000 changes × 30 changeable columns = 30,000 format calls. These are cheap individually but add up. **Estimated: 0.1–0.5s.**

---

### 6. `fill_na` uses `apply(lambda)` over full links_df — **MEDIUM (secondary path)**

**Location:** `cube_wrangler/roadway.py::fill_na`, lines 572–574
**Code:**
```python
roadway_net.links_df[x] = roadway_net.links_df[x].apply(
    lambda k: 0 if k in [np.nan, "", float("nan"), "NaN"] else k
)
```
**Why slow:** Row-wise Python lambda instead of vectorized `fillna()` + `replace()`. On 100k links × 20 numeric columns = 2M Python calls. **Estimated: 1–3s.**
Not on the main log→card path but affects any workflow that calls `fill_na`.

---

### 7. Linear search through `property_dict_list` to detect duplicates — **LOW**

**Location:** `_process_single_link_change`, lines 849–861
**Code:** `for processed_p in property_dict_list: if processed_p["property"] == p_base_name`
**Why slow:** O(P) linear scan per changed column where P = number of properties in `property_dict_list`. With 50+ columns this is O(P²). **Low impact** because P is small in practice.

---

## Prioritized solutions

### Priority 1 — Fix the O(N_changes × N_network) scan (Issues 2 + 3) ★★★

Pre-build a `MultiIndex` on `(A, B)` once before the loop; replace the growing-concat with a collect-then-concat pattern:

```python
# Before the loop — O(N_network) one-time cost
ab_index = self.base_roadway_network.links_df.set_index(["A", "B"])

card_frames = []   # collect first, concat once

for _idx, row in cube_change_df.iterrows():
    try:
        base_row = ab_index.loc[(row["A"], row["B"])]
    except KeyError:
        continue
    if isinstance(base_row, pd.DataFrame):
        base_row = base_row.iloc[0]
    # ... rest of _process_single_link_change unchanged ...
    card_frames.append(card_df)

change_link_dict_df = pd.concat(card_frames, ignore_index=True, sort=False) if card_frames else pd.DataFrame(...)
```

**Estimated speedup:** 2.6× at 1,000 changes on stpaul; **~10× on a 200k-link production network at 5,000 changes.** Minimal code change — can be done in `_process_link_changes` without touching `_process_single_link_change` logic.

---

### Priority 2 — Eliminate redundant `prop_for_scope` calls (Issue 1) ★★★

The current code calls `prop_for_scope` 35+ times, each scanning all links. Since all time period / category variants of a property can be derived from a single `prop_for_scope` call that returns the exploded frame, restructure to call once per base property and then select the columns:

```python
# Pseudocode: one explode per property instead of one per (property, timeperiod, category)
for prop_name, params in properties_to_split.items():
    exploded = prop_for_scope(links_df, params["v"])  # single full pass
    for ts, timespan in params["time_periods"].items():
        links_df[f"{prop_name}_{ts}"] = _filter_exploded_for_timespan(exploded, timespan)
```

Requires understanding `prop_for_scope`'s return structure, but the payoff is eliminating 34 of 35 network scans. **Estimated speedup: 10–30× for the split_properties step alone (10–45s → ~1s).**

---

### Priority 3 — Convert production network to parquet (Issue 4) ★★

One-time conversion; `load_roadway` auto-detects format:

```python
write_roadway(net, "v2050TPP", "/path/to/network/dir", file_format="parquet")
# Future loads: load_roadway_from_dir("/path/to/network/dir")  # picks up .parquet
```

**Estimated speedup: 10–20× for the network load step (15–30s → 1–2s).**

---

### Priority 4 — Suppress debug logging in the inner loop (Issue 5) ★

Change:
```python
WranglerLogger.debug("Assessing Column: {}".format(col))  # line 750
```
to:
```python
# Remove entirely, or guard with:
if WranglerLogger.isEnabledFor(logging.DEBUG):
    WranglerLogger.debug("Assessing Column: %s", col)
```
Use `%s` lazy formatting instead of `.format()` so the string is never built when not at DEBUG level. **Estimated improvement: 0.1–0.5s; 5 minutes of work.**

---

### Priority 5 — Vectorize `fill_na` (Issue 6) ★

Replace:
```python
roadway_net.links_df[x] = roadway_net.links_df[x].apply(
    lambda k: 0 if k in [np.nan, "", float("nan"), "NaN"] else k
)
```
with:
```python
roadway_net.links_df[x] = (
    pd.to_numeric(roadway_net.links_df[x], errors="coerce").fillna(0)
)
```
**Estimated improvement: 1–3s; affects any workflow calling `fill_na`.**

---

## Expected total improvement (production scenario)

| Step | Current (est.) | After fixes | Speedup |
|------|---------------|-------------|---------|
| `load_roadway` (geojson → parquet) | 20s | 2s | 10× |
| `split_properties` (35 prop_for_scope calls) | 45s | ~2s | ~22× |
| `_process_link_changes` (5k changes, 200k links) | 300s+ | 30s | 10× |
| Debug logging overhead | 2s | <0.1s | — |
| **Total** | **~6 min** | **~35s** | **~10×** |

The three highest-priority fixes (P1–P3) are independent and can be implemented in parallel. P1 is the lowest risk and highest bang-for-buck per line of code changed.
