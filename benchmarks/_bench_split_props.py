"""Internal helper: measure split_properties_by_time_period_and_category timing."""
import itertools
import time
import sys

sys.path.insert(0, "..")  # met_council_wrangler

from network_wrangler import load_roadway
from network_wrangler.roadway.links.scopes import prop_for_scope

STPAUL = "../../met_council_wrangler/examples/stpaul"

t0 = time.perf_counter()
net = load_roadway(
    links_file=f"{STPAUL}/link.json",
    nodes_file=f"{STPAUL}/node.geojson",
    shapes_file=f"{STPAUL}/shape.geojson",
)
print(f"load_roadway: {time.perf_counter()-t0:.2f}s, {len(net.links_df)} links")

time_period_to_time = {
    "EA": ("3:00", "6:00"),
    "AM": ("6:00", "10:00"),
    "MD": ("10:00", "15:00"),
    "PM": ("15:00", "19:00"),
    "NT": ("19:00", "3:00"),
}
categories = {
    "sov": ["sov", "default"],
    "hov2": ["hov2", "default", "sov"],
    "hov3": ["hov3", "hov2", "default", "sov"],
    "truck": ["trk", "sov", "default"],
}
properties_to_split = {
    "lanes": {"v": "lanes", "time_periods": time_period_to_time},
    "trn_priority": {"v": "trn_priority", "time_periods": time_period_to_time},
    "ttime_assert": {"v": "ttime_assert", "time_periods": time_period_to_time},
    "price": {"v": "price", "time_periods": time_period_to_time, "categories": categories},
}

print("\nBenchmarking prop_for_scope calls:")
t_total = time.perf_counter()
call_count = 0
for prop_name, params in properties_to_split.items():
    v = params["v"]
    tps = params.get("time_periods", {})
    cats = params.get("categories", {})
    if cats:
        for (ts, timespan), (cat, cat_vals) in itertools.product(tps.items(), cats.items()):
            t0 = time.perf_counter()
            prop_for_scope(net.links_df, v, category=cat_vals, timespan=list(timespan))
            elapsed = time.perf_counter() - t0
            call_count += 1
            print(f"  {v}_{cat}_{ts}: {elapsed:.3f}s")
    else:
        for ts, timespan in tps.items():
            t0 = time.perf_counter()
            prop_for_scope(net.links_df, v, timespan=list(timespan))
            elapsed = time.perf_counter() - t0
            call_count += 1
            print(f"  {v}_{ts}: {elapsed:.3f}s")

total = time.perf_counter() - t_total
print(f"\nTotal: {total:.2f}s for {call_count} prop_for_scope calls")
print(f"Average per call: {total/call_count*1000:.1f}ms")
