"""Utilities for generating synthetic Cube log files for testing and benchmarking.

Log files use the format from the stpaul example network:
  HighwayLayerLogX,"path",8,25,<date>
  Node,model_node_id,X,Y
  Link,A,B,model_link_id,drive_access,walk_access,bike_access,length
  L,C,<group>,<A>,<B>,<model_link_id>,<drive_access>,<walk_access>,<bike_access>,<length>

Columns are chosen from fields that are actually present and non-null in the
stpaul link.json test network and that are not in Project.STATIC_VALUES, so
_process_single_link_change will detect real diffs and exercise the full loop.
Note: the stpaul network stores lanes as a scoped property (always null as a
plain column), so it is excluded.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# Columns written into each synthetic log file's link header and data rows.
# Must all be present and non-null in the stpaul test network.
LOG_LINK_COLS = [
    "A",
    "B",
    "model_link_id",
    "drive_access",
    "walk_access",
    "bike_access",
    "length",
]

# Transformations applied to create a genuine change from the network's current value.
_CHANGES: dict[str, object] = {
    "drive_access": lambda v: not v,
}


def _log_header(date: str = "1/1/2024 12:00:00 PM") -> str:
    return (
        f'HighwayLayerLogX,"benchmark_network.net",8,25,{date}\n'
        "Node,model_node_id,X,Y\n"
        "Link," + ",".join(LOG_LINK_COLS) + "\n"
    )


def _link_row(obj: str, op: str, group: int, vals: dict) -> str:
    parts = [obj, op, str(group)] + [str(vals[c]) for c in LOG_LINK_COLS]
    return ",".join(parts)


def load_usable_links(link_json_path: Path) -> list[dict]:
    """Return links from *link_json_path* that have all required LOG_LINK_COLS.

    Args:
        link_json_path: Path to a network link.json file (list of dicts).

    Returns:
        Filtered list of link dicts.
    """
    with Path(link_json_path).open() as f:
        links = json.load(f)
    return [
        lk
        for lk in links
        if all(lk.get(c) is not None for c in LOG_LINK_COLS)
    ]


def generate_change_logfile(
    n: int,
    usable_links: list[dict],
    out_path: Path,
    *,
    seed: int = 42,
    date: str = "1/1/2024 12:00:00 PM",
) -> Path:
    """Write a synthetic Change-only log file with *n* rows.

    Each row modifies one or more properties so that _process_single_link_change
    detects a real difference against the base network.

    Args:
        n: Number of Change rows to generate.
        usable_links: List of link dicts (from load_usable_links).
        out_path: Destination path for the log file.
        seed: Random seed for reproducibility.
        date: Timestamp string for the log file header.

    Returns:
        The path the file was written to.
    """
    if not usable_links:
        msg = f"No usable links found — cannot generate {out_path.name}"
        raise ValueError(msg)
    rng = random.Random(seed)
    sample = rng.choices(usable_links, k=n)
    lines = [_log_header(date)]
    for i, lk in enumerate(sample):
        row = {c: lk[c] for c in LOG_LINK_COLS}
        for col, transform in _CHANGES.items():
            if col in row:
                row[col] = transform(row[col])
        lines.append(_link_row("L", "C", i % 5, row))
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
