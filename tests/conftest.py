"""Shared pytest fixtures for cube_wrangler tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"

STPAUL_LINK_JSON = DATA_DIR / "st_paul_link.json"
STPAUL_NODE_GEOJSON = DATA_DIR / "st_paul_node.geojson"
STPAUL_SHAPE_GEOJSON = DATA_DIR / "st_paul_shape.geojson"

CUBE_WRANGLER_ROOT = TESTS_DIR.parent


# ---------------------------------------------------------------------------
# Network fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def stpaul_links_df() -> pd.DataFrame:
    """Raw stpaul links loaded as a plain DataFrame (no network_wrangler overhead).

    Session-scoped so the 66k-row JSON is only read once per test run.
    """
    return pd.read_json(STPAUL_LINK_JSON)


@pytest.fixture(scope="session")
def stpaul_net():
    """Fully loaded stpaul RoadwayNetwork from network_wrangler's example data.

    Uses network_wrangler's example stpaul data (which fully conforms to
    RoadLinksTable) rather than the cube_wrangler test JSON, which contains
    NaN lanes that fail schema coercion.

    Session-scoped so the expensive load (geojson parse + validation) only
    happens once per test run.
    """
    import network_wrangler
    from network_wrangler import load_roadway

    nw_examples = Path(network_wrangler.__file__).parent.parent / "examples" / "stpaul"
    return load_roadway(
        links_file=nw_examples / "link.json",
        nodes_file=nw_examples / "node.geojson",
        shapes_file=nw_examples / "shape.geojson",
    )


@pytest.fixture(scope="session")
def stpaul_net_with_scoped(stpaul_net):
    """Network with synthetic scoped lanes values injected (stpaul fixture).

    Without scoped values both old and new split_properties paths are trivially
    fast (no sc_ column → return default immediately). This fixture populates
    sc_lanes on ~10% of links so the benchmark exercises the explode/filter path.
    """
    import copy

    import numpy as np
    from network_wrangler.models.roadway.types import ScopedLinkValueItem

    net = copy.copy(stpaul_net)  # shallow copy — we'll replace links_df
    links = stpaul_net.links_df.copy()

    # Build sc_lanes as a full-length object array (None for un-scoped links).
    sc_lanes = np.empty(len(links), dtype=object)
    sc_lanes[:] = None
    for i, (_idx, row) in enumerate(links.iterrows()):
        if i % 10 == 0:
            sc_lanes[i] = [ScopedLinkValueItem(timespan=["6:00", "10:00"], value=int(row["lanes"]) + 1)]

    links["sc_lanes"] = sc_lanes
    net.links_df = links
    return net


# ---------------------------------------------------------------------------
# Log file fixtures  (pre-generated files in tests/data/)
# ---------------------------------------------------------------------------


def _log_path(n: int) -> Path:
    return DATA_DIR / f"changes_{n}.log"


@pytest.fixture(scope="session")
def log_file_10() -> Path:
    """Path to a synthetic log file with 10 Change rows."""
    path = _log_path(10)
    if not path.exists():
        _generate_log(10, path)
    return path


@pytest.fixture(scope="session")
def log_file_50() -> Path:
    """Path to a synthetic log file with 50 Change rows."""
    path = _log_path(50)
    if not path.exists():
        _generate_log(50, path)
    return path


@pytest.fixture(scope="session")
def log_file_100() -> Path:
    """Path to a synthetic log file with 100 Change rows."""
    path = _log_path(100)
    if not path.exists():
        _generate_log(100, path)
    return path


@pytest.fixture(scope="session")
def log_file_500() -> Path:
    """Path to a synthetic log file with 500 Change rows."""
    path = _log_path(500)
    if not path.exists():
        _generate_log(500, path)
    return path


@pytest.fixture(scope="session")
def log_file_1000() -> Path:
    """Path to a synthetic log file with 1000 Change rows."""
    path = _log_path(1000)
    if not path.exists():
        _generate_log(1000, path)
    return path


def _generate_log(n: int, path: Path) -> None:
    """Auto-generate a missing log file using the stpaul network."""
    from utils.logfile import generate_change_logfile, load_usable_links

    usable = load_usable_links(STPAUL_LINK_JSON)
    generate_change_logfile(n, usable, path)


# ---------------------------------------------------------------------------
# Parameters fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cube_parameters():
    """Base cube_wrangler Parameters instance pointed at the repo root."""
    from cube_wrangler.parameters import Parameters

    return Parameters(base_dir=CUBE_WRANGLER_ROOT)
