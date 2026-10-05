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
