"""Generate synthetic Cube log files for benchmarking.

Uses real (A, B) node pairs from the stpaul example network.
Writes log files to tests/data/ so they are available to both the
benchmark scripts and the pytest benchmark tests.

Usage:
    python benchmarks/generate_test_logfile.py

Requires: the stpaul example network at
  met_council_wrangler/examples/stpaul/link.json
(or set LINK_JSON env var to point elsewhere).
"""

import os
import sys
from pathlib import Path

# Make tests/utils importable without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.utils.logfile import generate_change_logfile, load_usable_links

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
SIZES = [10, 50, 100, 500, 1000]

STPAUL_LINK_JSON = Path(
    os.environ.get(
        "LINK_JSON",
        Path(__file__).parent.parent.parent
        / "met_council_wrangler"
        / "examples"
        / "stpaul"
        / "link.json",
    )
)

OUT_DIR = Path(__file__).parent.parent / "tests" / "data"


def main():
    """Generate synthetic log files for all configured sizes."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading {STPAUL_LINK_JSON} …")
    usable = load_usable_links(STPAUL_LINK_JSON)
    print(f"  {len(usable)} usable links")

    for n in SIZES:
        out_path = OUT_DIR / f"changes_{n}.log"
        generate_change_logfile(n, usable, out_path)
        print(f"  Wrote {out_path}  ({n} change rows)")

    print("Done.")


if __name__ == "__main__":
    main()
