"""Parameters and configuration for cube_wrangler.

Cube Wrangler parameters are organised into nested pydantic dataclasses,
mirroring the pattern used in network_wrangler.  The top-level class is
:class:`Parameters`, which groups:

- :class:`TimePeriodsConfig`  – time period code → (start, end) time strings
- :class:`CategoriesConfig`   – vehicle category fallback lookup order

Column *types* are no longer declared here.  They are encoded in the pandera
``DataFrameModel`` schemas :class:`~cube_wrangler.models.tables.CubeLinksTable`
and :class:`~cube_wrangler.models.tables.CubeNodesTable`, and applied
automatically via :func:`~cube_wrangler.utils.models.coerce_df_to_model`.

File-path attributes (``settings_location``, ``scratch_location``, and the
output-file shortcuts) are derived in ``model_post_init`` from ``base_dir``.

Usage::

    # defaults work out of the box
    params = Parameters()

    # or point at a specific repo root
    params = Parameters(base_dir="/path/to/cube_wrangler")

    # override individual settings
    params = Parameters(time_periods=TimePeriodsConfig(AM=("7:00", "9:00")))

Backward-compatible dict-style access to time periods and categories::

    params.time_period_to_time   # {"EA": ("3:00","6:00"), "AM": ...}
    params.categories            # {"sov": [...], "hov2": [...], ...}
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic.dataclasses import dataclass

from .logger import WranglerLogger


# ---------------------------------------------------------------------------
# Nested config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TimePeriodsConfig:
    """Time-of-day period definitions.

    Each attribute maps a period code to a ``(start, end)`` time string pair
    in ``HH:MM`` format (24-hour clock).

    Attributes:
        EA: Early-morning period.
        AM: AM peak period.
        MD: Midday period.
        PM: PM peak period.
        NT: Night period.
    """

    EA: tuple[str, str] = ("3:00", "6:00")
    AM: tuple[str, str] = ("6:00", "10:00")
    MD: tuple[str, str] = ("10:00", "15:00")
    PM: tuple[str, str] = ("15:00", "19:00")
    NT: tuple[str, str] = ("19:00", "3:00")

    def as_dict(self) -> dict[str, tuple[str, str]]:
        """Return time periods as a plain dict (period code → time-span tuple)."""
        return {"EA": self.EA, "AM": self.AM, "MD": self.MD, "PM": self.PM, "NT": self.NT}


@dataclass
class CategoriesConfig:
    """Vehicle category fallback lookup order.

    Each attribute lists the sequence of source categories to try (in order)
    when looking up a scoped property value for that category.

    Attributes:
        sov: Single-occupancy vehicle.
        hov2: High-occupancy vehicle (2+ persons).
        hov3: High-occupancy vehicle (3+ persons).
        truck: Truck/commercial vehicle.
    """

    sov: list[str] = Field(default_factory=lambda: ["sov", "default"])
    hov2: list[str] = Field(default_factory=lambda: ["hov2", "default", "sov"])
    hov3: list[str] = Field(default_factory=lambda: ["hov3", "hov2", "default", "sov"])
    truck: list[str] = Field(default_factory=lambda: ["trk", "sov", "default"])

    def as_dict(self) -> dict[str, list[str]]:
        """Return categories as a plain dict (category code → fallback list)."""
        return {
            "sov": self.sov,
            "hov2": self.hov2,
            "hov3": self.hov3,
            "truck": self.truck,
        }


# ---------------------------------------------------------------------------
# Top-level Parameters dataclass
# ---------------------------------------------------------------------------


@dataclass
class Parameters:
    """All parameters defining the Cube Wrangler network processing pipeline.

    Parameters can be constructed with explicit overrides; anything not
    provided falls back to the documented defaults.

    Attributes:
        base_dir: Root directory of the cube_wrangler installation.
            Defaults to the current working directory.
        settings_location: Directory containing crosswalk CSV files.
            Derived from ``base_dir`` when not set explicitly.
        scratch_location: Directory for intermediate output files.
            Derived from ``base_dir`` when not set explicitly.
        time_periods: Time-of-day period definitions.
        categories: Vehicle category fallback lookup order.
        zones: Number of TAZs in the model.
        output_variables: Ordered list of columns written to the Cube output.
            Column *types* are encoded in
            :class:`~cube_wrangler.models.tables.CubeLinksTable` /
            :class:`~cube_wrangler.models.tables.CubeNodesTable`, not here.

    Path shortcuts (set in ``model_post_init``):
        net_to_dbf_crosswalk, log_to_net_crosswalk,
        output_link_shp, output_node_shp, output_link_csv, output_node_csv,
        output_link_txt, output_node_txt, output_link_header_width_txt,
        output_node_header_width_txt, output_cube_network_script.

    Backward-compatible properties:
        time_period_to_time: dict form of ``time_periods``.
        properties_to_split: computed mapping used by roadway splitting logic.
    """

    base_dir: Path = Field(default_factory=Path.cwd)
    settings_location: Path | None = None
    scratch_location: Path | None = None
    time_periods: TimePeriodsConfig = Field(default_factory=TimePeriodsConfig)
    categories: CategoriesConfig = Field(default_factory=CategoriesConfig)
    zones: int = 3061
    output_variables: list[str] = Field(
        default_factory=lambda: [
            "model_link_id",
            "link_id",
            "A",
            "B",
            "shstGeometryId",
            "shape_id",
            "distance",
            "roadway",
            "name",
            "roadway_class",
            "bike_access",
            "walk_access",
            "drive_access",
            "truck_access",
            "trn_priority_AM",
            "trn_priority_MD",
            "trn_priority_PM",
            "trn_priority_NT",
            "ttime_assert_AM",
            "ttime_assert_MD",
            "ttime_assert_PM",
            "ttime_assert_NT",
            "lanes_AM",
            "lanes_MD",
            "lanes_PM",
            "lanes_NT",
            "price_sov_AM",
            "price_hov2_AM",
            "price_hov3_AM",
            "price_truck_AM",
            "price_sov_MD",
            "price_hov2_MD",
            "price_hov3_MD",
            "price_truck_MD",
            "price_sov_PM",
            "price_hov2_PM",
            "price_hov3_PM",
            "price_truck_PM",
            "price_sov_NT",
            "price_hov2_NT",
            "price_hov3_NT",
            "price_truck_NT",
            "roadway_class_idx",
            "assign_group",
            "access_AM",
            "access_MD",
            "access_PM",
            "access_NT",
            "mpo",
            "area_type",
            "county",
            "centroidconnect",
            "AADT",
            "count_year",
            "count_AM",
            "count_MD",
            "count_PM",
            "count_NT",
            "count_daily",
            "model_node_id",
            "N",
            "osm_node_id",
            "bike_node",
            "transit_node",
            "walk_node",
            "drive_node",
            "geometry",
            "X",
            "Y",
            "ML_lanes_AM",
            "ML_lanes_MD",
            "ML_lanes_PM",
            "ML_lanes_NT",
            "segment_id",
            "managed",
            "bus_only",
            "rail_only",
            "bike_facility",
            "bike",
            "walk",
        ]
    )

    def __post_init__(self) -> None:
        """Derive path attributes and computed lookups after construction."""
        self.base_dir = Path(self.base_dir)

        if self.settings_location is None:
            self.settings_location = self.base_dir / "examples" / "settings"
        else:
            self.settings_location = Path(self.settings_location)

        if self.scratch_location is None:
            self.scratch_location = self.base_dir / "tests" / "scratch"
        else:
            self.scratch_location = Path(self.scratch_location)

        # Crosswalk files
        self.net_to_dbf_crosswalk: Path = self.settings_location / "net_to_dbf.csv"
        self.log_to_net_crosswalk: Path = self.settings_location / "log_to_net.csv"

        # Output file shortcuts
        self.output_link_shp: Path = self.scratch_location / "links.shp"
        self.output_node_shp: Path = self.scratch_location / "nodes.shp"
        self.output_link_csv: Path = self.scratch_location / "links.csv"
        self.output_node_csv: Path = self.scratch_location / "nodes.csv"
        self.output_link_txt: Path = self.scratch_location / "links.txt"
        self.output_node_txt: Path = self.scratch_location / "nodes.txt"
        self.output_link_header_width_txt: Path = (
            self.scratch_location / "links_header_width.txt"
        )
        self.output_node_header_width_txt: Path = (
            self.scratch_location / "nodes_header_width.txt"
        )
        self.output_cube_network_script: Path = (
            self.scratch_location / "make_complete_network_from_fixed_width_file.s"
        )

        # Computed roadway-splitting lookup (depends on time_periods + categories)
        tp = self.time_period_to_time
        self.properties_to_split: dict = {
            "trn_priority": {"v": "trn_priority", "time_periods": tp},
            "ttime_assert": {"v": "ttime_assert", "time_periods": tp},
            "lanes": {"v": "lanes", "time_periods": tp},
            "ML_lanes": {"v": "ML_lanes", "time_periods": tp},
            "price": {
                "v": "price",
                "time_periods": tp,
                "categories": self.categories.as_dict(),
            },
            "access": {"v": "access", "time_periods": tp},
        }

        WranglerLogger.debug(f"Parameters initialised with base_dir={self.base_dir}")

    # ------------------------------------------------------------------
    # Backward-compatible accessors
    # ------------------------------------------------------------------

    @property
    def time_period_to_time(self) -> dict[str, tuple[str, str]]:
        """Dict form of ``time_periods`` (period code → time-span tuple)."""
        return self.time_periods.as_dict()
