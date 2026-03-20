"""Pandera DataFrameModel schemas for Cube Wrangler network tables.

These schemas serve as the single source of truth for column types in the
Cube roadway network.  Declaring a column here (with ``coerce=True`` in
``Config``) removes the need for manual ``.astype()`` loops spread across
``roadway.py`` and ``project.py``.

Usage::

    from cube_wrangler.models.tables import CubeLinksTable, CubeNodesTable
    from cube_wrangler.utils.models import coerce_df_to_model

    links_df = coerce_df_to_model(links_df, CubeLinksTable)
    nodes_df = coerce_df_to_model(nodes_df, CubeNodesTable)
"""

from __future__ import annotations

from typing import Optional

import pandera.pandas as pa
from pandera.pandas import DataFrameModel, Field
from pandera.typing import Series


class CubeLinksTable(DataFrameModel):
    """Schema and type coercion rules for Cube roadway link records.

    All columns are marked ``Optional`` so that DataFrames with only a subset
    of these columns still validate successfully.  ``Config.coerce = True``
    means any column that *is* present will be cast to the declared dtype.

    Attributes:
        model_link_id: Unique integer link identifier.
        A: Integer model node ID of the link origin.
        B: Integer model node ID of the link destination.
        link_id: String link identifier (e.g. SharedStreets reference).
        shstGeometryId: SharedStreets geometry identifier.
        shape_id: Identifier referencing the shapes table.
        distance: Link distance (float).
        roadway: OSM road type string.
        name: Link name string.
        roadway_class: Roadway class integer code.
        rail_only: Boolean - link restricted to rail.
        bus_only: Boolean - link restricted to buses.
        drive_access: Boolean - driving is permitted.
        bike_access: Boolean - cycling is permitted.
        walk_access: Boolean - walking is permitted.
        truck_access: Boolean - trucks are permitted.
        trn_priority: Integer transit priority value.
        ttime_assert: Float travel-time assertion.
        lanes_AM: Integer lane count for AM period.
        lanes_MD: Integer lane count for MD period.
        lanes_PM: Integer lane count for PM period.
        lanes_NT: Integer lane count for NT period.
        price: Default link price (float).
        assign_group: Assignment group integer code.
        access_AM: AM period access value.
        access_MD: MD period access value.
        access_PM: PM period access value.
        access_NT: NT period access value.
        mpo: MPO membership indicator.
        area_type: Area type integer code.
        county: County integer code.
        centroidconnect: Centroid connector flag (int).
        AADT: Annual average daily traffic (int).
        count_year: Traffic count year.
        count_AM: AM period traffic count (int).
        count_MD: MD period traffic count (int).
        count_PM: PM period traffic count (int).
        count_NT: NT period traffic count (int).
        count_daily: Daily traffic count (int).
        bike_facility: Bike facility type integer code.
        bike: Bike network integer indicator.
        walk: Walk network integer indicator.
        ML_lanes_AM: Managed-lane count for AM period (int).
        ML_lanes_MD: Managed-lane count for MD period (int).
        ML_lanes_PM: Managed-lane count for PM period (int).
        ML_lanes_NT: Managed-lane count for NT period (int).
        segment_id: Segment identifier integer.
        managed: Managed-lane indicator integer.
        bus_only: Boolean - bus-only flag.
        drive_node: Drive node integer identifier.
        walk_node: Walk node integer identifier.
        bike_node: Bike node integer identifier.
        transit_node: Transit node integer identifier.
        X: Longitude / easting (float).
        Y: Latitude / northing (float).
    """

    # --- identifiers --------------------------------------------------------
    model_link_id: Optional[Series[int]] = Field(nullable=False, default=None)
    A: Optional[Series[int]] = Field(nullable=False, default=None)
    B: Optional[Series[int]] = Field(nullable=False, default=None)
    link_id: Optional[Series[str]] = Field(nullable=True, default=None)
    shstGeometryId: Optional[Series[str]] = Field(nullable=True, default=None)
    shape_id: Optional[Series[str]] = Field(nullable=True, default=None)

    # --- geometry / basic attributes ----------------------------------------
    distance: Optional[Series[float]] = Field(nullable=True, default=None)
    roadway: Optional[Series[str]] = Field(nullable=True, default=None)
    name: Optional[Series[str]] = Field(nullable=True, default=None)
    roadway_class: Optional[Series[int]] = Field(nullable=True, default=None)

    # --- boolean access flags -----------------------------------------------
    rail_only: Optional[Series[bool]] = Field(nullable=True, default=False)
    bus_only: Optional[Series[bool]] = Field(nullable=True, default=False)
    drive_access: Optional[Series[bool]] = Field(nullable=True, default=True)
    bike_access: Optional[Series[bool]] = Field(nullable=True, default=True)
    walk_access: Optional[Series[bool]] = Field(nullable=True, default=True)
    truck_access: Optional[Series[bool]] = Field(nullable=True, default=True)

    # --- time-period integer properties ------------------------------------
    trn_priority: Optional[Series[int]] = Field(nullable=True, default=None)
    ttime_assert: Optional[Series[float]] = Field(nullable=True, default=None)
    lanes_AM: Optional[Series[int]] = Field(nullable=True, default=None)
    lanes_MD: Optional[Series[int]] = Field(nullable=True, default=None)
    lanes_PM: Optional[Series[int]] = Field(nullable=True, default=None)
    lanes_NT: Optional[Series[int]] = Field(nullable=True, default=None)
    price: Optional[Series[float]] = Field(nullable=True, default=0.0)

    # --- assignment / grouping ---------------------------------------------
    assign_group: Optional[Series[int]] = Field(nullable=True, default=None)
    access_AM: Optional[Series[pa.Object]] = Field(nullable=True, default=None)
    access_MD: Optional[Series[pa.Object]] = Field(nullable=True, default=None)
    access_PM: Optional[Series[pa.Object]] = Field(nullable=True, default=None)
    access_NT: Optional[Series[pa.Object]] = Field(nullable=True, default=None)
    mpo: Optional[Series[pa.Object]] = Field(nullable=True, default=None)
    area_type: Optional[Series[int]] = Field(nullable=True, default=None)
    county: Optional[Series[int]] = Field(nullable=True, default=None)
    centroidconnect: Optional[Series[int]] = Field(nullable=True, default=0)

    # --- traffic counts ---------------------------------------------------
    AADT: Optional[Series[int]] = Field(nullable=True, default=0)
    count_year: Optional[Series[pa.Object]] = Field(nullable=True, default=None)
    count_AM: Optional[Series[int]] = Field(nullable=True, default=0)
    count_MD: Optional[Series[int]] = Field(nullable=True, default=0)
    count_PM: Optional[Series[int]] = Field(nullable=True, default=0)
    count_NT: Optional[Series[int]] = Field(nullable=True, default=0)
    count_daily: Optional[Series[int]] = Field(nullable=True, default=0)

    # --- bike / walk / transit node links ---------------------------------
    bike_facility: Optional[Series[int]] = Field(nullable=True, default=0)
    bike: Optional[Series[int]] = Field(nullable=True, default=0)
    walk: Optional[Series[int]] = Field(nullable=True, default=0)
    drive_node: Optional[Series[int]] = Field(nullable=True, default=None)
    walk_node: Optional[Series[int]] = Field(nullable=True, default=None)
    bike_node: Optional[Series[int]] = Field(nullable=True, default=None)
    transit_node: Optional[Series[int]] = Field(nullable=True, default=None)

    # --- managed lanes ----------------------------------------------------
    ML_lanes_AM: Optional[Series[int]] = Field(nullable=True, default=None)
    ML_lanes_MD: Optional[Series[int]] = Field(nullable=True, default=None)
    ML_lanes_PM: Optional[Series[int]] = Field(nullable=True, default=None)
    ML_lanes_NT: Optional[Series[int]] = Field(nullable=True, default=None)
    segment_id: Optional[Series[int]] = Field(nullable=True, default=None)
    managed: Optional[Series[int]] = Field(nullable=True, default=0)

    # --- coordinates -------------------------------------------------------
    X: Optional[Series[float]] = Field(nullable=True, default=None)
    Y: Optional[Series[float]] = Field(nullable=True, default=None)

    class Config:
        """Pandera model configuration."""

        coerce = True
        add_missing_columns = False


class CubeNodesTable(DataFrameModel):
    """Schema and type coercion rules for Cube roadway node records.

    All columns are Optional so DataFrames with only a subset validate cleanly.
    ``Config.coerce = True`` casts present columns to their declared dtypes.

    Attributes:
        model_node_id: Unique integer node identifier.
        N: Cube node number (same as model_node_id).
        osm_node_id: OSM node identifier string.
        drive_node: Boolean / integer - node is on the drive network.
        walk_node: Boolean / integer - node is on the walk network.
        bike_node: Boolean / integer - node is on the bike network.
        transit_node: Boolean / integer - node is on the transit network.
        X: Longitude / easting (float).
        Y: Latitude / northing (float).
    """

    model_node_id: Optional[Series[int]] = Field(nullable=False, default=None)
    N: Optional[Series[int]] = Field(nullable=True, default=None)
    osm_node_id: Optional[Series[str]] = Field(nullable=True, default=None)
    drive_node: Optional[Series[int]] = Field(nullable=True, default=0)
    walk_node: Optional[Series[int]] = Field(nullable=True, default=0)
    bike_node: Optional[Series[int]] = Field(nullable=True, default=0)
    transit_node: Optional[Series[int]] = Field(nullable=True, default=0)
    X: Optional[Series[float]] = Field(nullable=True, default=None)
    Y: Optional[Series[float]] = Field(nullable=True, default=None)

    class Config:
        """Pandera model configuration."""

        coerce = True
        add_missing_columns = False
