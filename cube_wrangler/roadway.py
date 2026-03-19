"""Roadway network transformations for Cube travel demand model output."""

from __future__ import annotations

import copy
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from network_wrangler.roadway.links.scopes import prop_for_scope
from network_wrangler.roadway.network import RoadwayNetwork
from pandas import DataFrame

from .logger import WranglerLogger
from .models.tables import CubeLinksTable, CubeNodesTable
from .parameters import Parameters
from .utils.models import coerce_df_to_model


def split_properties_by_time_period_and_category(
    roadway_net=None, parameters=None, properties_to_split=None
):
    """Splits properties by time period, assuming a variable structure of.

    Args:
        roadway_net: RoadwayNetwork whose links_df will be modified in place.
        parameters: Parameters instance providing default properties_to_split.
        properties_to_split: dict
            dictionary of output variable prefix mapped to the source variable and what to stratify it by
            e.g.
            {
                'trn_priority' : {'v':'trn_priority', 'times_periods':{"AM": ("6:00", "9:00"),"PM": ("16:00", "19:00")}},
                'ttime_assert' : {'v':'ttime_assert', 'times_periods':{"AM": ("6:00", "9:00"),"PM": ("16:00", "19:00")}},
                'lanes' : {'v':'lanes', 'times_periods':{"AM": ("6:00", "9:00"),"PM": ("16:00", "19:00")}},
                'ML_lanes' : {'v':'ML_lanes', 'times_periods':{"AM": ("6:00", "9:00"),"PM": ("16:00", "19:00")}},
                'price' : {'v':'price', 'times_periods':{"AM": ("6:00", "9:00"),"PM": ("16:00", "19:00")}},'categories': {"sov": ["sov", "default"],"hov2": ["hov2", "default", "sov"]}},
                'access' : {'v':'access', 'times_periods':{"AM": ("6:00", "9:00"),"PM": ("16:00", "19:00")}},
            }

    """
    import itertools

    link_attrs = copy.deepcopy(roadway_net.links_df.attrs)

    if properties_to_split == None:
        properties_to_split = parameters.properties_to_split

    for out_var, params in properties_to_split.items():
        if params["v"] not in roadway_net.links_df.columns:
            WranglerLogger.warning(
                "Specified variable to split: {} not in network variables: {}. Returning 0.".format(
                    params["v"], str(roadway_net.links_df.columns)
                )
            )
            if params.get("time_periods") and params.get("categories"):
                for time_suffix, category_suffix in itertools.product(
                    params["time_periods"], params["categories"]
                ):
                    roadway_net.links_df[out_var + "_" + time_suffix + "_" + category_suffix] = 0
            elif params.get("time_periods"):
                for time_suffix in params["time_periods"]:
                    roadway_net.links_df[out_var + "_" + time_suffix] = 0
        elif params.get("time_periods") and params.get("categories"):
            for time_suffix, category_suffix in itertools.product(
                params["time_periods"], params["categories"]
            ):
                roadway_net.links_df[out_var + "_" + category_suffix + "_" + time_suffix] = (
                    prop_for_scope(
                        roadway_net.links_df,
                        params["v"],
                        category=params["categories"][category_suffix],
                        timespan=params["time_periods"][time_suffix],
                    )[params["v"]]
                )
        elif params.get("time_periods"):
            for time_suffix in params["time_periods"]:
                roadway_net.links_df[out_var + "_" + time_suffix] = prop_for_scope(
                    roadway_net.links_df,
                    params["v"],
                    category=None,
                    timespan=params["time_periods"][time_suffix],
                )[params["v"]]
        else:
            msg = f"Shoudn't have a category without a time period: {params}"
            raise ValueError(msg)

    roadway_net.links_df.attrs = link_attrs

    return roadway_net


def calculate_distance_miles(roadway_net=None, network_variable="distance", overwrite=False):
    """Calculate link distance in miles.

    Args:
        roadway_net: RoadwayNetwork whose links_df will be updated.
        network_variable: Name of the distance column to write. Default "distance".
        overwrite (Bool): True if overwriting existing variable in network.  Default to False.

    Returns:
        None

    """
    link_attrs = copy.deepcopy(roadway_net.links_df.attrs)

    if network_variable in roadway_net.links_df:
        if overwrite or (roadway_net.links_df[network_variable].isnull().any()):
            WranglerLogger.info(
                f"Overwriting existing distance Variable '{network_variable}' already in network"
            )
        else:
            WranglerLogger.info(
                f"Distance Variable '{network_variable}' already in network. Returning without overwriting."
            )
            return roadway_net

    """
    Start actual process
    """

    temp_links_gdf = roadway_net.links_df.copy()
    temp_links_gdf.crs = "EPSG:4326"
    temp_links_gdf = temp_links_gdf.to_crs(epsg=26915)

    WranglerLogger.info("Calculating distance in miles for all links")
    temp_links_gdf[network_variable] = temp_links_gdf.geometry.length / 1609.34
    # overwrite 0 distance with 0.001 mile
    temp_links_gdf[network_variable] = np.where(
        temp_links_gdf[network_variable] == 0,
        0.001,
        temp_links_gdf[network_variable],
    )

    roadway_net.links_df[network_variable] = temp_links_gdf[network_variable]
    roadway_net.links_df.attrs = link_attrs

    return roadway_net


def calculate_distance(
    roadway_net=None,
    network_variable="distance",
    centroidconnect_only=False,
    overwrite=False,
):
    """Calculate link distance in miles.

    Args:
        roadway_net: RoadwayNetwork whose links_df will be updated.
        network_variable: Name of the distance column to write. Default "distance".
        centroidconnect_only (Bool):  True if calculating distance for centroidconnectors only.  Default to True.
        overwrite (Bool): True if overwriting existing variable in network.  Default to False.

    Returns:
        None

    """
    if network_variable in roadway_net.links_df:
        if overwrite:
            WranglerLogger.info(
                f"Overwriting existing distance Variable '{network_variable}' already in network"
            )
        else:
            WranglerLogger.info(
                f"Distance Variable '{network_variable}' already in network. Returning without overwriting."
            )
            return roadway_net

    """
    Verify inputs
    """

    if ("centroidconnect" not in roadway_net.links_df) & (
        "taz" not in roadway_net.links_df.roadway.unique()
    ) and centroidconnect_only:
        msg = "No variable specified for centroid connector, calculating centroidconnect first"
        WranglerLogger.error(msg)
        raise ValueError(msg)

    """
    Start actual process
    """
    link_attrs = copy.deepcopy(roadway_net.links_df.attrs)

    temp_links_gdf = roadway_net.links_df.copy()
    temp_links_gdf.crs = "EPSG:4326"
    temp_links_gdf = temp_links_gdf.to_crs(epsg=26915)

    if centroidconnect_only:
        WranglerLogger.info(f"Calculating {network_variable} for centroid connectors")
        temp_links_gdf[network_variable] = np.where(
            temp_links_gdf.centroidconnect == 1,
            temp_links_gdf.geometry.length / 1609.34,
            temp_links_gdf[network_variable],
        )
    else:
        WranglerLogger.info("Calculating distance for all links")
        temp_links_gdf[network_variable] = temp_links_gdf.geometry.length / 1609.34
        # overwrite 0 distance with 0.001 mile
        temp_links_gdf.loc[temp_links_gdf[network_variable] == 0, network_variable] = 0.001

    roadway_net.links_df[network_variable] = temp_links_gdf[network_variable]
    roadway_net.links_df.attrs = link_attrs

    return roadway_net


def create_ML_variable(
    roadway_net=None,
    network_variable="ML_lanes",
    overwrite=False,
):
    """Created ML lanes placeholder for project to write out ML changes.

    ML lanes default to 0, ML info comes from cube LOG file and store in project cards

    Args:
        roadway_net: RoadwayNetwork whose links_df will be updated.
        network_variable: Name of the ML lanes column to create. Default "ML_lanes".
        overwrite (Bool): True if overwriting existing variable in network.  Default to False.

    Returns:
        None
    """
    link_attrs = copy.deepcopy(roadway_net.links_df.attrs)

    if network_variable in roadway_net.links_df:
        if overwrite:
            WranglerLogger.info(
                f"Overwriting existing ML Variable '{network_variable}' already in network"
            )
            roadway_net.links_df[network_variable] = 0
        else:
            WranglerLogger.info(
                f"ML Variable '{network_variable}' already in network. Returning without overwriting."
            )
            return roadway_net

    """
    Verify inputs
    """

    WranglerLogger.info(f"Finished creating ML lanes variable: {network_variable}")
    roadway_net.links_df.attrs = link_attrs

    return roadway_net


def create_hov_corridor_variable(
    roadway_net=None,
    network_variable="segment_id",
    overwrite=False,
):
    """Created hov corridor placeholder for project to write out corridor changes.

    hov corridor id default to 0, its info comes from cube LOG file and store in project cards

    Args:
        roadway_net: RoadwayNetwork whose links_df will be updated.
        network_variable: Name of the HOV corridor column to create. Default "segment_id".
        overwrite (Bool): True if overwriting existing variable in network.  Default to False.

    Returns:
        None
    """
    link_attrs = copy.deepcopy(roadway_net.links_df.attrs)

    if network_variable in roadway_net.links_df:
        if overwrite:
            WranglerLogger.info(
                f"Overwriting existing hov corridor Variable '{network_variable}' already in network"
            )
        else:
            WranglerLogger.info(
                f"Hov corridor Variable '{network_variable}' already in network. Returning without overwriting."
            )
            return roadway_net

    """
    Verify inputs
    """

    roadway_net.links_df[network_variable] = 0

    WranglerLogger.info(f"Finished creating hov corridor variable: {network_variable}")
    roadway_net.links_df.attrs = link_attrs

    return roadway_net


def create_managed_variable(
    roadway_net=None,
    network_variable="managed",
    overwrite=False,
):
    """Created placeholder for project to write out managed.

    managed default to 0, its info comes from cube LOG file and store in project cards

    Args:
        roadway_net: RoadwayNetwork whose links_df will be updated.
        network_variable: Name of the managed column to create. Default "managed".
        overwrite (Bool): True if overwriting existing variable in network.  Default to False.

    Returns:
        None
    """
    link_attrs = copy.deepcopy(roadway_net.links_df.attrs)

    if network_variable in roadway_net.links_df:
        if overwrite:
            WranglerLogger.info(
                f"Overwriting existing managed Variable '{network_variable}' already in network"
            )
        else:
            WranglerLogger.info(
                f"Managed Variable '{network_variable}' already in network. Returning without overwriting."
            )
            return roadway_net

    """
    Verify inputs
    """

    roadway_net.links_df[network_variable] = 0

    WranglerLogger.info(f"Finished creating managed variable: {network_variable}")
    roadway_net.links_df.attrs = link_attrs

    return roadway_net


def add_variable_using_shst_reference(
    roadway_net=None,
    var_shst_csvdata=None,
    shst_csv_variable=None,
    network_variable=None,
    network_var_type=int,
    overwrite=False,
):
    """Join network links with source data, via SHST API node match result.

    Args:
        roadway_net: RoadwayNetwork whose links_df will be updated.
        var_shst_csvdata (str): File path to SHST API return.
        shst_csv_variable (str): Variable name in the source data.
        network_variable (str): Name of the variable that should be written to.
        network_var_type : Variable type in the written network.
        overwrite (bool): True is overwriting existing variable. Default to False.

    Returns:
        None

    """
    link_attrs = copy.deepcopy(roadway_net.links_df.attrs)

    WranglerLogger.info(
        f"Adding Variable {network_variable} using Shared Streets Reference from {var_shst_csvdata}"
    )

    var_shst_df = pd.read_csv(var_shst_csvdata)
    # there are aadt = 0 in the counts, drop them
    var_shst_df = var_shst_df[var_shst_df[shst_csv_variable] > 0].copy()
    # count station to shared street match - there are many-to-one matches, keep just one match
    var_shst_df = var_shst_df.drop_duplicates(subset=["shstReferenceId"])

    if "shstReferenceId" not in var_shst_df.columns:
        msg = f"'shstReferenceId' required but not found in {var_shst_data}"
        WranglerLogger.error(msg)
        raise ValueError(msg)

    if shst_csv_variable not in var_shst_df.columns:
        msg = f"{shst_csv_variable} required but not found in {var_shst_data}"
        WranglerLogger.error(msg)
        raise ValueError(msg)

    join_gdf = pd.merge(
        roadway_net.links_df,
        var_shst_df[["shstReferenceId", shst_csv_variable]],
        how="left",
        on="shstReferenceId",
    )

    join_gdf[shst_csv_variable] = join_gdf[shst_csv_variable].fillna(0)

    if network_variable in roadway_net.links_df.columns and not overwrite:
        join_gdf.loc[join_gdf[network_variable] == 0, network_variable] = join_gdf[
            shst_csv_variable
        ].astype(network_var_type)
    else:
        join_gdf[network_variable] = join_gdf[shst_csv_variable].astype(network_var_type)

    roadway_net.links_df[network_variable] = join_gdf[network_variable]

    # MN and WI counts are vehicles using the segment in both directions, no directional counts
    # we will make sure both direction has the same daily AADT
    dir_link_count_df = roadway_net.links_df[
        (roadway_net.links_df[network_variable] > 0)
        & (roadway_net.links_df["drive_access"] == True)
    ][["A", "B", network_variable]].copy()
    reverse_dir_link_count_df = dir_link_count_df.rename(columns={"A": "B", "B": "A"}).copy()

    link_count_df = pd.concat(
        [dir_link_count_df, reverse_dir_link_count_df],
        sort=False,
        ignore_index=True,
    )
    link_count_df = link_count_df.drop_duplicates(subset=["A", "B"])

    roadway_net.links_df = pd.merge(
        roadway_net.links_df.drop(network_variable, axis=1),
        link_count_df[["A", "B", network_variable]],
        how="left",
        on=["A", "B"],
    )
    roadway_net.links_df[network_variable] = roadway_net.links_df[network_variable].fillna(0)
    WranglerLogger.info(f"Added variable: {network_variable} using Shared Streets Reference")
    roadway_net.links_df.attrs = link_attrs

    return roadway_net


def convert_types(roadway_net=None, parameters=None):
    """Coerce link and node column types to the Cube network schema.

    Replaces the legacy ``convert_int``, ``convert_bool``, and ``fill_na``
    functions.  Column types are defined in
    :class:`~cube_wrangler.models.tables.CubeLinksTable` and
    :class:`~cube_wrangler.models.tables.CubeNodesTable`; coercion is handled
    by :func:`~cube_wrangler.utils.models.coerce_df_to_model`.

    Args:
        roadway_net: RoadwayNetwork whose ``links_df`` and ``nodes_df`` will
            be coerced in-place.
        parameters: Unused; retained for backward-compatible call sites.

    Returns:
        The same ``roadway_net`` object with coerced DataFrames.
    """
    WranglerLogger.info("Coercing column types to Cube network schema")
    roadway_net.links_df = coerce_df_to_model(roadway_net.links_df, CubeLinksTable)
    roadway_net.nodes_df = coerce_df_to_model(roadway_net.nodes_df, CubeNodesTable)
    return roadway_net


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------


def convert_int(roadway_net=None, parameters=None, int_col_names=None):
    """Convert integer columns.

    .. deprecated::
        Use :func:`convert_types` instead.  Column types are now encoded in
        :class:`~cube_wrangler.models.tables.CubeLinksTable`.
    """
    WranglerLogger.warning("convert_int() is deprecated; call convert_types() instead.")
    return convert_types(roadway_net=roadway_net, parameters=parameters)


def convert_bool(roadway_net=None, parameters=None, bool_col_names=None):
    """Convert boolean columns.

    .. deprecated::
        Use :func:`convert_types` instead.  Column types are now encoded in
        :class:`~cube_wrangler.models.tables.CubeLinksTable`.
    """
    WranglerLogger.warning("convert_bool() is deprecated; call convert_types() instead.")
    return convert_types(roadway_net=roadway_net, parameters=parameters)


def fill_na(roadway_net=None, parameters=None):
    """Fill NA values for numeric columns.

    .. deprecated::
        Use :func:`convert_types` instead.  Column defaults are now encoded in
        :class:`~cube_wrangler.models.tables.CubeLinksTable`.
    """
    WranglerLogger.warning("fill_na() is deprecated; call convert_types() instead.")
    return convert_types(roadway_net=roadway_net, parameters=parameters)


def rename_variables_for_dbf(
    input_df=None,
    parameters=None,
    variable_crosswalk: str | None = None,
    output_variables: list | None = None,
    convert_geometry_to_xy=False,
):
    """Rename attributes for DBF/SHP, make sure length within 10 chars.

    Args:
        input_df (dataframe): Network standard DataFrame.
        parameters: Parameters instance providing net_to_dbf_crosswalk if not given.
        variable_crosswalk (str): File path to variable name crosswalk from network standard to DBF names.
        output_variables (list): List of strings for DBF variables.
        convert_geometry_to_xy (bool): True if converting node geometry to X/Y

    Returns:
        dataframe

    """
    WranglerLogger.info("Renaming variables so that they are DBF-safe")

    """
    Verify inputs
    """

    variable_crosswalk = (
        variable_crosswalk if variable_crosswalk else parameters.net_to_dbf_crosswalk
    )

    output_variables = output_variables if output_variables else parameters.output_variables

    """
    Start actual process
    """

    crosswalk_df = pd.read_csv(variable_crosswalk)
    WranglerLogger.debug(f"Variable crosswalk: {variable_crosswalk} \n {crosswalk_df}")
    net_to_dbf_dict = dict(zip(crosswalk_df["net"], crosswalk_df["dbf"], strict=False))

    dbf_name_list = []

    dbf_df = copy.deepcopy(input_df)

    # only write out variables that we specify
    # if variable is specified in the crosswalk, rename it to that variable
    for c in dbf_df.columns:
        if c in output_variables:
            try:
                dbf_df = dbf_df.rename(columns={c: net_to_dbf_dict[c]})
                dbf_name_list += [net_to_dbf_dict[c]]
            except:
                dbf_name_list += [c]

    if "geometry" in dbf_df.columns and str(dbf_df["geometry"].iloc[0].geom_type) == "Point":
        dbf_df["X"] = dbf_df.geometry.apply(lambda g: g.x)
        dbf_df["Y"] = dbf_df.geometry.apply(lambda g: g.y)
        if "X" not in dbf_name_list:
            dbf_name_list += ["X", "Y"]

    WranglerLogger.debug("DBF Variables: {}".format(",".join(dbf_name_list)))

    return dbf_df[dbf_name_list]


def write_roadway_as_shp(
    roadway_net=None,
    parameters=None,
    node_output_variables: list | None = None,
    link_output_variables: list | None = None,
    data_to_csv: bool = True,
    data_to_dbf: bool = False,
    output_link_shp: str | None = None,
    output_node_shp: str | None = None,
    output_link_csv: str | None = None,
    output_node_csv: str | None = None,
    export_drive_only: bool = False,
):
    """Write out dbf/shp for cube.  Write out csv in addition to shp with full length variable names.

    Args:
        roadway_net: RoadwayNetwork to write out.
        parameters: Parameters instance providing output path defaults.
        node_output_variables (list): List of strings for node output variables.
        link_output_variables (list): List of strings for link output variables.
        data_to_csv (bool): True if write network in csv format.
        data_to_dbf (bool): True if write network in dbf/shp format.
        output_link_shp (str): File path to output link dbf/shp.
        output_node_shp (str): File path to output node dbf/shp.
        output_link_csv (str): File path to output link csv.
        output_node_csv (str): File path to output node csv.
        export_drive_only (bool) : True if write out drive links/nodes only.

    Returns:
        None
    """
    WranglerLogger.info("Writing Network as Shapefile")
    WranglerLogger.debug(
        "Output Variables: \n - {}".format("\n - ".join(parameters.output_variables))
    )

    """
    Verify inputs
    """

    WranglerLogger.debug(
        "Network Link Variables: \n - {}".format("\n - ".join(roadway_net.links_df.columns))
    )
    WranglerLogger.debug(
        "Network Node Variables: \n - {}".format("\n - ".join(roadway_net.nodes_df.columns))
    )

    link_output_variables = (
        link_output_variables
        if link_output_variables
        else [c for c in roadway_net.links_df.columns if c in parameters.output_variables]
    )

    node_output_variables = (
        node_output_variables
        if node_output_variables
        else [c for c in roadway_net.nodes_df.columns if c in parameters.output_variables]
    )

    # unless specified that all the data goes to the DBF, only output A and B
    dbf_link_output_variables = (
        link_output_variables if data_to_dbf else ["A", "B", "shape_id", "geometry"]
    )

    output_link_shp = output_link_shp if output_link_shp else parameters.output_link_shp

    output_node_shp = output_node_shp if output_node_shp else parameters.output_node_shp

    output_link_csv = output_link_csv if output_link_csv else parameters.output_link_csv

    output_node_csv = output_node_csv if output_node_csv else parameters.output_node_csv

    """
    Start Process
    """

    WranglerLogger.info("Renaming DBF Node Variables")
    nodes_dbf_df = rename_variables_for_dbf(
        input_df=roadway_net.nodes_df,
        parameters=parameters,
        output_variables=node_output_variables,
    )
    WranglerLogger.info("Renaming DBF Link Variables")
    links_dbf_df = rename_variables_for_dbf(
        input_df=roadway_net.links_df,
        parameters=parameters,
        output_variables=dbf_link_output_variables,
    )

    links_dbf_df = gpd.GeoDataFrame(links_dbf_df, geometry=links_dbf_df["geometry"])

    if export_drive_only == True:
        nodes_dbf_df = nodes_dbf_df[nodes_dbf_df.drive_node == 1].copy()
        links_dbf_df = links_dbf_df[links_dbf_df.drive == 1].copy()

    WranglerLogger.info(f"Writing Node Shapes:\n - {output_node_shp}")
    nodes_dbf_df.to_file(output_node_shp)
    WranglerLogger.info(f"Writing Link Shapes:\n - {output_link_shp}")
    links_dbf_df.to_file(output_link_shp)

    if data_to_csv:
        WranglerLogger.info(
            f"Writing Network Data to CSVs:\n - {output_link_csv}\n - {output_node_csv}"
        )
        roadway_net.links_df[link_output_variables].to_csv(output_link_csv, index=False)
        roadway_net.nodes_df[node_output_variables].to_csv(output_node_csv, index=False)


def write_roadway_as_fixedwidth(
    roadway_net=None,
    parameters=None,
    zones: int | None = None,
    node_output_variables: list | None = None,
    link_output_variables: list | None = None,
    output_link_txt: str | None = None,
    output_node_txt: str | None = None,
    output_link_header_width_txt: str | None = None,
    output_node_header_width_txt: str | None = None,
    output_cube_network_script: str | None = None,
    drive_only: bool = False,
):
    """Writes out fixed width file.

    This function does:
    1. write out link and node fixed width data files for cube.
    2. write out header and width correspondence.
    3. write out cube network building script with header and width specification.

    Args:
        roadway_net: RoadwayNetwork to write out.
        parameters: Parameters instance providing output path defaults.
        zones: Number of zones (TAZ) in the network. Defaults to parameters.zones.
        node_output_variables (list): list of node variable names.
        link_output_variables (list): list of link variable names.
        output_link_txt (str): File path to output link database.
        output_node_txt (str): File path to output node database.
        output_link_header_width_txt (str): File path to link column width records.
        output_node_header_width_txt (str): File path to node column width records.
        output_cube_network_script (str): File path to CUBE network building script.
        drive_only (bool): If True, only writes drive nodes and links

    Returns:
        None

    """
    """
    Verify inputs
    """

    WranglerLogger.debug(
        "Network Link Variables: \n - {}".format("\n - ".join(roadway_net.links_df.columns))
    )
    WranglerLogger.debug(
        "Network Node Variables: \n - {}".format("\n - ".join(roadway_net.nodes_df.columns))
    )

    zones = zones if zones else parameters.zones

    link_output_variables = (
        link_output_variables
        if link_output_variables
        else [c for c in parameters.output_variables if c in roadway_net.links_df.columns]
    )

    node_output_variables = (
        node_output_variables
        if node_output_variables
        else [c for c in parameters.output_variables if c in roadway_net.nodes_df.columns]
    )

    output_link_txt = output_link_txt if output_link_txt else parameters.output_link_txt

    output_node_txt = output_node_txt if output_node_txt else parameters.output_node_txt

    output_link_header_width_txt = (
        output_link_header_width_txt
        if output_link_header_width_txt
        else parameters.output_link_header_width_txt
    )

    output_node_header_width_txt = (
        output_node_header_width_txt
        if output_node_header_width_txt
        else parameters.output_node_header_width_txt
    )

    output_cube_network_script = (
        output_cube_network_script
        if output_cube_network_script
        else parameters.output_cube_network_script
    )

    """
    Start Process
    """
    # make sure nodes_df X and Y are in the model crs
    # check if the nodes_df is in the output crs
    if roadway_net.nodes_df.crs != parameters.output_epsg:
        roadway_net.nodes_df = roadway_net.nodes_df.to_crs(parameters.output_epsg)
        # convert geometry to X and Y
        roadway_net.nodes_df["X"] = roadway_net.nodes_df.geometry.x
        roadway_net.nodes_df["Y"] = roadway_net.nodes_df.geometry.y

    # convert boolean columns to 1/0
    bool_link_col = [col for col in parameters.bool_col if col in roadway_net.links_df.columns]
    bool_node_col = [col for col in parameters.bool_col if col in roadway_net.nodes_df.columns]

    link_ff_df, link_max_width_dict = dataframe_to_fixed_width(
        roadway_net.links_df[link_output_variables], bool_link_col
    )

    if drive_only:
        link_ff_df = link_ff_df.loc[link_ff_df["drive_access"] == 1]

    WranglerLogger.info("Writing out link database")

    link_ff_df.to_csv(output_link_txt, sep=";", index=False, header=False)

    # write out header and width correspondence
    WranglerLogger.info("Writing out link header and width ----")
    link_max_width_df = DataFrame(list(link_max_width_dict.items()), columns=["header", "width"])
    link_max_width_df.to_csv(output_link_header_width_txt, index=False)

    # make sure model_node_id is renamed to N for CUBE
    if ("model_node_id" in roadway_net.nodes_df.columns) & (
        "N" not in roadway_net.nodes_df.columns
    ):
        WranglerLogger.info("Renaming model_node_id to N for fixed width conversion")
        roadway_net.nodes_df = roadway_net.nodes_df.rename(columns={"model_node_id": "N"})
        # remove model_node_id from node_output_variables
        if "model_node_id" in node_output_variables:
            node_output_variables.remove("model_node_id")

    # make sure N is in the node output variables for CUBE
    if "N" not in node_output_variables:
        node_output_variables.append("N")

    node_ff_df, node_max_width_dict = dataframe_to_fixed_width(
        roadway_net.nodes_df[node_output_variables], bool_node_col
    )
    WranglerLogger.info("Writing out node database")

    if drive_only:
        node_ff_df = node_ff_df.loc[node_ff_df["drive_node"] == 1]

    node_ff_df.to_csv(output_node_txt, sep=";", index=False, header=False)

    # write out header and width correspondence
    WranglerLogger.info("Writing out node header and width")
    node_max_width_df = DataFrame(list(node_max_width_dict.items()), columns=["header", "width"])
    node_max_width_df.to_csv(output_node_header_width_txt, index=False)

    # write out cube script
    s = 'RUN PGM = NETWORK MSG = "Read in network from fixed width file" \n'
    s += f'FILEI LINKI[1] = "{output_link_txt}",'
    start_pos = 1
    for i in range(len(link_max_width_df)):
        s += " VAR=" + link_max_width_df.header.iloc[i]

        if roadway_net.links_df.dtypes.loc[link_max_width_df.header.iloc[i]] == "O":
            s += "(C" + str(link_max_width_df.width.iloc[i]) + ")"

        s += ", BEG=" + str(start_pos) + ", LEN=" + str(link_max_width_df.width.iloc[i]) + ","

        start_pos += link_max_width_df.width.iloc[i] + 1

    s = s[:-1]
    s += "\n"
    s += f'FILEI NODEI[1] = "{output_node_txt}",'
    start_pos = 1
    for i in range(len(node_max_width_df)):
        s += " VAR=" + node_max_width_df.header.iloc[i]

        if roadway_net.nodes_df.dtypes.loc[node_max_width_df.header.iloc[i]] == "O":
            s += "(C" + str(node_max_width_df.width.iloc[i]) + ")"

        s += ", BEG=" + str(start_pos) + ", LEN=" + str(node_max_width_df.width.iloc[i]) + ","

        start_pos += node_max_width_df.width.iloc[i] + 1

    s = s[:-1]
    s += "\n"
    s += 'FILEO NETO = "complete_network.net" \n\n'
    s += f"ZONES = {zones} \n\n"
    # trim whitespace from string columns
    for col in parameters.string_col:
        if col in link_max_width_dict:
            s += f"{col} = LTRIM(TRIM({col})) \n"
        if col in node_max_width_dict:
            s += f"{col} = LTRIM(TRIM({col})) \n"
    if "ROADWAY" in link_max_width_dict:
        s += "ROADWAY = LTRIM(TRIM(ROADWAY)) \n"
    if "NAME" in link_max_width_dict:
        s += "NAME = LTRIM(TRIM(NAME)) \n"
    s += "\n \nENDRUN"

    with Path(output_cube_network_script).open("w") as f:
        f.write(s)


# this should be moved to util
# @staticmethod
def dataframe_to_fixed_width(df, bool_col):
    """Convert dataframe to fixed width format, geometry column will not be transformed.

    Args:
        df: pandas DataFrame to convert to fixed-width format.
        bool_col: list of boolean column names to cast to int before conversion.

    Returns:
        pandas dataframe:  dataframe with fixed width for each column.
        dict: dictionary with columns names as keys, column width as values.
    """
    WranglerLogger.info("Starting fixed width conversion")

    # get the max length for each variable column
    max_width_dict = {
        col: df[col].dropna().astype(str).str.len().max()
        for col in df.columns
        if col != "geometry"
    }

    # CUBE does not like column LEN=0, so we set them to 1
    for col, width in max_width_dict.items():
        if width == 0:
            max_width_dict[col] = 1

    fw_df = df.copy()
    if "geometry" in df.columns:
        fw_df = fw_df.drop("geometry", axis=1)
    for col in bool_col:
        if col in fw_df.columns:
            fw_df[col] = fw_df[col].astype(int)
        else:
            WranglerLogger.debug(
                f"Boolean column {col} not found in output fixed width DataFrame."
            )

    for c in fw_df.columns:
        fw_df[c] = fw_df[c].apply(lambda x: str(x))
        fw_df["pad"] = fw_df[c].apply(lambda x, _c=c: " " * (max_width_dict[_c] - len(x)))
        fw_df[c] = fw_df.apply(lambda x, _c=c: x["pad"] + x[_c], axis=1)

    return fw_df, max_width_dict


# @staticmethod
def read_match_result(path):
    """Reads the shst geojson match returns.

    Returns shst dataframe.

    Reading lots of same type of file and concatenating them into a single DataFrame.

    Args:
        path (str): File path to SHST match results.

    Returns:
        geodataframe: geopandas geodataframe

    ##todo
    not sure why we need, but should be in utilities not this class
    """
    refId_gdf = DataFrame()
    refid_file = glob.glob(path)
    for i in refid_file:
        new = gpd.read_file(i)
        refId_gdf = pd.concat([refId_gdf, new], ignore_index=True, sort=False)
    return refId_gdf


# @staticmethod
def get_attribute(
    links_df,
    join_key,  # either "shstReferenceId", or "shstGeometryId", tests showed the latter gave better coverage
    source_shst_ref_df,  # source shst refId
    source_gdf,  # source dataframe
    field_name,  # , # targetted attribute from source
):
    """Gets attribute from source data using SHST match result.

    Args:
        links_df (dataframe): The network dataframe that new attribute should be written to.
        join_key (str): SHST ID variable name used to join source data with network dataframe.
        source_shst_ref_df (str): File path to source data SHST match result.
        source_gdf (str): File path to source data.
        field_name (str): Name of the attribute to get from source data.

    Returns:
        None
    """
    # join based on shared streets geometry ID
    # pp_link_id is shared streets match return
    # source_ink_id is mrcc
    WranglerLogger.debug(
        f"source ShSt rename_variables_for_dbf columns\n{source_shst_ref_df.columns}"
    )
    WranglerLogger.debug(f"source gdf columns\n{source_gdf.columns}")
    # end up with OSM network with the MRCC Link ID
    # could also do with route_sys...would that be quicker?
    join_refId_df = pd.merge(
        links_df,
        source_shst_ref_df[[join_key, "pp_link_id", "score"]].rename(
            columns={"pp_link_id": "source_link_id", "score": "source_score"}
        ),
        how="left",
        on=join_key,
    )

    # joined with MRCC dataframe to get route_sys

    join_refId_df = pd.merge(
        join_refId_df,
        source_gdf[["LINK_ID", field_name]].rename(columns={"LINK_ID": "source_link_id"}),
        how="left",
        on="source_link_id",
    )

    # drop duplicated records with same field value

    join_refId_df = join_refId_df.drop_duplicates(
        subset=["model_link_id", "shstReferenceId", field_name]
    )

    # more than one match, take the best score

    join_refId_df = join_refId_df.sort_values(
        by=["model_link_id", "source_score"],
        ascending=True,
        na_position="first",
    )

    join_refId_df = join_refId_df.drop_duplicates(subset=["model_link_id"], keep="last")

    # self.links_df[field_name] = join_refId_df[field_name]

    return join_refId_df[[*links_df.columns.tolist(), field_name, "source_link_id"]]
