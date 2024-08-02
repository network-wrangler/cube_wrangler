import copy
from typing import Optional, Union

import geopandas as gpd
import pandas as pd

from geopandas import GeoDataFrame
from pandas import DataFrame
import numpy as np

from network_wrangler.roadway.network import RoadwayNetwork
from .parameters import Parameters
from .logger import WranglerLogger


# class ModelRoadwayNetwork(RoadwayNetwork):
#     """
#     Subclass of network_wrangler class :ref:`RoadwayNetwork <network_wrangler:RoadwayNetwork>`

#     A representation of the physical roadway network and its properties.
#     """

#     def __init__(
#         self,
#         nodes: GeoDataFrame,
#         links: DataFrame,
#         shapes: GeoDataFrame,
#         parameters: Union[Parameters, dict] = {},
#     ):
#         """
#         Constructor

#         Args:
#             nodes: geodataframe of nodes
#             links: dataframe of links
#             shapes: geodataframe of shapes
#             parameters: dictionary of parameter settings (see Parameters class) or an instance of Parameters. If not specified, will use default parameters.

#         """
#         super().__init__(nodes, links, shapes)

#         # will have to change if want to alter them
#         if type(parameters) is dict:
#             self.parameters = Parameters(**parameters)
#         elif isinstance(parameters, Parameters):
#             # self.parameters = Parameters(**parameters.__dict__)
#             self.parameters = parameters
#         else:
#             msg = "Parameters should be a dict or instance of Parameters: found {} which is of type:{}".format(
#                 parameters, type(parameters)
#             )
#             WranglerLogger.error(msg)
#             raise ValueError(msg)

#     @staticmethod
#     def read(
#         link_file: str,
#         node_file: str,
#         shape_file: str,
#         fast: bool = False,
#         recalculate_calculated_variables=False,
#         parameters: Union[dict, Parameters] = {},
#     ):
#         """
#         Reads in links and nodes network standard.

#         Args:
#             link_file (str): File path to link json.
#             node_file (str): File path to node geojson.
#             shape_file (str): File path to link true shape geojson
#             fast (bool): boolean that will skip validation to speed up read time.
#             recalculate_calculated_variables (bool): calculates fields from spatial joins, etc.
#             recalculate_distance (bool):  re-calculates distance.
#             parameters: dictionary of parameter settings (see Parameters class) or an instance of Parameters. If not specified, will use default parameters.

#         Returns:
#             ModelRoadwayNetwork
#         """
#         # road_net =  super().read(link_file, node_file, shape_file, fast=fast)
#         road_net = RoadwayNetwork.read(link_file, node_file, shape_file, fast=fast)

#         m_road_net = ModelRoadwayNetwork(
#             road_net.nodes_df,
#             road_net.links_df,
#             road_net.shapes_df,
#             parameters=parameters,
#         )

#         if recalculate_calculated_variables:
#             m_road_net.create_calculated_variables()

#         return m_road_net

#     @staticmethod
#     def from_RoadwayNetwork(
#         roadway_network_object, parameters: Union[dict, Parameters] = {}
#     ):
#         """
#         RoadwayNetwork to ModelRoadwayNetwork

#         Args:
#             roadway_network_object (RoadwayNetwork).
#             parameters: dictionary of parameter settings (see Parameters class) or an instance of Parameters. If not specified, will use default parameters.

#         Returns:
#             ModelRoadwayNetwork
#         """
#         return ModelRoadwayNetwork(
#             roadway_network_object.nodes_df,
#             roadway_network_object.links_df,
#             roadway_network_object.shapes_df,
#             parameters=parameters,
#         )

def split_properties_by_time_period_and_category(roadway_net=None, parameters=None, properties_to_split=None):
    """
    Splits properties by time period, assuming a variable structure of

    Args:
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
                    roadway_net.links_df[
                        out_var + "_" + time_suffix + "_" + category_suffix
                    ] = 0
            elif params.get("time_periods"):
                for time_suffix in params["time_periods"]:
                    roadway_net.links_df[out_var + "_" + time_suffix] = 0
        elif params.get("time_periods") and params.get("categories"):
            for time_suffix, category_suffix in itertools.product(
                params["time_periods"], params["categories"]
            ):
                roadway_net.links_df[
                    out_var + "_" + category_suffix + "_" + time_suffix
                ] = roadway_net.get_property_by_timespan_and_group(
                    params["v"],
                    category=params["categories"][category_suffix],
                    timespan=params["time_periods"][time_suffix],
                )[params["v"]]
        elif params.get("time_periods"):
            for time_suffix in params["time_periods"]:
                roadway_net.links_df[
                    out_var + "_" + time_suffix
                ] = roadway_net.get_property_by_timespan_and_group(
                    params["v"],
                    category=None,
                    timespan=params["time_periods"][time_suffix],
                )[params["v"]]
        else:
            raise ValueError(
                "Shoudn't have a category without a time period: {}".format(params)
            )
    return roadway_net
    
# def create_calculated_variables(self):
#     """
#     Creates calculated roadway variables.

#     Args:
#         None
#     """
#     WranglerLogger.info("Creating calculated roadway variables.")
#     self.calculate_distance_miles(overwrite=True)

def calculate_distance_miles(roadway_net=None, network_variable="distance", overwrite=False):
    """
    calculate link distance in miles

    Args:
        overwrite (Bool): True if overwriting existing variable in network.  Default to False.

    Returns:
        None

    """

    if network_variable in roadway_net.links_df:
        if overwrite or (roadway_net.links_df[network_variable].isnull().any()):
            WranglerLogger.info(
                "Overwriting existing distance Variable '{}' already in network".format(
                    network_variable
                )
            )
        else:
            WranglerLogger.info(
                "Distance Variable '{}' already in network. Returning without overwriting.".format(
                    network_variable
                )
            )
            return roadway_net

    """
    Start actual process
    """

    temp_links_gdf = roadway_net.links_df.copy()
    temp_links_gdf.crs = "EPSG:4326"
    temp_links_gdf = temp_links_gdf.to_crs(epsg=26915)

    WranglerLogger.info(
        "Calculating distance in miles for all links".format(network_variable)
    )
    temp_links_gdf[network_variable] = temp_links_gdf.geometry.length / 1609.34
    # overwrite 0 distance with 0.001 mile
    temp_links_gdf[network_variable] = np.where(
        temp_links_gdf[network_variable] == 0,
        0.001,
        temp_links_gdf[network_variable],
    )

    roadway_net.links_df[network_variable] = temp_links_gdf[network_variable]

    return roadway_net

def calculate_distance(
    roadway_net=None, network_variable="distance", centroidconnect_only=False, overwrite=False
):
    """
    calculate link distance in miles

    Args:
        centroidconnect_only (Bool):  True if calculating distance for centroidconnectors only.  Default to True.
        overwrite (Bool): True if overwriting existing variable in network.  Default to False.

    Returns:
        None

    """

    if network_variable in roadway_net.links_df:
        if overwrite:
            WranglerLogger.info(
                "Overwriting existing distance Variable '{}' already in network".format(
                    network_variable
                )
            )
        else:
            WranglerLogger.info(
                "Distance Variable '{}' already in network. Returning without overwriting.".format(
                    network_variable
                )
            )
            return roadway_net

    """
    Verify inputs
    """

    if ("centroidconnect" not in roadway_net.links_df) & (
        "taz" not in roadway_net.links_df.roadway.unique()
    ):
        if centroidconnect_only:
            msg = "No variable specified for centroid connector, calculating centroidconnect first"
            WranglerLogger.error(msg)
            raise ValueError(msg)

    """
    Start actual process
    """

    temp_links_gdf = roadway_net.links_df.copy()
    temp_links_gdf.crs = "EPSG:4326"
    temp_links_gdf = temp_links_gdf.to_crs(epsg=26915)

    if centroidconnect_only:
        WranglerLogger.info(
            "Calculating {} for centroid connectors".format(network_variable)
        )
        temp_links_gdf[network_variable] = np.where(
            temp_links_gdf.centroidconnect == 1,
            temp_links_gdf.geometry.length / 1609.34,
            temp_links_gdf[network_variable],
        )
    else:
        WranglerLogger.info(
            "Calculating distance for all links".format(network_variable)
        )
        temp_links_gdf[network_variable] = temp_links_gdf.geometry.length / 1609.34
        # overwrite 0 distance with 0.001 mile
        temp_links_gdf.loc[
            temp_links_gdf[network_variable] == 0, network_variable
        ] = 0.001

    roadway_net.links_df[network_variable] = temp_links_gdf[network_variable]

    return roadway_net

def create_ML_variable(
    roadway_net=None,
    network_variable="ML_lanes",
    overwrite=False,
):
    """
    Created ML lanes placeholder for project to write out ML changes

    ML lanes default to 0, ML info comes from cube LOG file and store in project cards

    Args:
        overwrite (Bool): True if overwriting existing variable in network.  Default to False.

    Returns:
        None
    """
    if network_variable in roadway_net.links_df:
        if overwrite:
            WranglerLogger.info(
                "Overwriting existing ML Variable '{}' already in network".format(
                    network_variable
                )
            )
            roadway_net.links_df[network_variable] = int(0)
        else:
            WranglerLogger.info(
                "ML Variable '{}' already in network. Returning without overwriting.".format(
                    network_variable
                )
            )
            return roadway_net

    """
    Verify inputs
    """

    WranglerLogger.info(
        "Finished creating ML lanes variable: {}".format(network_variable)
    )
    return roadway_net

def create_hov_corridor_variable(
    roadway_net=None,
    network_variable="segment_id",
    overwrite=False,
):
    """
    Created hov corridor placeholder for project to write out corridor changes

    hov corridor id default to 0, its info comes from cube LOG file and store in project cards

    Args:
        overwrite (Bool): True if overwriting existing variable in network.  Default to False.

    Returns:
        None
    """
    if network_variable in roadway_net.links_df:
        if overwrite:
            WranglerLogger.info(
                "Overwriting existing hov corridor Variable '{}' already in network".format(
                    network_variable
                )
            )
        else:
            WranglerLogger.info(
                "Hov corridor Variable '{}' already in network. Returning without overwriting.".format(
                    network_variable
                )
            )
            return roadway_net

    """
    Verify inputs
    """

    roadway_net.links_df[network_variable] = int(0)

    WranglerLogger.info(
        "Finished creating hov corridor variable: {}".format(network_variable)
    )
    return roadway_net

def create_managed_variable(
    roadway_net=None,
    network_variable="managed",
    overwrite=False,
):
    """
    Created placeholder for project to write out managed

    managed default to 0, its info comes from cube LOG file and store in project cards

    Args:
        overwrite (Bool): True if overwriting existing variable in network.  Default to False.

    Returns:
        None
    """
    if network_variable in roadway_net.links_df:
        if overwrite:
            WranglerLogger.info(
                "Overwriting existing managed Variable '{}' already in network".format(
                    network_variable
                )
            )
        else:
            WranglerLogger.info(
                "Managed Variable '{}' already in network. Returning without overwriting.".format(
                    network_variable
                )
            )
            return roadway_net

    """
    Verify inputs
    """

    roadway_net.links_df[network_variable] = int(0)

    WranglerLogger.info(
        "Finished creating managed variable: {}".format(network_variable)
    )
    return roadway_net

def add_variable_using_shst_reference(
    roadway_net=None,
    var_shst_csvdata=None,
    shst_csv_variable=None,
    network_variable=None,
    network_var_type=int,
    overwrite=False,
):
    """
    Join network links with source data, via SHST API node match result.

    Args:
        var_shst_csvdata (str): File path to SHST API return.
        shst_csv_variable (str): Variable name in the source data.
        network_variable (str): Name of the variable that should be written to.
        network_var_type : Variable type in the written network.
        overwrite (bool): True is overwriting existing variable. Default to False.

    Returns:
        None

    """
    WranglerLogger.info(
        "Adding Variable {} using Shared Streets Reference from {}".format(
            network_variable, var_shst_csvdata
        )
    )

    var_shst_df = pd.read_csv(var_shst_csvdata)
    # there are aadt = 0 in the counts, drop them
    var_shst_df = var_shst_df[var_shst_df[shst_csv_variable] > 0].copy()
    # count station to shared street match - there are many-to-one matches, keep just one match
    var_shst_df.drop_duplicates(subset=["shstReferenceId"], inplace=True)

    if "shstReferenceId" not in var_shst_df.columns:
        msg = "'shstReferenceId' required but not found in {}".format(var_shst_data)
        WranglerLogger.error(msg)
        raise ValueError(msg)

    if shst_csv_variable not in var_shst_df.columns:
        msg = "{} required but not found in {}".format(
            shst_csv_variable, var_shst_data
        )
        WranglerLogger.error(msg)
        raise ValueError(msg)

    join_gdf = pd.merge(
        roadway_net.links_df,
        var_shst_df[["shstReferenceId", shst_csv_variable]],
        how="left",
        on="shstReferenceId",
    )

    join_gdf[shst_csv_variable].fillna(0, inplace=True)

    if network_variable in roadway_net.links_df.columns and not overwrite:
        join_gdf.loc[join_gdf[network_variable] == 0, network_variable] = join_gdf[
            shst_csv_variable
        ].astype(network_var_type)
    else:
        join_gdf[network_variable] = join_gdf[shst_csv_variable].astype(
            network_var_type
        )

    roadway_net.links_df[network_variable] = join_gdf[network_variable]

    # MN and WI counts are vehicles using the segment in both directions, no directional counts
    # we will make sure both direction has the same daily AADT
    dir_link_count_df = roadway_net.links_df[
        (roadway_net.links_df[network_variable] > 0) & (roadway_net.links_df["drive_access"] == True)
    ][["A", "B", network_variable]].copy()
    reverse_dir_link_count_df = dir_link_count_df.rename(
        columns={"A": "B", "B": "A"}
    ).copy()

    link_count_df = pd.concat(
        [dir_link_count_df, reverse_dir_link_count_df],
        sort=False,
        ignore_index=True,
    )
    link_count_df.drop_duplicates(subset=["A", "B"], inplace=True)

    roadway_net.links_df = pd.merge(
        roadway_net.links_df.drop(network_variable, axis=1),
        link_count_df[["A", "B", network_variable]],
        how="left",
        on=["A", "B"],
    )
    roadway_net.links_df[network_variable].fillna(0, inplace=True)
    WranglerLogger.info(
        "Added variable: {} using Shared Streets Reference".format(network_variable)
    )
    return roadway_net

def convert_int(roadway_net=None, parameters=None, int_col_names=[]):
    """
    Convert integer columns
    """

    WranglerLogger.info("Converting variable type to MetCouncil standard")

    if not int_col_names:
        int_col_names = parameters.int_col

    ##Why are we doing this?
    # int_col_names.remove("lanes")

    for c in list(set(roadway_net.links_df.columns) & set(int_col_names)):
        roadway_net.links_df[c] = roadway_net.links_df[c].replace(np.nan, 0)
        # REPLACE BLANKS WITH ZERO FOR INTEGER COLUMNS
        roadway_net.links_df[c] = roadway_net.links_df[c].replace("", 0)
        try:
            roadway_net.links_df[c] = roadway_net.links_df[c].astype(int)
        except ValueError:
            try:
                roadway_net.links_df[c] = roadway_net.links_df[c].astype(float)
                roadway_net.links_df[c] = roadway_net.links_df[c].astype(int)
            except:
                msg = f"Could not convert column {c} to integer."
                WranglerLogger.error(msg)
                raise ValueError(msg)

    for c in list(set(roadway_net.nodes_df.columns) & set(int_col_names)):
        roadway_net.nodes_df[c] = roadway_net.nodes_df[c].replace(np.nan, 0)
        # REPLACE BLANKS WITH ZERO FOR INTEGER COLUMNS
        roadway_net.nodes_df[c] = roadway_net.nodes_df[c].replace("", 0)
        try:
            roadway_net.nodes_df[c] = roadway_net.nodes_df[c].astype(int)
        except ValueError:
            msg = f"Could not convert column {c} to integer."
            WranglerLogger.error(msg)
            raise ValueError(msg)
    
    return roadway_net

def convert_bool(roadway_net=None, parameters=None, bool_col_names=[]):
    """
    Convert boolean columns
    """

    WranglerLogger.info("Converting variable type to MetCouncil standard")

    if not bool_col_names:
        bool_col_names = parameters.bool_col_names

    for c in list(set(roadway_net.links_df.columns) & set(bool_col_names)):
        roadway_net.links_df[c] = roadway_net.links_df[c].replace(np.nan, False)
        # REPLACE BLANKS WITH ZERO FOR INTEGER COLUMNS
        roadway_net.links_df[c] = roadway_net.links_df[c].replace("", False)
        roadway_net.links_df[c] = roadway_net.links_df[c].replace("0", False)
        roadway_net.links_df[c] = roadway_net.links_df[c].replace("1", True)
        try:
            roadway_net.links_df[c] = roadway_net.links_df[c].astype(bool)
        except ValueError:
            msg = f"Could not convert column {c} to boolean."
            WranglerLogger.error(msg)
            raise ValueError(msg)

    for c in list(set(roadway_net.nodes_df.columns) & set(bool_col_names)):
        roadway_net.nodes_df[c] = roadway_net.nodes_df[c].replace(np.nan, False)
        roadway_net.nodes_df[c] = roadway_net.nodes_df[c].replace("", False)
        roadway_net.nodes_df[c] = roadway_net.nodes_df[c].replace("0", False)
        roadway_net.nodes_df[c] = roadway_net.nodes_df[c].replace("1", True)
        try:
            roadway_net.nodes_df[c] = roadway_net.nodes_df[c].astype(bool)
        except ValueError:
            msg = f"Could not convert column {c} to boolean."
            WranglerLogger.error(msg)
            raise ValueError(msg)
    
    return roadway_net


def fill_na(roadway_net=None, parameters=None):
    """
    Fill na values from create_managed_lane_network()
    """

    WranglerLogger.info("Filling nan for network from network wrangler")

    num_col = parameters.int_col + parameters.float_col

    for x in list(roadway_net.links_df.columns):
        if x in num_col:
            roadway_net.links_df[x].fillna(0, inplace=True)
            roadway_net.links_df[x] = roadway_net.links_df[x].apply(
                lambda k: 0 if k in [np.nan, "", float("nan"), "NaN"] else k
            )

        else:
            roadway_net.links_df[x].fillna("", inplace=True)

    for x in list(roadway_net.nodes_df.columns):
        if x in num_col:
            roadway_net.nodes_df[x].fillna(0, inplace=True)
        else:
            roadway_net.nodes_df[x].fillna("", inplace=True)
    return roadway_net

def rename_variables_for_dbf(
    input_df=None,
    parameters=None,
    variable_crosswalk: str = None,
    output_variables: list = None,
    convert_geometry_to_xy=False,
):
    """
    Rename attributes for DBF/SHP, make sure length within 10 chars.

    Args:
        input_df (dataframe): Network standard DataFrame.
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
        variable_crosswalk
        if variable_crosswalk
        else parameters.net_to_dbf_crosswalk
    )

    output_variables = (
        output_variables if output_variables else parameters.output_variables
    )

    """
    Start actual process
    """

    crosswalk_df = pd.read_csv(variable_crosswalk)
    WranglerLogger.debug(
        "Variable crosswalk: {} \n {}".format(variable_crosswalk, crosswalk_df)
    )
    net_to_dbf_dict = dict(zip(crosswalk_df["net"], crosswalk_df["dbf"]))

    dbf_name_list = []

    dbf_df = copy.deepcopy(input_df)

    # only write out variables that we specify
    # if variable is specified in the crosswalk, rename it to that variable
    for c in dbf_df.columns:
        if c in output_variables:
            try:
                dbf_df.rename(columns={c: net_to_dbf_dict[c]}, inplace=True)
                dbf_name_list += [net_to_dbf_dict[c]]
            except:
                dbf_name_list += [c]

    if "geometry" in dbf_df.columns:
        if str(dbf_df["geometry"].iloc[0].geom_type) == "Point":
            dbf_df["X"] = dbf_df.geometry.apply(lambda g: g.x)
            dbf_df["Y"] = dbf_df.geometry.apply(lambda g: g.y)
            if "X" not in dbf_name_list:
                dbf_name_list += ["X", "Y"]

    WranglerLogger.debug("DBF Variables: {}".format(",".join(dbf_name_list)))

    return dbf_df[dbf_name_list]

def write_roadway_as_shp(
    roadway_net=None,
    parameters=None,
    node_output_variables: list = None,
    link_output_variables: list = None,
    data_to_csv: bool = True,
    data_to_dbf: bool = False,
    output_link_shp: str = None,
    output_node_shp: str = None,
    output_link_csv: str = None,
    output_node_csv: str = None,
    export_drive_only: bool = False,
):
    """
    Write out dbf/shp for cube.  Write out csv in addition to shp with full length variable names.

    Args:
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
        "Output Variables: \n - {}".format(
            "\n - ".join(parameters.output_variables)
        )
    )

    """
    Verify inputs
    """

    WranglerLogger.debug(
        "Network Link Variables: \n - {}".format(
            "\n - ".join(roadway_net.links_df.columns)
        )
    )
    WranglerLogger.debug(
        "Network Node Variables: \n - {}".format(
            "\n - ".join(roadway_net.nodes_df.columns)
        )
    )

    link_output_variables = (
        link_output_variables
        if link_output_variables
        else [
            c
            for c in roadway_net.links_df.columns
            if c in parameters.output_variables
        ]
    )

    node_output_variables = (
        node_output_variables
        if node_output_variables
        else [
            c
            for c in roadway_net.nodes_df.columns
            if c in parameters.output_variables
        ]
    )

    # unless specified that all the data goes to the DBF, only output A and B
    dbf_link_output_variables = (
        link_output_variables if data_to_dbf else ["A", "B", "shape_id", "geometry"]
    )

    output_link_shp = (
        output_link_shp if output_link_shp else parameters.output_link_shp
    )

    output_node_shp = (
        output_node_shp if output_node_shp else parameters.output_node_shp
    )

    output_link_csv = (
        output_link_csv if output_link_csv else parameters.output_link_csv
    )

    output_node_csv = (
        output_node_csv if output_node_csv else parameters.output_node_csv
    )

    """
    Start Process
    """

    WranglerLogger.info("Renaming DBF Node Variables")
    nodes_dbf_df = rename_variables_for_dbf(
        input_df = roadway_net.nodes_df, 
        parameters = parameters,
        output_variables=node_output_variables
    )
    WranglerLogger.info("Renaming DBF Link Variables")
    links_dbf_df = rename_variables_for_dbf(
        input_df = roadway_net.links_df,
        parameters = parameters,
        output_variables=dbf_link_output_variables
    )

    links_dbf_df = gpd.GeoDataFrame(links_dbf_df, geometry=links_dbf_df["geometry"])

    if export_drive_only == True:
        nodes_dbf_df = nodes_dbf_df[nodes_dbf_df.drive_node == 1].copy()
        links_dbf_df = links_dbf_df[links_dbf_df.drive == 1].copy()

    WranglerLogger.info("Writing Node Shapes:\n - {}".format(output_node_shp))
    nodes_dbf_df.to_file(output_node_shp)
    WranglerLogger.info("Writing Link Shapes:\n - {}".format(output_link_shp))
    links_dbf_df.to_file(output_link_shp)

    if data_to_csv:
        WranglerLogger.info(
            "Writing Network Data to CSVs:\n - {}\n - {}".format(
                output_link_csv, output_node_csv
            )
        )
        roadway_net.links_df[link_output_variables].to_csv(output_link_csv, index=False)
        roadway_net.nodes_df[node_output_variables].to_csv(output_node_csv, index=False)

def write_roadway_as_fixedwidth(
    roadway_net=None,
    parameters=None,
    zones: int=None,
    node_output_variables: list = None,
    link_output_variables: list = None,
    output_link_txt: str = None,
    output_node_txt: str = None,
    output_link_header_width_txt: str = None,
    output_node_header_width_txt: str = None,
    output_cube_network_script: str = None,
    drive_only: bool = False,
):
    """
    Writes out fixed width file.

    This function does:
    1. write out link and node fixed width data files for cube.
    2. write out header and width correspondence.
    3. write out cube network building script with header and width specification.

    Args:
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
        "Network Link Variables: \n - {}".format(
            "\n - ".join(roadway_net.links_df.columns)
        )
    )
    WranglerLogger.debug(
        "Network Node Variables: \n - {}".format(
            "\n - ".join(roadway_net.nodes_df.columns)
        )
    )

    zones = zones if zones else parameters.zones

    link_output_variables = (
        link_output_variables
        if link_output_variables
        else [
            c
            for c in roadway_net.links_df.columns
            if c in parameters.output_variables
        ]
    )

    node_output_variables = (
        node_output_variables
        if node_output_variables
        else [
            c
            for c in roadway_net.nodes_df.columns
            if c in parameters.output_variables
        ]
    )

    output_link_txt = (
        output_link_txt if output_link_txt else parameters.output_link_txt
    )

    output_node_txt = (
        output_node_txt if output_node_txt else parameters.output_node_txt
    )

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
    link_ff_df, link_max_width_dict = dataframe_to_fixed_width(
        roadway_net.links_df[link_output_variables]
    )

    if drive_only:
        link_ff_df = link_ff_df.loc[link_ff_df["drive_access"] == 1]

    WranglerLogger.info("Writing out link database")

    link_ff_df.to_csv(output_link_txt, sep=";", index=False, header=False)

    # write out header and width correspondence
    WranglerLogger.info("Writing out link header and width ----")
    link_max_width_df = DataFrame(
        list(link_max_width_dict.items()), columns=["header", "width"]
    )
    link_max_width_df.to_csv(output_link_header_width_txt, index=False)

    node_ff_df, node_max_width_dict = dataframe_to_fixed_width(
        roadway_net.nodes_df[node_output_variables]
    )
    WranglerLogger.info("Writing out node database")

    if drive_only:
        node_ff_df = node_ff_df.loc[node_ff_df["drive_node"] == 1]

    node_ff_df.to_csv(output_node_txt, sep=";", index=False, header=False)

    # write out header and width correspondence
    WranglerLogger.info("Writing out node header and width")
    node_max_width_df = DataFrame(
        list(node_max_width_dict.items()), columns=["header", "width"]
    )
    node_max_width_df.to_csv(output_node_header_width_txt, index=False)

    # write out cube script
    s = 'RUN PGM = NETWORK MSG = "Read in network from fixed width file" \n'
    s += 'FILEI LINKI[1] = "{}",'.format(output_link_txt)
    start_pos = 1
    for i in range(len(link_max_width_df)):
        s += " VAR=" + link_max_width_df.header.iloc[i]

        if roadway_net.links_df.dtypes.loc[link_max_width_df.header.iloc[i]] == "O":
            s += "(C" + str(link_max_width_df.width.iloc[i]) + ")"

        s += (
            ", BEG="
            + str(start_pos)
            + ", LEN="
            + str(link_max_width_df.width.iloc[i])
            + ","
        )

        start_pos += link_max_width_df.width.iloc[i] + 1

    s = s[:-1]
    s += "\n"
    s += 'FILEI NODEI[1] = "{}",'.format(output_node_txt)
    start_pos = 1
    for i in range(len(node_max_width_df)):
        s += " VAR=" + node_max_width_df.header.iloc[i]

        if roadway_net.nodes_df.dtypes.loc[node_max_width_df.header.iloc[i]] == "O":
            s += "(C" + str(node_max_width_df.width.iloc[i]) + ")"

        s += (
            ", BEG="
            + str(start_pos)
            + ", LEN="
            + str(node_max_width_df.width.iloc[i])
            + ","
        )

        start_pos += node_max_width_df.width.iloc[i] + 1

    s = s[:-1]
    s += "\n"
    s += 'FILEO NETO = "complete_network.net" \n\n'
    s += "ZONES = {} \n\n".format(zones)
    s += "ROADWAY = LTRIM(TRIM(ROADWAY)) \n"
    s += "NAME = LTRIM(TRIM(NAME)) \n"
    s += "\n \nENDRUN"

    with open(output_cube_network_script, "w") as f:
        f.write(s)

# this should be moved to util
# @staticmethod
def dataframe_to_fixed_width(df):
    """
    Convert dataframe to fixed width format, geometry column will not be transformed.

    Args:
        df (pandas DataFrame).

    Returns:
        pandas dataframe:  dataframe with fixed width for each column.
        dict: dictionary with columns names as keys, column width as values.
    """
    WranglerLogger.info("Starting fixed width conversion")

    # get the max length for each variable column
    max_width_dict = dict(
        [
            (v, df[v].apply(lambda r: len(str(r)) if r != None else 0).max())
            for v in df.columns.values
            if v != "geometry"
        ]
    )

    fw_df = df.drop("geometry", axis=1).copy()
    for c in fw_df.columns:
        fw_df[c] = fw_df[c].apply(lambda x: str(x))
        fw_df["pad"] = fw_df[c].apply(lambda x: " " * (max_width_dict[c] - len(x)))
        fw_df[c] = fw_df.apply(lambda x: x["pad"] + x[c], axis=1)

    return fw_df, max_width_dict

# @staticmethod
def read_match_result(path):
    """
    Reads the shst geojson match returns.

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
    """
    Gets attribute from source data using SHST match result.

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
        "source ShSt rename_variables_for_dbf columns\n{}".format(
            source_shst_ref_df.columns
        )
    )
    WranglerLogger.debug("source gdf columns\n{}".format(source_gdf.columns))
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
        source_gdf[["LINK_ID", field_name]].rename(
            columns={"LINK_ID": "source_link_id"}
        ),
        how="left",
        on="source_link_id",
    )

    # drop duplicated records with same field value

    join_refId_df.drop_duplicates(
        subset=["model_link_id", "shstReferenceId", field_name], inplace=True
    )

    # more than one match, take the best score

    join_refId_df.sort_values(
        by=["model_link_id", "source_score"],
        ascending=True,
        na_position="first",
        inplace=True,
    )

    join_refId_df.drop_duplicates(
        subset=["model_link_id"], keep="last", inplace=True
    )

    # self.links_df[field_name] = join_refId_df[field_name]

    return join_refId_df[links_df.columns.tolist() + [field_name, "source_link_id"]]
