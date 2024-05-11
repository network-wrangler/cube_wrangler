import copy
from typing import Optional, Union

import geopandas as gpd
import pandas as pd

from geopandas import GeoDataFrame
from pandas import DataFrame
import numpy as np

from network_wrangler import RoadwayNetwork
from .parameters import Parameters
from .logger import WranglerLogger


class ModelRoadwayNetwork(RoadwayNetwork):
    """
    Subclass of network_wrangler class :ref:`RoadwayNetwork <network_wrangler:RoadwayNetwork>`

    A representation of the physical roadway network and its properties.
    """

    def __init__(
        self,
        nodes: GeoDataFrame,
        links: DataFrame,
        shapes: GeoDataFrame,
        parameters: Union[Parameters, dict] = {},
    ):
        """
        Constructor

        Args:
            nodes: geodataframe of nodes
            links: dataframe of links
            shapes: geodataframe of shapes
            parameters: dictionary of parameter settings (see Parameters class) or an instance of Parameters. If not specified, will use default parameters.

        """
        super().__init__(nodes, links, shapes)

        # will have to change if want to alter them
        if type(parameters) is dict:
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            self.parameters = Parameters(**parameters.__dict__)
        else:
            msg = "Parameters should be a dict or instance of Parameters: found {} which is of type:{}".format(
                parameters, type(parameters)
            )
            WranglerLogger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def read(
        link_file: str,
        node_file: str,
        shape_file: str,
        fast: bool = False,
        recalculate_calculated_variables=False,
        parameters: Union[dict, Parameters] = {},
    ):
        """
        Reads in links and nodes network standard.

        Args:
            link_file (str): File path to link json.
            node_file (str): File path to node geojson.
            shape_file (str): File path to link true shape geojson
            fast (bool): boolean that will skip validation to speed up read time.
            recalculate_calculated_variables (bool): calculates fields from spatial joins, etc.
            recalculate_distance (bool):  re-calculates distance.
            parameters: dictionary of parameter settings (see Parameters class) or an instance of Parameters. If not specified, will use default parameters.

        Returns:
            ModelRoadwayNetwork
        """
        # road_net =  super().read(link_file, node_file, shape_file, fast=fast)
        road_net = RoadwayNetwork.read(link_file, node_file, shape_file, fast=fast)

        m_road_net = ModelRoadwayNetwork(
            road_net.nodes_df,
            road_net.links_df,
            road_net.shapes_df,
            parameters=parameters,
        )

        if recalculate_calculated_variables:
            m_road_net.create_calculated_variables()

        return m_road_net

    @staticmethod
    def from_RoadwayNetwork(
        roadway_network_object, parameters: Union[dict, Parameters] = {}
    ):
        """
        RoadwayNetwork to ModelRoadwayNetwork

        Args:
            roadway_network_object (RoadwayNetwork).
            parameters: dictionary of parameter settings (see Parameters class) or an instance of Parameters. If not specified, will use default parameters.

        Returns:
            ModelRoadwayNetwork
        """
        return ModelRoadwayNetwork(
            roadway_network_object.nodes_df,
            roadway_network_object.links_df,
            roadway_network_object.shapes_df,
            parameters=parameters,
        )

    def split_properties_by_time_period_and_category(self, properties_to_split=None):
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
            properties_to_split = self.parameters.properties_to_split

        for out_var, params in properties_to_split.items():
            if params["v"] not in self.links_df.columns:
                WranglerLogger.warning(
                    "Specified variable to split: {} not in network variables: {}. Returning 0.".format(
                        params["v"], str(self.links_df.columns)
                    )
                )
                if params.get("time_periods") and params.get("categories"):

                    for time_suffix, category_suffix in itertools.product(
                        params["time_periods"], params["categories"]
                    ):
                        self.links_df[
                            out_var + "_" + time_suffix + "_" + category_suffix
                        ] = 0
                elif params.get("time_periods"):
                    for time_suffix in params["time_periods"]:
                        self.links_df[out_var + "_" + time_suffix] = 0
            elif params.get("time_periods") and params.get("categories"):
                for time_suffix, category_suffix in itertools.product(
                    params["time_periods"], params["categories"]
                ):
                    self.links_df[
                        out_var + "_" + category_suffix + "_" + time_suffix
                    ] = self.get_property_by_time_period_and_group(
                        params["v"],
                        category=params["categories"][category_suffix],
                        time_period=params["time_periods"][time_suffix],
                    )
            elif params.get("time_periods"):
                for time_suffix in params["time_periods"]:
                    self.links_df[out_var + "_" + time_suffix] = (
                        self.get_property_by_time_period_and_group(
                            params["v"],
                            category=None,
                            time_period=params["time_periods"][time_suffix],
                        )
                    )
            else:
                raise ValueError(
                    "Shoudn't have a category without a time period: {}".format(params)
                )

    def create_calculated_variables(self):
        """
        Creates calculated roadway variables.

        Args:
            None
        """
        WranglerLogger.info("Creating calculated roadway variables.")
        self.calculate_distance_miles(overwrite=True)

    def calculate_distance_miles(self, network_variable="distance", overwrite=False):
        """
        calculate link distance in miles

        Args:
            overwrite (Bool): True if overwriting existing variable in network.  Default to False.

        Returns:
            None

        """

        if network_variable in self.links_df:
            if overwrite or (self.links_df[network_variable].isnull().any()):
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
                return

        """
        Start actual process
        """

        temp_links_gdf = self.links_df.copy()
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

        self.links_df[network_variable] = temp_links_gdf[network_variable]

    def create_ML_variable(
        self,
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
        if network_variable in self.links_df:
            if overwrite:
                WranglerLogger.info(
                    "Overwriting existing ML Variable '{}' already in network".format(
                        network_variable
                    )
                )
                self.links_df[network_variable] = int(0)
            else:
                WranglerLogger.info(
                    "ML Variable '{}' already in network. Returning without overwriting.".format(
                        network_variable
                    )
                )
                return

        """
        Verify inputs
        """

        WranglerLogger.info(
            "Finished creating ML lanes variable: {}".format(network_variable)
        )

    def create_hov_corridor_variable(
        self,
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
        if network_variable in self.links_df:
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
                return

        """
        Verify inputs
        """

        self.links_df[network_variable] = int(0)

        WranglerLogger.info(
            "Finished creating hov corridor variable: {}".format(network_variable)
        )

    def create_managed_variable(
        self,
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
        if network_variable in self.links_df:
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
                return

        """
        Verify inputs
        """

        self.links_df[network_variable] = int(0)

        WranglerLogger.info(
            "Finished creating managed variable: {}".format(network_variable)
        )

    def convert_int(self, int_col_names=[]):
        """
        Convert integer columns
        """

        WranglerLogger.info("Converting variable type to MetCouncil standard")

        if not int_col_names:
            int_col_names = self.parameters.int_col

        ##Why are we doing this?
        # int_col_names.remove("lanes")

        for c in list(set(self.links_df.columns) & set(int_col_names)):
            self.links_df[c] = self.links_df[c].replace(np.nan, 0)
            # REPLACE BLANKS WITH ZERO FOR INTEGER COLUMNS
            self.links_df[c] = self.links_df[c].replace("", 0)
            try:
                self.links_df[c] = self.links_df[c].astype(int)
            except ValueError:
                try:
                    self.links_df[c] = self.links_df[c].astype(float)
                    self.links_df[c] = self.links_df[c].astype(int)
                except:
                    msg = f"Could not convert column {c} to integer."
                    WranglerLogger.error(msg)
                    raise ValueError(msg)

        for c in list(set(self.nodes_df.columns) & set(int_col_names)):
            self.nodes_df[c] = self.nodes_df[c].replace(np.nan, 0)
            # REPLACE BLANKS WITH ZERO FOR INTEGER COLUMNS
            self.nodes_df[c] = self.nodes_df[c].replace("", 0)
            try:
                self.nodes_df[c] = self.nodes_df[c].astype(int)
            except ValueError:
                msg = f"Could not convert column {c} to integer."
                WranglerLogger.error(msg)
                raise ValueError(msg)

    def fill_na(self):
        """
        Fill na values from create_managed_lane_network()
        """

        WranglerLogger.info("Filling nan for network from network wrangler")

        num_col = self.parameters.int_col + self.parameters.float_col

        for x in list(self.links_df.columns):
            if x in num_col:
                self.links_df[x].fillna(0, inplace=True)
                self.links_df[x] = self.links_df[x].apply(
                    lambda k: 0 if k in [np.nan, "", float("nan"), "NaN"] else k
                )

            else:
                self.links_df[x].fillna("", inplace=True)

        for x in list(self.nodes_df.columns):
            if x in num_col:
                self.nodes_df[x].fillna(0, inplace=True)
            else:
                self.nodes_df[x].fillna("", inplace=True)

    def rename_variables_for_dbf(
        self,
        input_df,
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
            else self.parameters.net_to_dbf_crosswalk
        )

        output_variables = (
            output_variables if output_variables else self.parameters.output_variables
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
        self,
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
                "\n - ".join(self.parameters.output_variables)
            )
        )

        """
        Verify inputs
        """

        WranglerLogger.debug(
            "Network Link Variables: \n - {}".format(
                "\n - ".join(self.links_df.columns)
            )
        )
        WranglerLogger.debug(
            "Network Node Variables: \n - {}".format(
                "\n - ".join(self.nodes_df.columns)
            )
        )

        link_output_variables = (
            link_output_variables
            if link_output_variables
            else [
                c
                for c in self.links_df.columns
                if c in self.parameters.output_variables
            ]
        )

        node_output_variables = (
            node_output_variables
            if node_output_variables
            else [
                c
                for c in self.nodes_df.columns
                if c in self.parameters.output_variables
            ]
        )

        # unless specified that all the data goes to the DBF, only output A and B
        dbf_link_output_variables = (
            link_output_variables if data_to_dbf else ["A", "B", "shape_id", "geometry"]
        )

        output_link_shp = (
            output_link_shp if output_link_shp else self.parameters.output_link_shp
        )

        output_node_shp = (
            output_node_shp if output_node_shp else self.parameters.output_node_shp
        )

        output_link_csv = (
            output_link_csv if output_link_csv else self.parameters.output_link_csv
        )

        output_node_csv = (
            output_node_csv if output_node_csv else self.parameters.output_node_csv
        )

        """
        Start Process
        """

        WranglerLogger.info("Renaming DBF Node Variables")
        nodes_dbf_df = self.rename_variables_for_dbf(
            self.nodes_df, output_variables=node_output_variables
        )
        WranglerLogger.info("Renaming DBF Link Variables")
        links_dbf_df = self.rename_variables_for_dbf(
            self.links_df, output_variables=dbf_link_output_variables
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
            self.links_df[link_output_variables].to_csv(output_link_csv, index=False)
            self.nodes_df[node_output_variables].to_csv(output_node_csv, index=False)

    def write_roadway_as_fixedwidth(
        self,
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
                "\n - ".join(self.links_df.columns)
            )
        )
        WranglerLogger.debug(
            "Network Node Variables: \n - {}".format(
                "\n - ".join(self.nodes_df.columns)
            )
        )

        link_output_variables = (
            link_output_variables
            if link_output_variables
            else [
                c
                for c in self.links_df.columns
                if c in self.parameters.output_variables
            ]
        )

        node_output_variables = (
            node_output_variables
            if node_output_variables
            else [
                c
                for c in self.nodes_df.columns
                if c in self.parameters.output_variables
            ]
        )

        output_link_txt = (
            output_link_txt if output_link_txt else self.parameters.output_link_txt
        )

        output_node_txt = (
            output_node_txt if output_node_txt else self.parameters.output_node_txt
        )

        output_link_header_width_txt = (
            output_link_header_width_txt
            if output_link_header_width_txt
            else self.parameters.output_link_header_width_txt
        )

        output_node_header_width_txt = (
            output_node_header_width_txt
            if output_node_header_width_txt
            else self.parameters.output_node_header_width_txt
        )

        output_cube_network_script = (
            output_cube_network_script
            if output_cube_network_script
            else self.parameters.output_cube_network_script
        )

        """
        Start Process
        """
        link_ff_df, link_max_width_dict = self.dataframe_to_fixed_width(
            self.links_df[link_output_variables]
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

        node_ff_df, node_max_width_dict = self.dataframe_to_fixed_width(
            self.nodes_df[node_output_variables]
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

            if self.links_df.dtypes.loc[link_max_width_df.header.iloc[i]] == "O":
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

            if self.nodes_df.dtypes.loc[node_max_width_df.header.iloc[i]] == "O":
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
        s += "ZONES = {} \n\n".format(self.parameters.zones)
        s += "ROADWAY = LTRIM(TRIM(ROADWAY)) \n"
        s += "NAME = LTRIM(TRIM(NAME)) \n"
        s += "\n \nENDRUN"

        with open(output_cube_network_script, "w") as f:
            f.write(s)
