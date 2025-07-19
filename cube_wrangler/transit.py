"""Transit-related classes to parse, compare, and write standard and cube transit files.

Typical usage example:

  tn = CubeTransit.create_from_cube(CUBE_DIR)
  transit_change_list = tn.evaluate_differences(base_transit_network)

  cube_transit_net = StandardTransit.read_gtfs(BASE_TRANSIT_DIR)
  cube_transit_net.write_as_cube_lin(os.path.join(WRITE_DIR, "outfile.lin"))
"""

import os
import copy
import csv
import datetime, time
from typing import Any, Dict, Optional, Union

from lark import Lark, Transformer, v_args
from pandas import DataFrame

import pandas as pd
import partridge as ptg

from network_wrangler.transit.network import TransitNetwork

from .logger import WranglerLogger
from .parameters import Parameters


class StandardTransit(object):
    """Holds a standard transit feed as a Partridge object and contains
    methods to manipulate and translate the GTFS data to MetCouncil's
    Cube Line files.

    .. highlight:: python
    Typical usage example:
    ::
        cube_transit_net = StandardTransit.read_gtfs(BASE_TRANSIT_DIR)
        cube_transit_net.write_as_cube_lin(os.path.join(WRITE_DIR, "outfile.lin"))

    Attributes:
        feed: Partridge Feed object containing read-only access to GTFS feed
        parameters (Parameters): Parameters instance containing information
            about time periods and variables.
    """

    def __init__(
        self, ptg_feed, road_net=None, parameters: Union[Parameters, dict] = {}
    ):
        """

        Args:
            ptg_feed: partridge feed object
            parameters: dictionary of parameter settings (see Parameters class) or an instance of Parameters
        """
        self.feed = ptg_feed
        self.road_net = road_net

        if type(parameters) is dict:
            self.parameters = Parameters(**parameters)
        elif isinstance(parameters, Parameters):
            # self.parameters = Parameters(**parameters.__dict__)
            self.parameters = parameters
        else:
            msg = "Parameters should be a dict or instance of Parameters: found {} which is of type:{}".format(
                parameters, type(parameters)
            )
            WranglerLogger.error(msg)
            raise ValueError(msg)

    @staticmethod
    def fromTransitNetwork(
        transit_network_object: TransitNetwork, parameters: Union[Parameters, dict] = {}
    ):
        """
        RoadwayNetwork to ModelRoadwayNetwork

        Args:
            transit_network_object: Reference to an instance of TransitNetwork.
            parameters: dictionary of parameter settings (see Parameters class) or an instance of Parameters. If not provided will
                use default parameters.

        Returns:
            StandardTransit
        """
        return StandardTransit(
            transit_network_object.feed,
            transit_network_object.road_net,
            parameters=parameters,
        )

    @staticmethod
    def read_gtfs(gtfs_feed_dir: str, parameters: Union[Parameters, dict] = {}):
        """
        Reads GTFS files from a directory and returns a StandardTransit
        instance.

        Args:
            gtfs_feed_dir: location of the GTFS files
            parameters: dictionary of parameter settings (see Parameters class) or an instance of Parameters. If not provided will
                use default parameters.

        Returns:
            StandardTransit instance
        """
        return StandardTransit(ptg.load_feed(gtfs_feed_dir), parameters=parameters)

    def write_as_cube_lin(self, outpath: str = None, line_name_xwalk: str = None):
        """
        Writes the gtfs feed as a cube line file after
        converting gtfs properties to MetCouncil cube properties.

        Args:
            outpath: File location for output cube line file.

        """
        if not outpath:
            outpath = os.path.join(self.parameters.scratch_location, "outtransit.lin")
        # trip_cube_df = self.route_properties_gtfs_to_cube(self, line_name_xwalk)

        # trip_cube_df["LIN"] = trip_cube_df.apply(self.cube_format, axis=1)

        l = self.feed.trip_cube_df["LIN"].tolist()
        l = [";;<<PT>><<LINE>>;;"] + l

        with open(outpath, "w") as f:
            f.write("\n".join(l))

    def time_to_cube_time_period(
        self, start_time_secs: int, as_str: bool = True, verbose: bool = False
    ):
        """
        Converts seconds from midnight to the cube time period.

        Args:
            start_time_secs: start time for transit trip in seconds
                from midnight
            as_str: if True, returns the time period as a string,
                otherwise returns a numeric time period

        Returns:
            this_tp_num: if as_str is False, returns the numeric
                time period
            this_tp: if as_str is True, returns the Cube time period
                name abbreviation
        """
        from .util import hhmmss_to_datetime, secs_to_datetime

        # set initial time as the time that spans midnight

        start_time_dt = secs_to_datetime(start_time_secs)

        # set initial time as the time that spans midnight
        this_tp = "NA"
        for tp_name, _times in self.parameters.time_period_to_time.items():
            _start_time, _end_time = _times
            _dt_start_time = hhmmss_to_datetime(_start_time)
            _dt_end_time = hhmmss_to_datetime(_end_time)
            if _dt_start_time > _dt_end_time:
                this_tp = tp_name
                break

        for tp_name, _times in self.parameters.time_period_to_time.items():
            _start_time, _end_time = _times
            _dt_start_time = hhmmss_to_datetime(_start_time)
            if start_time_dt >= _dt_start_time:
                this_time = _dt_start_time
                this_tp = tp_name

        if verbose:
            WranglerLogger.debug(
                "Finding Cube Time Period from Start Time: \
                \n  - start_time_sec: {} \
                \n  - start_time_dt: {} \
                \n  - this_tp: {}".format(
                    start_time_secs, start_time_dt, this_tp
                )
            )

        if as_str:
            return this_tp

        name_to_num = {v: k for k, v in self.parameters.cube_time_periods.items}
        this_tp_num = name_to_num.get(this_tp)

        if not this_tp_num:
            msg = (
                "Cannot find time period number in {} for time period name: {}".format(
                    name_to_num, this_tp
                )
            )
            WranglerLogger.error(msg)
            raise ValueError(msg)

        return this_tp_num


class CubeTransformer(Transformer):
    """A lark-parsing Transformer which transforms the parse-tree to
    a dictionary.

    .. highlight:: python
    Typical usage example:
    ::
        transformed_tree_data = CubeTransformer().transform(parse_tree)

    Attributes:
        line_order (int): a dynamic counter to hold the order of the nodes within
            a route shape
        lines_list (list): a list of the line names
    """

    def __init__(self):
        self.line_order = 0
        self.lines_list = []

    def lines(self, line):
        # WranglerLogger.debug("lines: \n {}".format(line))

        # This MUST be a tuple because it returns to start in the tree
        lines = {k: v for k, v in line}
        return ("lines", lines)

    @v_args(inline=True)
    def program_type_line(self, PROGRAM_TYPE, whitespace=None):
        # WranglerLogger.debug("program_type_line:{}".format(PROGRAM_TYPE))
        self.program_type = PROGRAM_TYPE.value

        # This MUST be a tuple because it returns to start  in the tree
        return ("program_type", PROGRAM_TYPE.value)

    @v_args(inline=True)
    def line(self, lin_attributes, nodes):
        # WranglerLogger.debug("line...attributes:\n  {}".format(lin_attributes))
        # WranglerLogger.debug("line...nodes:\n  {}".format(nodes))
        lin_name = lin_attributes["NAME"]

        self.line_order = 0
        # WranglerLogger.debug("parsing: {}".format(lin_name))

        return (lin_name, {"line_properties": lin_attributes, "line_shape": nodes})

    @v_args(inline=True)
    def lin_attributes(self, *lin_attr):
        lin_attr = {k: v for (k, v) in lin_attr}
        # WranglerLogger.debug("lin_attributes:  {}".format(lin_attr))
        return lin_attr

    @v_args(inline=True)
    def lin_attr(self, lin_attr_name, attr_value, SEMICOLON_COMMENT=None):
        # WranglerLogger.debug("lin_attr {}:  {}".format(lin_attr_name, attr_value))
        return lin_attr_name, attr_value

    def lin_attr_name(self, args):
        attr_name = args[0].value.upper()
        # WranglerLogger.debug(".......args {}".format(args))
        if attr_name in ["USERA", "FREQ", "HEADWAY"]:
            attr_name = attr_name + "[" + str(args[2]) + "]"
        return attr_name

    def attr_value(self, attr_value):
        try:
            return int(attr_value[0].value)
        except:
            return attr_value[0].value

    def nodes(self, lin_node):
        lin_node = DataFrame(lin_node)
        # WranglerLogger.debug("nodes:\n {}".format(lin_node))

        return lin_node

    @v_args(inline=True)
    def lin_node(self, NODE_NUM, SEMICOLON_COMMENT=None, *lin_nodeattr):
        self.line_order += 1
        n = int(NODE_NUM.value)
        return {"node_id": abs(n), "node": n, "stop": n > 0, "order": self.line_order}

    start = dict


TRANSIT_LINE_FILE_GRAMMAR = r"""

start             : program_type_line? lines
WHITESPACE        : /[ \t\r\n]/+
STRING            : /("(?!"").*?(?<!\\)(\\\\)*?"|'(?!'').*?(?<!\\)(\\\\)*?')/i
SEMICOLON_COMMENT : /;[^\n]*/
BOOLEAN           : "T"i | "F"i
program_type_line : ";;<<" PROGRAM_TYPE ">><<LINE>>;;" WHITESPACE?
PROGRAM_TYPE      : "PT" | "TRNBUILD"

lines             : line*
line              : "LINE" lin_attributes nodes

lin_attributes    : lin_attr+
lin_attr          : lin_attr_name "=" attr_value "," SEMICOLON_COMMENT*
TIME_PERIOD       : "1".."5"
!lin_attr_name     : "allstops"i
                    | "color"i
                    | ("freq"i "[" TIME_PERIOD "]")
                    | ("headway"i "[" TIME_PERIOD "]")
                    | "mode"i
                    | "name"i
                    | "oneway"i
                    | "owner"i
                    | "runtime"i
                    | "timefac"i
                    | "xyspeed"i
                    | "longname"i
                    | "shortname"i
                    | ("usera"i TIME_PERIOD)
                    | ("usern2"i)
                    | "circular"i
                    | "vehicletype"i
                    | "operator"i
                    | "faresystem"i

attr_value        : BOOLEAN | STRING | SIGNED_INT

nodes             : lin_node+
lin_node          : ("N" | "NODES")? "="? NODE_NUM ","? SEMICOLON_COMMENT? lin_nodeattr*
NODE_NUM          : SIGNED_INT
lin_nodeattr      : lin_nodeattr_name "=" attr_value ","? SEMICOLON_COMMENT*
!lin_nodeattr_name : "access_c"i
                    | "access"i
                    | "delay"i
                    | "xyspeed"i
                    | "timefac"i
                    | "nntime"i
                    | "time"i

operator          : SEMICOLON_COMMENT* "OPERATOR" opmode_attr* SEMICOLON_COMMENT*
mode              : SEMICOLON_COMMENT* "MODE" opmode_attr* SEMICOLON_COMMENT*
opmode_attr       : ( (opmode_attr_name "=" attr_value) ","?  )
opmode_attr_name  : "number" | "name" | "longname"

%import common.SIGNED_INT
%import common.WS
%ignore WS

"""
