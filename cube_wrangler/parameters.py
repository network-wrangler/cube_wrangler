import os
from .logger import WranglerLogger


def get_base_dir(cube_wrangler_base_dir=os.getcwd()):
    d = cube_wrangler_base_dir
    for i in range(3):
        if "conda-environments" in os.listdir(d):
            WranglerLogger.info("cube_wrangler base directory set as: {}".format(d))
            return d
        d = os.path.dirname(d)

    msg = "Cannot find cube_wrangler base directory from {}, please input using keyword in parameters: `cube_wrangler_base_dir =` ".format(
        cube_wrangler_base_dir
    )
    WranglerLogger.error(msg)
    raise (ValueError(msg))


class Parameters:
    """A class representing all the parameters defining the networks
    including time of day, categories, etc.

    Parameters can be set at runtime by initializing a parameters instance
    with a keyword argument setting the attribute.  Parameters that are
    not explicitly set will use default parameters listed in this class.
    .. highlight:: python
    ##TODO potentially split this between several classes.

    Attr:
    """

    def __init__(self, **kwargs):
        """
        constructor for the Parameters class
        """
        if "cube_wrangler_base_dir" in kwargs:
            self.base_dir = get_base_dir(
                cube_wrangler_base_dir=kwargs.get("cube_wrangler_base_dir")
            )
        else:
            self.base_dir = get_base_dir()

        if "settings_location" in kwargs:
            self.settings_location = kwargs.get("settings_location")
        else:
            self.settings_location = os.path.join(self.base_dir, "examples", "settings")

        if "scratch_location" in kwargs:
            self.scratch_location = kwargs.get("scratch_location")
        else:
            self.scratch_location = os.path.join(self.base_dir, "tests", "scratch")

        if "time_periods_to_time" in kwargs:
            self.time_periods_to_time = kwargs.get("time_periods_to_time")
        else:
            self.time_period_to_time = {
                "EA": ("3:00", "6:00"),
                "AM": ("6:00", "10:00"),  ##TODO FILL IN with real numbers
                "MD": ("10:00", "15:00"),
                "PM": ("15:00", "19:00"),
                "NT": ("19:00", "3:00"),
            }

        if "categories" in kwargs:
            self.categories = kwargs.get("categories")
        else:
            self.categories = {
                # suffix, source (in order of search)
                "sov": ["sov", "default"],
                "hov2": ["hov2", "default", "sov"],
                "hov3": ["hov3", "hov2", "default", "sov"],
                "truck": ["trk", "sov", "default"],
            }

        # prefix, source variable, categories
        self.properties_to_split = {
            "trn_priority": {
                "v": "trn_priority",
                "time_periods": self.time_period_to_time,
            },
            "ttime_assert": {
                "v": "ttime_assert",
                "time_periods": self.time_period_to_time,
            },
            "lanes": {"v": "lanes", "time_periods": self.time_period_to_time},
            "ML_lanes": {"v": "ML_lanes", "time_periods": self.time_period_to_time},
            "price": {
                "v": "price",
                "time_periods": self.time_period_to_time,
                "categories": self.categories,
            },
            "access": {"v": "access", "time_periods": self.time_period_to_time},
        }

        self.net_to_dbf_crosswalk = os.path.join(
            self.settings_location, "net_to_dbf.csv"
        )

        self.log_to_net_crosswalk = os.path.join(
            self.settings_location, "log_to_net.csv"
        )

        self.output_variables = [
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

        self.output_link_shp = os.path.join(self.scratch_location, "links.shp")
        self.output_node_shp = os.path.join(self.scratch_location, "nodes.shp")
        self.output_link_csv = os.path.join(self.scratch_location, "links.csv")
        self.output_node_csv = os.path.join(self.scratch_location, "nodes.csv")
        self.output_link_txt = os.path.join(self.scratch_location, "links.txt")
        self.output_node_txt = os.path.join(self.scratch_location, "nodes.txt")
        self.output_link_header_width_txt = os.path.join(
            self.scratch_location, "links_header_width.txt"
        )
        self.output_node_header_width_txt = os.path.join(
            self.scratch_location, "nodes_header_width.txt"
        )
        self.output_cube_network_script = os.path.join(
            self.scratch_location, "make_complete_network_from_fixed_width_file.s"
        )

        self.bool_col = [
            "rail_only",
            "bus_only",
            "drive_access",
            "bike_access",
            "walk_access",
            "truck_access",
        ]

        self.int_col = [
            "model_link_id",
            "model_node_id",
            "A",
            "B",
            "lanes_AM",
            "lanes_MD",
            "lanes_PM",
            "lanes_NT",
            "roadway_class",
            "assign_group",
            "county",
            "area_type",
            "trn_priority",
            "AADT",
            "count_AM",
            "count_MD",
            "count_PM",
            "count_NT",
            "count_daily",
            "centroidconnect",
            "bike_facility",
            "truck_access",
            "drive_node",
            "walk_node",
            "bike_node",
            "transit_node",
            "ML_lanes_AM",
            "ML_lanes_MD",
            "ML_lanes_PM",
            "ML_lanes_NT",
            "segment_id",
            "managed",
            "bike",
            "walk",
        ]

        self.float_col = ["distance", "ttime_assert", "price", "X", "Y"]

        self.zones = 3061

        self.__dict__.update(kwargs)
