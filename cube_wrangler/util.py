"""Utility functions for cube_wrangler."""

import re

import numpy as np
from unidecode import unidecode

from .logger import WranglerLogger


def get_shared_streets_intersection_hash(lat, long, osm_node_id=None):
    """Compute a SharedStreets intersection hash for a given lat/long.

    Calculated per:
    https://github.com/sharedstreets/sharedstreets-js/blob/0e6d7de0aee2e9ae3b007d1e45284b06cc241d02/src/index.ts#L553-L565

    Expected in/out:
      -93.0965985, 44.952112199999995 osm_node_id = 954734870
      69f13f881649cb21ee3b359730790bb9.

    """
    import hashlib

    message = f"Intersection {long:.5f} {long:.5f}"
    if osm_node_id:
        message += f" {osm_node_id}"
    unhashed = message.encode("utf-8")
    return hashlib.md5(unhashed).hexdigest()


def hhmmss_to_datetime(hhmmss_str: str):
    """Creates a datetime time object from a string of hh:mm:ss.

    Args:
        hhmmss_str: string of hh:mm:ss
    Returns:
        dt: datetime.time object representing time
    """
    import datetime

    return datetime.time(*[int(i) for i in hhmmss_str.split(":")])


def secs_to_datetime(secs: int):
    """Creates a datetime time object from a seconds from midnight.

    Args:
        secs: seconds from midnight
    Returns:
        dt: datetime.time object representing time
    """
    import datetime

    return (datetime.datetime.min + datetime.timedelta(seconds=secs)).time()


def column_name_to_parts(c, parameters=None):
    """Split a column name into its base name, time period, category, and managed flag.

    Args:
        c: Column name string to split.
        parameters: Parameters instance. Defaults to a new Parameters() instance.

    Returns:
        Tuple of (base_name, time_period, category, managed).
    """
    if not parameters:
        from .parameters import Parameters

        parameters = Parameters()

    managed = 1 if c[0:2] == "ML" else 0

    time_period = None
    category = None

    if c.split("_")[0] not in parameters.properties_to_split:
        if c.split("_")[-1] in parameters.time_period_to_time:
            time_period = c.split("_")[-1]
            base_name = c.split(time_period)[-2][:-1]
            if base_name not in parameters.properties_to_split:
                return c, None, None, managed
        else:
            return c, None, None, managed

    tps = parameters.time_period_to_time.keys()
    cats = parameters.categories.keys()

    if c.split("_")[-1] in tps:
        time_period = c.split("_")[-1]
        base_name = c.split(time_period)[-2][:-1]
        if c.split("_")[-2] in cats:
            category = c.split("_")[-2]
            base_name = c.split(category)[-2][:-1]
    elif c.split("_")[-1] in cats:
        category = c.split("_")[-1]
        base_name = c.split(category)[-2][:-1]
    else:
        msg = f"Can't split property correctly: {c}"
        WranglerLogger.error(msg)

    return base_name, time_period, category, managed


def shorten_name(name):
    """Shorten and normalize a name string by removing special characters and duplicates.

    Args:
        name: Name string, list, float, or numpy integer to shorten.

    Returns:
        A cleaned, deduplicated, ASCII-only name string.
    """
    if type(name) == str:
        name_list = name.split(",")
    elif type(name) in [float, np.int32, np.int64]:
        name_list = str(name)
    else:
        name_list = name
    name_list = [re.sub(r"\W+", " ", str(c)).replace("nan", "").strip(" ") for c in name_list]

    name_list = list(set(name_list))
    # name_list.remove('')

    name_new = " ".join(name_list).strip(" ")

    # convert non english character to english
    return unidecode(name_new)
