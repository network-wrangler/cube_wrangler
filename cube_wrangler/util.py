import re
from .logger import WranglerLogger
from unidecode import unidecode

import numpy as np


def get_shared_streets_intersection_hash(lat, long, osm_node_id=None):
    """
    Calculated per:
       https://github.com/sharedstreets/sharedstreets-js/blob/0e6d7de0aee2e9ae3b007d1e45284b06cc241d02/src/index.ts#L553-L565
    Expected in/out
      -93.0965985, 44.952112199999995 osm_node_id = 954734870
       69f13f881649cb21ee3b359730790bb9

    """
    import hashlib

    message = "Intersection {0:.5f} {0:.5f}".format(long, lat)
    if osm_node_id:
        message += " {}".format(osm_node_id)
    unhashed = message.encode("utf-8")
    hash = hashlib.md5(unhashed).hexdigest()
    return hash


def hhmmss_to_datetime(hhmmss_str: str):
    """
    Creates a datetime time object from a string of hh:mm:ss

    Args:
        hhmmss_str: string of hh:mm:ss
    Returns:
        dt: datetime.time object representing time
    """
    import datetime

    dt = datetime.time(*[int(i) for i in hhmmss_str.split(":")])

    return dt


def secs_to_datetime(secs: int):
    """
    Creates a datetime time object from a seconds from midnight

    Args:
        secs: seconds from midnight
    Returns:
        dt: datetime.time object representing time
    """
    import datetime

    dt = (datetime.datetime.min + datetime.timedelta(seconds=secs)).time()

    return dt


def column_name_to_parts(c, parameters=None):

    if not parameters:
        from .parameters import Parameters

        parameters = Parameters()

    if c[0:2] == "ML":
        managed = 1
    else:
        managed = 0

    time_period = None
    category = None

    if c.split("_")[0] not in parameters.properties_to_split.keys():
        if c.split("_")[-1] in parameters.time_period_to_time.keys():
            time_period = c.split("_")[-1]
            base_name = c.split(time_period)[-2][:-1]
            if base_name not in parameters.properties_to_split.keys():
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
        msg = "Can't split property correctly: {}".format(c)
        WranglerLogger.error(msg)

    return base_name, time_period, category, managed


def shorten_name(name):
    if type(name) == str:
        name_list = name.split(",")
    elif type(name) in [float, np.int32, np.int64]:
        name_list = str(name)
    else:
        name_list = name
    name_list = [
        re.sub(r"\W+", " ", str(c)).replace("nan", "").strip(" ") for c in name_list
    ]

    name_list = list(set(name_list))
    # name_list.remove('')

    name_new = " ".join(name_list).strip(" ")

    # convert non english character to english
    name_new = unidecode(name_new)

    return name_new
