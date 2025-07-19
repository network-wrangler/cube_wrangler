import os

import pytest

from cube_wrangler.roadway import Parameters

STPAUL_DIR = os.path.join(os.getcwd(), "data")

STPAUL_SHAPE_FILE = os.path.join(STPAUL_DIR, "st_paul_shape.geojson")
STPAUL_LINK_FILE = os.path.join(STPAUL_DIR, "st_paul_link.json")
STPAUL_NODE_FILE = os.path.join(STPAUL_DIR, "st_paul_node.geojson")


@pytest.mark.roadway
def test_parameter_read(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)

    params = Parameters()
    print(params.__dict__)
    ## todo write an assert that actually tests something


@pytest.mark.roadway
@pytest.mark.skip(reason="This feature is moved.")
def test_network_calculate_variables(request):
    """
    Tests that parameters are read
    """
    print("\n--Starting:", request.node.name)
