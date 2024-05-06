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
