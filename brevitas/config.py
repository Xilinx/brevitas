import os
from distutils.util import strtobool


def env_to_bool(name, default):
    return bool(strtobool(os.environ.get(name, "{}".format(default))))

IGNORE_MISSING_KEYS = env_to_bool('BREVITAS_IGNORE_MISSING_KEYS', False)
REINIT_WEIGHT_QUANT_ON_LOAD = env_to_bool('BREVITAS_REINIT_WEIGHT_QUANT_ON_LOAD', True)