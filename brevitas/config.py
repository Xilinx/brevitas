import os
from distutils.util import strtobool


def env_to_bool(name, default):
    return bool(strtobool(os.environ.get(name, "{}".format(default))))


MIN_TORCH_JITTABLE_VERSION = "1.3.0"
MAX_TORCH_JITTABLE_VERSION = "1.4.0"

IGNORE_MISSING_KEYS = env_to_bool('BREVITAS_IGNORE_MISSING_KEYS', False)
