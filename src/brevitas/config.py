import os
from distutils.util import strtobool
try:
    from torch.jit import _enabled
except ImportError:
    from torch.jit._state import _enabled


def env_to_bool(name, default):
    return bool(strtobool(os.environ.get(name, "{}".format(default))))


IGNORE_MISSING_KEYS = env_to_bool('BREVITAS_IGNORE_MISSING_KEYS', False)
JIT_ENABLED = env_to_bool('BREVITAS_JIT', False) and _enabled
VERBOSE = env_to_bool('BREVITAS_VERBOSE', False)

# Internal global variables
_IS_INSIDE_QUANT_LAYER = None
_ONGOING_EXPORT = None

