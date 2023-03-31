# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from distutils.util import strtobool
import os

try:
    from torch.jit import _enabled
except ImportError:
    from torch.jit._state import _enabled


def env_to_bool(name, default):
    return bool(strtobool(os.environ.get(name, "{}".format(default))))


REINIT_ON_STATE_DICT_LOAD = env_to_bool('BREVITAS_REINIT_ON_STATE_DICT_LOAD', True)
IGNORE_MISSING_KEYS = env_to_bool('BREVITAS_IGNORE_MISSING_KEYS', False)
# JIT_ENABLED triggers NATIVE_STE_BACKEND_ENABLED to True, but not the other way around
JIT_ENABLED = env_to_bool('BREVITAS_JIT', False) and _enabled
NATIVE_STE_BACKEND_ENABLED = env_to_bool('BREVITAS_NATIVE_STE_BACKEND', False)
VERBOSE = env_to_bool('BREVITAS_VERBOSE', False)

# Internal global variables
_IS_INSIDE_QUANT_LAYER = None
_ONGOING_EXPORT = None
