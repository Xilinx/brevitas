# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import glob
import os
from typing import List, Optional
import warnings

from packaging import version
from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution
import torch
from torch import Tensor
from torch.utils import cpp_extension

from brevitas import config
from brevitas import jit as jit

pkg_dir = os.path.dirname(os.path.abspath(__file__))

if torch.__version__.endswith('+cpu'):
    torch_version = version.parse(torch.__version__.rstrip('+cpu'))
else:
    torch_version = version.parse(torch.__version__)

try:
    # Attempt _dynamo import
    is_dynamo_compiling = torch._dynamo.is_compiling
except:
    is_dynamo_compiling = lambda: False

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

if config.JIT_ENABLED or config.NATIVE_STE_BACKEND_ENABLED:
    config.NATIVE_STE_BACKEND_ENABLED = True  # for consistency, in case only JIT_ENABLED was true
    extensions_dir = os.path.join(pkg_dir, 'csrc')
    sources = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    sources = [os.path.join(extensions_dir, s) for s in sources]

    try:
        cpp_extension.load(
            name='autograd_ste_ops',
            sources=sources,
            is_python_module=False,
            verbose=config.VERBOSE)
        NATIVE_STE_BACKEND_LOADED = True
    except Exception as e:
        if config.VERBOSE:
            # Warnings calls str on the message argument, can't pass an f-string directly
            error_message = (
                f"The Brevitas native STE backend is enabled but couldn't be loaded.\n"
                f"Ensure that the \"ninja\" build system is installed (e.g. apt install ninja-build)"
                f"\nException: {e}.")
            warnings.warn(error_message)
        NATIVE_STE_BACKEND_LOADED = False
else:
    NATIVE_STE_BACKEND_LOADED = False

from brevitas import ops as ops
