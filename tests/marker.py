# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import platform

from packaging.version import parse
import pytest

from brevitas import config
from brevitas import torch_version


def requires_pt_ge(pt_version: str, system: str = None):
    skip = parse(pt_version) > torch_version
    if system is not None:
        skip = skip and platform.system() == system

    def skip_wrapper(f):
        return pytest.mark.skipif(skip, reason=f'Requires Pytorch >= {pt_version}')(f)

    return skip_wrapper


def requires_pt_lt(pt_version: str, system: str = None):
    skip = parse(pt_version) <= torch_version
    if system is not None:
        skip = skip and platform.system() == system

    def skip_wrapper(f):
        return pytest.mark.skipif(skip, reason=f'Requires Pytorch < {pt_version}')(f)

    return skip_wrapper


def jit_disabled_for_export():
    skip = config.JIT_ENABLED

    def skip_wrapper(f):
        return pytest.mark.skipif(skip, reason=f'Export requires JIT to be disabled')(f)

    return skip_wrapper


def jit_disabled_for_mock():
    skip = config.JIT_ENABLED

    def skip_wrapper(f):
        return pytest.mark.skipif(skip, reason=f'Mock requires JIT to be disabled')(f)

    return skip_wrapper


def jit_disabled_for_local_loss():
    skip = config.JIT_ENABLED

    def skip_wrapper(f):
        return pytest.mark.skipif(
            skip, reason=f'Local loss functions (e.g., MSE) require JIT to be disabled')(f)

    return skip_wrapper


def jit_disabled_for_dynamic_quant_act():
    skip = config.JIT_ENABLED

    def skip_wrapper(f):
        return pytest.mark.skipif(skip, reason=f'Dynamic Act Quant requires JIT to be disabled')(f)

    return skip_wrapper


skip_on_macos_nox = pytest.mark.skipif(
    platform.system() == "Darwin", reason="Known issue with Nox and MacOS.")
