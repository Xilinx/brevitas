# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import platform
import zlib

import torch

SEED = 123456
torch.random.manual_seed(SEED)

MIN_QONNX_VERSION = '0.5.0'


def get_hypothesis_seed():
    """Global seed for all Hypothesis randomisation, derived from the environment.

    Uses zlib.crc32 (not builtin hash(), which is per-process randomised) so xdist workers agree.
    """
    fingerprint = '|'.join([
        platform.python_version(), torch.__version__, platform.system(), platform.machine()])
    return zlib.crc32(fingerprint.encode('utf-8'))


def pytest_configure(config):
    # Apply the global seed unless the user passed --hypothesis-seed (mirrors the plugin).
    from hypothesis import core
    if config.getoption('--hypothesis-seed',
                        default=None) is None and core.global_force_seed is None:
        core.global_force_seed = get_hypothesis_seed()
