# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import platform
import zlib

import torch

SEED = 123456
torch.random.manual_seed(SEED)

MIN_QONNX_VERSION = '0.5.0'


def get_hypothesis_seed():
    """Global seed for all Hypothesis-driven randomisation (single source of truth).

    Derived intrinsically from the environment (Python/torch/platform) rather than an
    external variable, so a given machine / CI matrix job is reproducible while different
    jobs still explore different examples. ``zlib.crc32`` (not builtin ``hash()``, which is
    per-process randomised) keeps xdist workers in agreement.
    """
    fingerprint = '|'.join(
        [platform.python_version(), torch.__version__, platform.system(), platform.machine()])
    return zlib.crc32(fingerprint.encode('utf-8'))


def pytest_configure(config):
    # Apply the global Hypothesis seed unless the user passed --hypothesis-seed. This is the
    # same mechanism the Hypothesis pytest plugin uses for that flag.
    from hypothesis import core
    if config.getoption('--hypothesis-seed', default=None) is None and core.global_force_seed is None:
        core.global_force_seed = get_hypothesis_seed()
