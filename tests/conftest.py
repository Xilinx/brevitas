# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import platform
import zlib

import torch

SEED = 123456
torch.random.manual_seed(SEED)

MIN_QONNX_VERSION = '0.5.0'

# Name of the Hypothesis profile registered/loaded for the whole test suite.
HYPOTHESIS_PROFILE = 'brevitas'


def get_hypothesis_seed():
    """Return the global seed used for *all* Hypothesis-driven randomisation.

    This is the single source of truth for Hypothesis test seeding across the repo.
    It is applied globally in :func:`pytest_configure` via ``core.global_force_seed``
    (equivalent to passing ``--hypothesis-seed``), so every ``@given`` test - not just
    the ORT integration tests - becomes reproducible from it.

    The seed is derived *intrinsically* from the environment (Python version, torch
    version, platform) rather than from an external variable. Consequently:

    * A given machine / CI matrix job is fully deterministic and reproducible.
    * Different matrix jobs (python x pytorch x platform) explore different examples,
      so combinatorial coverage still accumulates across the matrix over time.

    ``hash()`` is deliberately avoided: builtin string hashing is randomised per process
    (``PYTHONHASHSEED``), which would make xdist workers disagree on collected/generated
    examples. ``zlib.crc32`` is stable across processes and machines.

    An explicit ``--hypothesis-seed`` on the command line still takes precedence (see
    :func:`pytest_configure`).
    """
    fingerprint = '|'.join([
        platform.python_version(),
        torch.__version__,
        platform.system(),
        platform.machine(),])
    return zlib.crc32(fingerprint.encode('utf-8'))


def pytest_configure(config):
    """Register a repo-wide Hypothesis profile and apply the global seed.

    The profile disables the per-example deadline (ORT export + inference is far slower
    than Hypothesis' 200ms default) and suppresses the slow/function-scoped-fixture
    health checks. It is loaded here - not in a strategy helper module - so it applies
    to test directories (e.g. tests/brevitas_ort) that don't import those helpers.
    """
    from hypothesis import core
    from hypothesis import HealthCheck
    from hypothesis import settings

    suppress = [HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
    settings.register_profile(HYPOTHESIS_PROFILE, deadline=None, suppress_health_check=suppress)
    settings.load_profile(HYPOTHESIS_PROFILE)

    # Apply the global seed unless the user explicitly overrode it on the command line
    # (--hypothesis-seed). Setting core.global_force_seed is exactly what the Hypothesis
    # pytest plugin does for that flag; it has lower precedence than an in-test @seed(...)
    # and than settings.derandomize (which we intentionally do not enable, so that this
    # seed - not a per-test digest - controls generation).
    explicit_seed = config.getoption('--hypothesis-seed', default=None)
    if explicit_seed is None and core.global_force_seed is None:
        core.global_force_seed = get_hypothesis_seed()
