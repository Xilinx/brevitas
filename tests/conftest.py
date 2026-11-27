# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import platform
import zlib

import pytest
import torch

SEED = 123456
torch.random.manual_seed(SEED)

MIN_QONNX_VERSION = '0.5.0'

# Set when the user fixes the Hypothesis seed (--hypothesis-seed); it then applies to every test.
_USER_HYPOTHESIS_SEED = False
_HYPOTHESIS_SEED_KEY = pytest.StashKey[int]()


def get_hypothesis_seed_salt():
    """Per-run salt: scheduled (periodic) CI draws new examples on every run.

    GITHUB_RUN_ID is unchanged when a job is re-run, so CI re-runs reproduce the original draws.
    """
    if os.environ.get('GITHUB_EVENT_NAME') == 'schedule':
        return os.environ.get('GITHUB_RUN_ID', '')
    return ''


def get_hypothesis_seed(nodeid=''):
    """Hypothesis seed derived from the environment, the scheduled run and the test node id.

    Including the node id gives each parametrisation of a test its own examples. Uses zlib.crc32
    (not builtin hash(), which is per-process randomised) so xdist workers agree.
    """
    fingerprint = '|'.join([
        platform.python_version(),
        torch.__version__,
        platform.system(),
        platform.machine(),
        get_hypothesis_seed_salt(),
        nodeid])
    return zlib.crc32(fingerprint.encode('utf-8'))


def pytest_configure(config):
    # Apply an environment-derived seed unless the user passed --hypothesis-seed (mirrors the
    # plugin), in which case that seed is used unchanged for every test.
    global _USER_HYPOTHESIS_SEED
    from hypothesis import core
    if config.getoption('--hypothesis-seed',
                        default=None) is None and core.global_force_seed is None:
        core.global_force_seed = get_hypothesis_seed()
    else:
        _USER_HYPOTHESIS_SEED = True


def pytest_report_header(config):
    if _USER_HYPOTHESIS_SEED:
        return 'hypothesis seed: fixed by --hypothesis-seed'
    return (
        f"hypothesis seed: per test, GITHUB_EVENT_NAME={os.environ.get('GITHUB_EVENT_NAME')!r}, "
        f"salt={get_hypothesis_seed_salt()!r}")


def _is_hypothesis_test(item):
    return getattr(getattr(item, 'obj', None), 'is_hypothesis_test', False)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    # Hypothesis reads core.global_force_seed when the test runs, so set it per test.
    if _USER_HYPOTHESIS_SEED or not _is_hypothesis_test(item):
        yield
        return
    from hypothesis import core
    seed = get_hypothesis_seed(item.nodeid)
    item.stash[_HYPOTHESIS_SEED_KEY] = seed
    previous_seed = core.global_force_seed
    core.global_force_seed = seed
    try:
        yield
    finally:
        core.global_force_seed = previous_seed


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    seed = item.stash.get(_HYPOTHESIS_SEED_KEY, None)
    if seed is not None and report.when == 'call' and report.failed:
        # GITHUB_EVENT_NAME also selects the number of examples in some tests, so keep it.
        event = os.environ.get('GITHUB_EVENT_NAME')
        env = f'GITHUB_EVENT_NAME={event} ' if event else ''
        report.sections.append((
            'hypothesis seed',
            f'seed={seed} (GITHUB_EVENT_NAME={event!r}, salt={get_hypothesis_seed_salt()!r})\n'
            f'Reproduce with the same Python/torch/OS:\n'
            f'  {env}pytest "{item.nodeid}" --hypothesis-seed={seed}'))
