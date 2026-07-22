# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os

from filelock import FileLock
import pytest

# Default dataset location used by brevitas_examples.bnn_pynq.bnn_pynq_train.launch
# (resolved relative to the current working directory, i.e. the repo root under nox).
DATADIR = os.path.abspath(os.path.join(os.getcwd(), 'data'))


def _download_datasets(datadir):
    """Serially download the datasets used by the bnn_pynq example tests.

    The bnn_pynq trainer builds both the train and test splits of MNIST and
    CIFAR10 with ``download=True``. Under ``pytest-xdist`` multiple workers would
    otherwise race to download the same files into ``datadir`` on first access,
    corrupting them. Downloading them once, serially, before the tests run avoids
    the race.
    """
    from torchvision.datasets import CIFAR10

    from brevitas_examples.bnn_pynq.trainer import MirrorMNIST

    os.makedirs(datadir, exist_ok=True)
    for builder in (MirrorMNIST, CIFAR10):
        for train in (True, False):
            builder(root=datadir, train=train, download=True)


@pytest.fixture(scope='session', autouse=True)
def bnn_pynq_datasets():
    """Ensure the bnn_pynq datasets are available before any xdist worker uses them.

    A file lock plus a sentinel file guarantee that only the first process
    performs the (serial) download while the others wait and then reuse the
    cached copy.
    """
    try:
        import torchvision  # noqa: F401

        from brevitas_examples.bnn_pynq.trainer import MirrorMNIST  # noqa: F401
    except ImportError:
        # torchvision / bnn_pynq deps not installed in this session
        # (e.g. the LLM sessions); nothing to pre-download.
        yield
        return
    os.makedirs(DATADIR, exist_ok=True)
    lock_path = os.path.join(DATADIR, '.download.lock')
    sentinel_path = os.path.join(DATADIR, '.download.done')
    with FileLock(lock_path):
        if not os.path.exists(sentinel_path):
            _download_datasets(DATADIR)
            with open(sentinel_path, 'w') as f:
                f.write('ok')
    yield
