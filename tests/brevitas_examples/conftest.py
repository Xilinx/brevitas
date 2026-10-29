# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from tests.brevitas_examples.assets_for_tests import prepare_bnn_pynq_datasets


@pytest.fixture(scope='session')
def bnn_pynq_datasets():
    """Ensure the bnn_pynq datasets are available before any xdist worker uses them.

    This fixture is opt-in (not ``autouse``): only tests that request it, such as the bnn_pynq
    pretrained accuracy test, trigger the download, so unrelated sessions (LLM, vision,
    diffusion) are unaffected.
    """
    prepare_bnn_pynq_datasets()
    yield
