# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas import torch_version

from tests.conftest import get_hypothesis_seed

from .sampling import LSTM_MAX_EXAMPLES
from .sampling import LSTM_SHARDS
from .sampling import WBIOL_MAX_EXAMPLES
from .sampling import WBIOL_SHARDS


def pytest_report_header(config):
    # Surface the Hypothesis seed and per-shard example budgets so any run (or failure)
    # is reproducible. The seed is the repo-wide value from tests/conftest.py and can be
    # overridden with --hypothesis-seed.
    wbiol_total = WBIOL_MAX_EXAMPLES * len(WBIOL_SHARDS)
    lstm_total = LSTM_MAX_EXAMPLES * len(LSTM_SHARDS)
    return [
        f'brevitas-ort hypothesis: seed={get_hypothesis_seed()} (torch {torch_version})',
        f'  wbiol: {len(WBIOL_SHARDS)} shards x {WBIOL_MAX_EXAMPLES} examples ~= {wbiol_total}',
        f'  lstm:  {len(LSTM_SHARDS)} shards x {LSTM_MAX_EXAMPLES} examples ~= {lstm_total}',]
