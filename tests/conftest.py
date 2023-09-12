# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import hypothesis
from packaging import version
import torch

SEED = 123456
DIFFERING_EXECUTOR_ENUM = 10  # Disabled since it fails with normal test setup
HYPOTHESIS_HEALTHCHECK_VERSION = '6.83.2'  # The new healthcheck was introduced in this version

torch.random.manual_seed(SEED)
if version.parse(hypothesis.__version__) >= version.parse(HYPOTHESIS_HEALTHCHECK_VERSION):
    hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck(DIFFERING_EXECUTOR_ENUM)])
