# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import hypothesis
import torch

SEED = 123456
DIFFERING_EXECUTOR_ENUM = 10  # Disabled since it fails with normal test setup

torch.random.manual_seed(SEED)
hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck(DIFFERING_EXECUTOR_ENUM)])
