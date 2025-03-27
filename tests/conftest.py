# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

SEED = 123456
torch.random.manual_seed(SEED)

MIN_QONNX_VERSION = '0.5.0'
