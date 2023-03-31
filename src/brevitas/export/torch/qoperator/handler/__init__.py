# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

try:
    import torch.nn.quantized.functional as qF

# Skip for pytorch 1.1.0
except:
    qF = None
