# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

try:
    import torch

    # Avoid fast algorithms that might introduce extra error during fake-quantization
    torch.use_deterministic_algorithms(True)
except:
    # Introduced in 1.8.0
    pass
