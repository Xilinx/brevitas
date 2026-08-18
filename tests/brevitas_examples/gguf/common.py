# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gguf
from gguf import QK_K
import numpy as np

Q4_0 = gguf.GGMLQuantizationType.Q4_0
Q4_1 = gguf.GGMLQuantizationType.Q4_1
Q8_0 = gguf.GGMLQuantizationType.Q8_0
Q2_K = gguf.GGMLQuantizationType.Q2_K
Q3_K = gguf.GGMLQuantizationType.Q3_K
Q4_K = gguf.GGMLQuantizationType.Q4_K
Q5_K = gguf.GGMLQuantizationType.Q5_K
Q6_K = gguf.GGMLQuantizationType.Q6_K


def fp16(a):
    """Round-trip through fp16, matching how scales are stored on disk."""
    return a.astype(np.float16).astype(np.float32)


def normal(seed: int, nb: int, block: int = QK_K):
    return np.random.default_rng(seed).standard_normal((nb, block)).astype(np.float32)


def _outlier():
    # A large spike forces a wide sub-block scale and the [-32, 31] clamp.
    x = normal(9, 4)
    x[:, 0] = 50.0
    return x


# Random model tensors varying block counts, constants (incl. negative),
# zero, and a high-dynamic-range spike.
MODEL_TENSORS = {
    "normal_1blk": normal(0, 1),
    "normal_4blk": normal(1, 4),
    "normal_17blk": normal(7, 17),
    "const_pos": np.full((2, QK_K), 0.37, dtype=np.float32),
    "const_large": np.full((2, QK_K), 5.0, dtype=np.float32),
    "const_neg": np.full((2, QK_K), -2.3, dtype=np.float32),
    "zero": np.zeros((2, QK_K), dtype=np.float32),
    "outlier": _outlier(),}
