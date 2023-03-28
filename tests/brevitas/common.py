# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

RTOL = 0
ATOL = 1e-23

FP32_BIT_WIDTH = 32
MIN_INT_BIT_WIDTH = 2
MAX_INT_BIT_WIDTH = 8
INT_BIT_WIDTH_TO_TEST = range(MIN_INT_BIT_WIDTH, MAX_INT_BIT_WIDTH + 1)
BOOLS = [True, False]


def assert_allclose(generated, reference):
    assert torch.allclose(generated, reference, RTOL, ATOL)


def assert_zero_or_none(value):
    if isinstance(value, torch.Tensor):
        assert (value == torch.tensor(0.)).all()
    else:
        assert value is None
