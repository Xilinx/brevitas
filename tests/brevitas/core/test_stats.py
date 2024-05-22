# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

from brevitas.core.stats import AbsPercentile
from brevitas.core.stats import NegativePercentileOrZero
from brevitas.core.stats import PercentileInterval
# Use custom implementation of kthvalue as work around to (b)float16 kernel limitations
from brevitas.utils.torch_utils import kthvalue


def test_abs_percentile_per_tensor():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for v in values:
        tensor = torch.Tensor(values)
        abs_percentile = AbsPercentile(v * 10, None)
        out = abs_percentile(tensor)
        assert v == out.item()


def test_abs_percentile_per_channel():
    v = 90
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tensor = torch.Tensor(values)
    tensor = tensor.repeat(2, 1)
    abs_percentile = AbsPercentile(v, stats_reduce_dim=1)
    out = abs_percentile(tensor)
    assert out.isclose(torch.Tensor([9, 9])).all().item()


class TestPercentile:

    def compute_percentile(self, x, low_q=None, high_q=None):
        low_p, high_p = None, None
        if low_q is not None:
            k = int(math.ceil(.01 * low_q * x.numel()))
            low_p = kthvalue(x.view(-1), k=k)[0]
        if high_q is not None:
            k = int(math.floor(.01 * high_q * x.numel() + 0.5))
            high_p = kthvalue(x.view(-1), k=k)[0]
        return low_p, high_p

    def test_negative_percentile(self):
        values = [-1., -2., 5]
        values = torch.tensor(values)
        neg_percentile = NegativePercentileOrZero(0.01)
        out = neg_percentile(values)

        expected_out = torch.min(torch.tensor(0.), self.compute_percentile(values, low_q=0.01)[0])

        assert torch.allclose(out, expected_out)

    def test_zero_percentile(self):
        values = [1., 2., 5]
        values = torch.tensor(values)
        neg_percentile = NegativePercentileOrZero(0.01)
        out = neg_percentile(values)

        expected_out = torch.tensor(0.)

        assert torch.allclose(out, expected_out)

    def test_interval_percentile(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        values = torch.tensor(values, dtype=torch.float32)
        interval_percentile = PercentileInterval(low_percentile_q=0.01, high_percentile_q=99.9)
        out = interval_percentile(values)

        range = self.compute_percentile(values, low_q=0.01, high_q=99.9)
        # Clamp is to make sure the lower bound is not positive to align with zero-point statistics
        low_result = torch.clamp(range[0], max=torch.tensor(0.0))
        expected_out = torch.abs(range[1] - low_result)
        assert torch.allclose(out, expected_out)
