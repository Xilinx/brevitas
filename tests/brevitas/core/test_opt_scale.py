# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import pytest_cases
import torch

from brevitas.core.stats.stats_op import OptimalIntSymmetricScale
from tests.conftest import SEED

# Number of weights to generate
P = 100
GRID_SEARCH_ITERS = 1000
ATOL = 1. / GRID_SEARCH_ITERS


class TestScale:

    def test_optimal_scale_ternary(self):
        # Quantized values are {-1, 0, 1}
        N = 1
        # Generate a vector of random weights
        x = torch.rand((P,), dtype=torch.float32)

        # Optimal scale in the ternary case admits a closed-form solution
        # See https://arxiv.org/pdf/1707.04319
        abs_sorted_x, _ = torch.sort(torch.abs(x), descending=True)
        j_optimal = torch.argmax(
            torch.cumsum(abs_sorted_x, dim=-1) / torch.sqrt(torch.arange(start=1, end=P + 1)))
        gt_optimal_scale = torch.sum(abs_sorted_x[:j_optimal + 1]) / (j_optimal + 1)

        optimal_int_symmetric_scale = OptimalIntSymmetricScale(N=N)
        optimal_scale = optimal_int_symmetric_scale(x)

        # Compare scales
        assert torch.allclose(gt_optimal_scale, optimal_scale)

    @pytest_cases.parametrize("N", [2, 3, 5])
    # Quantized values are {-N, ..., 0, ..., 1}
    def test_optimal_scale_grid_search(self, N):
        # Generate a vector of random weights
        x = torch.rand((P,), dtype=torch.float32)

        # Compute optimal scale
        optimal_int_symmetric_scale = OptimalIntSymmetricScale(N=N)
        optimal_scale = optimal_int_symmetric_scale(x)

        # Compare with that obtained via grid-search
        def error_closure(scale):
            return torch.sum(torch.square(x - scale * torch.clamp(torch.round(x / scale), -N, N)))

        gt_optimal_scale = None
        gt_optimal_error = float('inf')

        for i in range(GRID_SEARCH_ITERS):
            curr_scale = torch.tensor(i / GRID_SEARCH_ITERS, dtype=torch.float32)
            curr_error = error_closure(curr_scale)
            if curr_error < gt_optimal_error:
                gt_optimal_error = curr_error
                gt_optimal_scale = curr_scale

        torch.allclose(optimal_scale, gt_optimal_scale, atol=ATOL, rtol=1e-1)
