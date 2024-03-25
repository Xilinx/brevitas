# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.core.quant import QuantType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import StatsOp
from brevitas.nn import QuantReLU

BIT_WIDTH = 8
MAX_VAL = 6.0
RANDOM_ITERS = 32


class TestQuantReLU:

    def test_scaling_stats_to_parameter(self):

        stats_act = QuantReLU(
            bit_width=BIT_WIDTH,
            max_val=MAX_VAL,
            quant_type=QuantType.INT,
            scaling_impl_type=ScalingImplType.STATS,
            scaling_stats_permute_dims=None,
            scaling_stats_op=StatsOp.MAX)
        stats_act.train()
        for i in range(RANDOM_ITERS):
            inp = torch.randn([8, 3, 64, 64])
            stats_act(inp)

        stats_state_dict = stats_act.state_dict()

        param_act = QuantReLU(
            bit_width=BIT_WIDTH,
            max_val=MAX_VAL,
            quant_type=QuantType.INT,
            scaling_impl_type=ScalingImplType.PARAMETER)
        param_act.load_state_dict(stats_state_dict)

        stats_act.eval()
        param_act.eval()

        assert (torch.allclose(stats_act.act_quant.scale(), param_act.act_quant.scale()))

    def test_scaling_parameter_grad(self):
        stats_act = QuantReLU(
            bit_width=BIT_WIDTH,
            max_val=MAX_VAL,
            quant_type=QuantType.INT,
            scaling_impl_type=ScalingImplType.PARAMETER)
        stats_act.train()
        for i in range(RANDOM_ITERS):
            inp = torch.randn([8, 3, 64, 64])
            stats_act(inp)
            out = stats_act(inp)
            out.sum().backward()
            tensor_quant = stats_act.act_quant.fused_activation_quant_proxy.tensor_quant
            scaling_value = tensor_quant.scaling_impl.value
            assert scaling_value.grad is not None

    def test_scaling_parameter_from_stats(self):
        shape = [8, 3, 64, 64]
        collect_stats_steps = 100
        stats_act = QuantReLU(
            bit_width=BIT_WIDTH,
            quant_type=QuantType.INT,
            scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS,
            scaling_stats_permute_dims=None,
            scaling_stats_op=StatsOp.PERCENTILE,
            collect_stats_steps=collect_stats_steps,
            scaling_min_val=None,
            high_percentile_q=99.0)
        stats_act.train()
        tensor_quant = stats_act.act_quant.fused_activation_quant_proxy.tensor_quant
        scaling_value = tensor_quant.scaling_impl.value
        for i in range(collect_stats_steps):
            inp = torch.randn(shape)
            out = stats_act(inp)
            out.requires_grad_(True)  # i need something to require a grad
            out.sum().backward()
            assert scaling_value.grad == 0.
        inp = torch.randn(shape)
        out = stats_act(inp)
        out.sum().backward()
        assert scaling_value.grad != 0.
