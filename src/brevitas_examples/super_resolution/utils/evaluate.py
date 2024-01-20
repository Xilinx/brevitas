# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor
import torch.nn as nn

from brevitas.core.scaling import AccumulatorAwareParameterPreScaling
from brevitas.core.scaling import AccumulatorAwareZeroCenterParameterPreScaling
import brevitas.nn as qnn
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL

EPS = 1e-10


def _get_a2q_module(module: nn.Module):
    for submod in module.modules():
        if isinstance(submod, AccumulatorAwareParameterPreScaling):
            return submod
    return None


def _calc_a2q_acc_bit_width(
        weight_max_l1_norm: Tensor, input_bit_width: Tensor, input_is_signed: bool):
    """Using the closed-form bounds on accumulator bit-width as derived in
    `A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance`.
    This function returns the minimum accumulator bit-width that can be used
    without risk of overflow."""
    assert weight_max_l1_norm.numel() == 1
    input_is_signed = float(input_is_signed)
    weight_max_l1_norm = torch.clamp_min(weight_max_l1_norm, EPS)
    alpha = torch.log2(weight_max_l1_norm) + input_bit_width - input_is_signed
    phi = lambda x: torch.log2(1. + pow(2., -x))
    min_bit_width = alpha + phi(alpha) + 1.
    min_bit_width = torch.ceil(min_bit_width)
    return min_bit_width


def _calc_a2q_plus_acc_bit_width(
        weight_max_l1_norm: Tensor, input_bit_width: Tensor, input_is_signed: bool):
    """Using the closed-form bounds on accumulator bit-width as derived in `A2Q+:
    Improving Accumulator-Aware Weight Quantization`. This function returns the
    minimum accumulator bit-width that can be used without risk of overflow,
    assuming that the floating-point weights are zero-centered."""
    input_is_signed = float(input_is_signed)
    assert weight_max_l1_norm.numel() == 1
    weight_max_l1_norm = torch.clamp_min(weight_max_l1_norm, EPS)
    input_range = pow(2., input_bit_width) - 1.  # 2^N - 1.
    min_bit_width = torch.log2(weight_max_l1_norm * input_range + 2.)
    min_bit_width = torch.ceil(min_bit_width)
    return min_bit_width


def _calc_min_acc_bit_width(module: QuantWBIOL) -> Tensor:
    assert isinstance(module, qnn.QuantConv2d), "Error: function only support QuantConv2d."

    # bit-width and sign need to come from the quant tensor of the preceding layer if no io_quant
    input_bit_width = module.quant_input_bit_width()
    input_is_signed = float(module.is_quant_input_signed)

    # the tensor quantizer requires a QuantTensor with specified bit-width and sign
    quant_weight = module.quant_weight()
    quant_weight = quant_weight.int().float()
    quant_weight_per_channel_l1_norm = quant_weight.norm(p=1, dim=(1, 2, 3))

    # using the closed-form bounds on accumulator bit-width
    min_bit_width = _calc_a2q_acc_bit_width(
        quant_weight_per_channel_l1_norm.max(),
        input_bit_width=input_bit_width,
        input_is_signed=input_is_signed)
    if isinstance(_get_a2q_module(module), AccumulatorAwareZeroCenterParameterPreScaling):
        min_bit_width = _calc_a2q_plus_acc_bit_width(
            quant_weight_per_channel_l1_norm.max(),
            input_bit_width=input_bit_width,
            input_is_signed=input_is_signed)
    return min_bit_width


def evaluate_accumulator_bit_widths(model: nn.Module, inp: Tensor):
    model(inp)  # collect quant inputs now that caching is enabled
    stats = dict()
    for name, module in model.named_modules():
        # ESPCN only has quantized conv2d nodes and the last one (i.e., conv4.conv) is decoupled
        # from the input quantizer. Will check for more layer types in the future with other
        # example models with different neural architectures.
        if isinstance(module, qnn.QuantConv2d):
            acc_bit_width = _calc_min_acc_bit_width(module)
            stats[name] = acc_bit_width.item()
    return stats
