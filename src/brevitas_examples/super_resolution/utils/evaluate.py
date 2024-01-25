# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL

EPS = 1e-10


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
    weight_max_l1_norm = quant_weight_per_channel_l1_norm.max()
    weight_max_l1_norm = torch.clamp_min(weight_max_l1_norm, EPS)
    alpha = torch.log2(weight_max_l1_norm) + input_bit_width - input_is_signed
    phi = lambda x: torch.log2(1. + pow(2., -x))
    min_bit_width = alpha + phi(alpha) + 1.
    min_bit_width = torch.ceil(min_bit_width)
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
