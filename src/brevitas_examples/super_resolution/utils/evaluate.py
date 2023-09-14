# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch import Tensor
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.nn.utils import calculate_min_accumulator_bit_width
from brevitas_examples.super_resolution.models.espcn import QuantESPCN


def _calc_min_acc_bit_width(module: QuantWBIOL) -> Tensor:
    # bit-width and sign need to come from the quant tensor of the preceding layer if no io_quant
    input_bit_width = module.quant_input_bit_width()
    input_is_signed = module.is_quant_input_signed

    # the tensor quantizer requires a QuantTensor with specified bit-width and sign
    quant_weight = module.quant_weight()
    quant_weight = quant_weight.int().float()
    if isinstance(module,
                  qnn.QuantConv2d):  # shape = (out_channels, in_channels, kernel_size, kernel_size)
        quant_weight_per_channel_l1_norm = quant_weight.norm(p=1, dim=(1, 2, 3))

    # using the closed-form bounds on accumulator bit-width
    cur_acc_bit_width = calculate_min_accumulator_bit_width(
        input_bit_width, input_is_signed, quant_weight_per_channel_l1_norm.max())
    return cur_acc_bit_width


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
