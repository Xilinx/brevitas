"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from torch import nn

from brevitas import nn as qnn
from brevitas.graph.base import ModuleToModuleByInstance


def quantize(model, weight_quant, weight_bit_width, weight_block_size):
    """
    Replace float layers with quant layers in the target model
    """
    transforms = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_to_quant_linear = ModuleToModuleByInstance(
                module,
                qnn.QuantLinear,
                weight_quant=weight_quant,
                weight_bit_width=weight_bit_width,
                weight_block_size=weight_block_size)
            transforms.append(linear_to_quant_linear)
    for t in transforms:
        model = t.apply(model)
