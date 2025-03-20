# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from sharktank.types import StaticScaledQuantizer
import torch
import torch.nn as nn

from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector


class SharkWeightQuant(nn.Module):
    handled_layer = WeightQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.layer_name = None
        self.shared_dict = None

    def attach_debug_info(self, module: nn.Module):
        pass

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:
            # Continguous is used to be extra-safe with torch.compile
            self.scale = module.scale().contiguous()
            self.zero_point = module.zero_point().contiguous()
            self.zero_point = self.zero_point.to(self.scale.device)
            self.bit_width = module.bit_width()
            self.min_clamp = min_int(module.is_signed, module.is_narrow_range, self.bit_width)
            self.max_clamp = max_int(module.is_signed, module.is_narrow_range, self.bit_width)

    def forward(self, x):
        assert self.layer_name is not None
        assert self.shared_dict is not None

        zero_point = None if torch.count_nonzero(self.zero_point) == 0 else self.zero_point
        if zero_point:
            zero_point -= 128  # TODO: check
        weight_quant = StaticScaledQuantizer(
            scale=torch.reciprocal(self.scale),
            reciprocal_scale=self.scale,
            offset=zero_point,
            dtype=torch.int8)
        quant_weight = weight_quant.quantize(x)
        self.shared_dict[self.layer_name + 'weight'] = quant_weight
        return x, self.scale, self.zero_point, self.bit_width


class SharkActQuant(nn.Module):
    handled_layer = ActQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.layer_name = None
        self.shared_dict = None

    def attach_debug_info(self, module: nn.Module):
        pass

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:
            # Continguous is used to be extra-safe with torch.compile
            self.scale = module.scale().contiguous()
            self.zero_point = module.zero_point().contiguous()
            self.zero_point = self.zero_point.to(self.scale.device)
            self.bit_width = module.bit_width()
            self.min_clamp = min_int(module.is_signed, module.is_narrow_range, self.bit_width)
            self.max_clamp = max_int(module.is_signed, module.is_narrow_range, self.bit_width)

    def forward(self, x):
        assert self.layer_name is not None
        assert self.shared_dict is not None

        zero_point = None if torch.count_nonzero(self.zero_point) == 0 else self.zero_point
        if zero_point:
            zero_point -= 128  # TODO: check
        input_quant = StaticScaledQuantizer(
            scale=torch.reciprocal(self.scale),
            reciprocal_scale=self.scale,
            offset=zero_point,
            dtype=torch.int8)
        self.shared_dict[self.layer_name + 'q_input'] = input_quant
        return x, self.scale, self.zero_point, self.bit_width
