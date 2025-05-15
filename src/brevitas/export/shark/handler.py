# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from sharktank.types import StaticScaledQuantizer
import torch
import torch.nn as nn

from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.proxy import ActFloatQuantProxyFromInjector
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjector
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
            zero_point = module.zero_point().contiguous().to(torch.float32)
            zero_point = None if torch.count_nonzero(zero_point) == 0 else (zero_point - 128.).to(
                self.scale.device)
            self.zero_point = zero_point
            self.bit_width = module.bit_width()

    def forward(self, x):
        assert self.layer_name is not None
        assert self.shared_dict is not None

        weight_quant = StaticScaledQuantizer(
            name=self.layer_name + '.weight',
            scale=torch.reciprocal(self.scale),
            reciprocal_scale=self.scale,
            offset=self.zero_point,
            dtype=torch.int8)
        quant_weight = weight_quant.quantize(x, name=self.layer_name + '.weight')
        self.shared_dict[self.layer_name + '.weight'] = quant_weight
        return x, self.scale, torch.tensor(0.).type_as(self.scale), self.bit_width


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
            zero_point = module.zero_point().contiguous().to(torch.float32)
            zero_point = None if torch.count_nonzero(zero_point) == 0 else (zero_point - 128.).to(
                self.scale.device)
            self.zero_point = zero_point
            self.bit_width = module.bit_width()

    def forward(self, x):
        assert self.layer_name is not None
        assert self.shared_dict is not None

        input_quant = StaticScaledQuantizer(
            name=self.layer_name + '.q_input',
            scale=torch.reciprocal(self.scale),
            reciprocal_scale=self.scale,
            offset=self.zero_point,
            dtype=torch.int8)
        self.shared_dict[self.layer_name + '.q_input'] = input_quant
        return x, self.scale, torch.tensor(0.).type_as(self.scale), self.bit_width


class SharkActFloatQuant(nn.Module):
    handled_layer = ActFloatQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.layer_name = None
        self.shared_dict = None

    def attach_debug_info(self, module: nn.Module):
        pass

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:
            # Continguous is used to be extra-safe with torch.compile

            self.dtype = module.standard_float_dtype
            assert self.dtype is not None
            self.scale = module.scale().contiguous()
            zero_point = module.zero_point().contiguous().to(torch.float32)
            zero_point = None if torch.count_nonzero(zero_point) == 0 else (zero_point - 128.).to(
                self.scale.device)
            self.zero_point = zero_point
            self.mantissa_bit_width = module.mantissa_bit_width()
            self.exponent_bit_width = module.mantissa_bit_width()
            self.exponent_bias = module.mantissa_bit_width()
            self.mantissa_bit_width = module.mantissa_bit_width()

    def forward(self, x):
        assert self.layer_name is not None
        assert self.shared_dict is not None

        input_quant = StaticScaledQuantizer(
            name=self.layer_name,
            scale=torch.reciprocal(self.scale),
            reciprocal_scale=self.scale,
            offset=self.zero_point,
            dtype=self.dtype)
        self.shared_dict[self.layer_name] = input_quant
        return x, self.scale, torch.tensor(0.).type_as(self.scale), self.mantissa_bit_width, self.mantissa_bit_width, self.mantissa_bit_width, self.mantissa_bit_width, None, None


class SharkWeightFloatQuant(nn.Module):
    handled_layer = WeightFloatQuantProxyFromInjector

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
            zero_point = module.zero_point().contiguous().to(torch.float32)
            zero_point = None if torch.count_nonzero(zero_point) == 0 else (zero_point - 128.).to(
                self.scale.device)
            self.zero_point = zero_point
            self.mantissa_bit_width = module.mantissa_bit_width()
            self.exponent_bit_width = module.mantissa_bit_width()
            self.exponent_bias = module.mantissa_bit_width()
            self.mantissa_bit_width = module.mantissa_bit_width()

    def forward(self, x):
        assert self.layer_name is not None
        assert self.shared_dict is not None

        weight_quant = StaticScaledQuantizer(
            name=self.layer_name,
            scale=torch.reciprocal(self.scale).to(device=x.device),
            reciprocal_scale=self.scale.to(device=x.device),
            offset=self.zero_point,
            dtype=torch.float8_e4m3fnuz)
        quant_weight = weight_quant.quantize(x, name=self.layer_name)
        self.shared_dict[self.layer_name] = quant_weight
        return x, self.scale, torch.tensor(0.).type_as(self.scale), self.mantissa_bit_width, self.mantissa_bit_width, self.mantissa_bit_width, self.mantissa_bit_width, None, None
