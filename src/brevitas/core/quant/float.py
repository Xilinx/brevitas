# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

import brevitas
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.scaling import ConstScaling
from brevitas.core.utils import StatelessBuffer
from brevitas.function import compute_max_mantissa
from brevitas.utils.torch_utils import float_internal_scale


def min_internal_scale(exponent_bias, mantissa_bit_width):
    return 1. - exponent_bias - mantissa_bit_width


class FloatQuant(brevitas.jit.ScriptModule):
    __constants__ = ['signed', 'eps']

    def __init__(
            self,
            signed: bool,
            exponent_bit_width_impl: nn.Module,
            mantissa_bit_width_impl: nn.Module,
            exponent_bias_impl: nn.Module,
            float_clamp_impl: nn.Module,
            input_view_impl: nn.Module,
            pre_computed_max_mantissa: nn.Module,
            scaling_impl: Optional[nn.Module] = None,
            float_scaling_impl: Optional[nn.Module] = None,
            float_to_int_impl: nn.Module = RoundSte(),
            device: Optional[str] = None,
            dtype: Optional[torch.dtype] = None):
        super(FloatQuant, self).__init__()

        self.signed: bool = signed
        self.float_to_int_impl = float_to_int_impl

        self.exponent_bias_impl = exponent_bias_impl
        self.exponent_bit_width_impl = exponent_bit_width_impl
        self.mantissa_bit_width_impl = mantissa_bit_width_impl
        if exponent_bit_width_impl() == 0:
            raise RuntimeError("Exponent bit width cannot be 0.")
        if scaling_impl is None:
            scaling_impl = ConstScaling(1., device=device, dtype=dtype)

        self.input_view_impl = input_view_impl
        # Zero-point is currently hardcoded to 0
        self.zero_point_impl = StatelessBuffer(torch.tensor(0., device=device, dtype=dtype))
        self.float_scaling_impl = float_scaling_impl
        self.scaling_impl = scaling_impl
        self.float_clamp_impl = float_clamp_impl

        # To avoid log(0), we add small a small value based on the used dtype
        if dtype is None:
            dtype = torch.get_default_dtype()
        self.eps = torch.finfo(dtype).tiny
        self.observer_only = brevitas.jit.Attribute(False, bool)

        self.pre_computed_max_mantissa = pre_computed_max_mantissa

    @brevitas.jit.script_method
    def quantize(self, x: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_view_impl(x)
        scaled_x = x / scale
        fp_internal_scale_min = min_internal_scale(
            self.exponent_bias_impl(), self.mantissa_bit_width_impl())
        internal_scale = float_internal_scale(
            scaled_x, self.mantissa_bit_width_impl(), fp_internal_scale_min, self.eps)
        val_fp_quant = internal_scale * self.float_to_int_impl(scaled_x / internal_scale)
        return val_fp_quant, scale

    @brevitas.jit.script_method
    def dequantize(self, y, scale):
        return y * scale

    @brevitas.jit.script_method
    def forward(self, x):
        if self.float_scaling_impl is not None:
            float_scaling_impl_value = self.float_scaling_impl(
                self.exponent_bit_width_impl(),
                self.pre_computed_max_mantissa(self.mantissa_bit_width_impl()),
                self.exponent_bias_impl())
        else:
            float_scaling_impl_value = None
        scale = self.scaling_impl(x, float_scaling_impl_value)
        if self.observer_only:
            y = x
            saturating, inf_values, nan_values = self.float_clamp_impl.saturating, self.float_clamp_impl.inf_values, self.float_clamp_impl.nan_values
        else:
            y, scale = self.quantize(x, scale)
            # after quantizing, clamp to special cases like NaN/inf if they are set
            y, saturating, inf_values, nan_values = self.float_clamp_impl(
                y, self.exponent_bit_width_impl(), self.pre_computed_max_mantissa(self.mantissa_bit_width_impl()), self.exponent_bias_impl())
            y = self.dequantize(y, scale)
        # This is to respect the current interface of proxies
        return y, scale, self.zero_point_impl(), self.exponent_bit_width_impl(), self.mantissa_bit_width_impl(), self.exponent_bias_impl(), saturating, inf_values, nan_values
