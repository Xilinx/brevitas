# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple

import torch
import torch.nn as nn

import brevitas
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.scaling import ConstScaling
from brevitas.core.utils import StatelessBuffer
from brevitas.function.ops import float_internal_scale
from brevitas.function.ops_ste import floor_ste


class FloatQuant(brevitas.jit.ScriptModule):
    __constants__ = ['signed']

    def __init__(
            self,
            bit_width: int,
            signed: bool,
            exponent_bit_width: int,
            mantissa_bit_width: int,
            exponent_bias: int,
            float_clamp_impl: nn.Module,
            scaling_impl: Optional[nn.Module] = None,
            float_scaling_impl: Optional[nn.Module] = None,
            float_to_int_impl: nn.Module = RoundSte(),
            device: Optional[str] = None,
            dtype: Optional[torch.dtype] = None):
        super(FloatQuant, self).__init__()
        if bit_width != exponent_bit_width + mantissa_bit_width + int(signed):
            raise RuntimeError("Mismatch between total bit-width, exponent, mantissa and sign.")
        self.bit_width = StatelessBuffer(torch.tensor(float(bit_width), device=device, dtype=dtype))
        self.signed: bool = signed
        self.float_to_int_impl = float_to_int_impl
        if exponent_bit_width == 0:
            raise RuntimeError("Exponent bit width cannot be 0.")
        self.exponent_bit_width = StatelessBuffer(
            torch.tensor(float(exponent_bit_width), device=device, dtype=dtype))
        if mantissa_bit_width == 0:
            raise RuntimeError("Mantissa bit width cannot be 0.")
        self.mantissa_bit_width = StatelessBuffer(
            (torch.tensor(float(mantissa_bit_width), device=device, dtype=dtype)))
        self.exponent_bias = StatelessBuffer(
            torch.tensor(float(exponent_bias), device=device, dtype=dtype))

        self.fp_internal_scale_min = StatelessBuffer(
            1. - self.exponent_bias() - self.mantissa_bit_width())

        if scaling_impl is None:
            scaling_impl = ConstScaling(1., device=device, dtype=dtype)

        # Zero-point is currently hardcoded to 0
        self.zero_point_impl = StatelessBuffer(torch.tensor(0., device=device, dtype=dtype))
        self.float_scaling_impl = float_scaling_impl
        self.scaling_impl = scaling_impl
        self.float_clamp_impl = float_clamp_impl

    @brevitas.jit.script_method
    def quantize(self, x: torch.Tensor):
        scaling_impl_value = self.scaling_impl(x)
        float_scaling_impl_value = self.float_scaling_impl(
            self.exponent_bit_width(), self.mantissa_bit_width(), self.exponent_bias())
        scale = scaling_impl_value / float_scaling_impl_value
        scaled_x = x / scale
        internal_scale = float_internal_scale(
            scaled_x, self.mantissa_bit_width(), self.fp_internal_scale_min())
        val_fp_quant = internal_scale * self.float_to_int_impl(scaled_x / internal_scale)
        return val_fp_quant, scale

    @brevitas.jit.script_method
    def dequantize(self, y, scale):
        return y * scale

    @brevitas.jit.script_method
    def forward(self, x):
        y, scale = self.quantize(x)
        # after quantizing, clamp to special cases like NaN/inf if they are set
        y, saturating, inf_values, nan_values = self.float_clamp_impl(
            y, self.exponent_bit_width(), self.mantissa_bit_width(), self.exponent_bias())
        y = self.dequantize(y, scale)
        # This is to respect the current interface of proxies
        return y, scale, self.zero_point_impl(), self.exponent_bit_width(), self.mantissa_bit_width(), self.exponent_bias(), saturating, inf_values, nan_values
