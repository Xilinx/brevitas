# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import torch
import torch.nn as nn

import brevitas
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.scaling import ConstScaling
from brevitas.core.utils import StatelessBuffer
from brevitas.function.ops import max_float
from brevitas.function.ops_ste import floor_ste


class FloatQuant(brevitas.jit.ScriptModule):
    __constants__ = ['signed']

    def __init__(
            self,
            bit_width: int,
            signed: bool,
            exponent_bit_width: int,
            mantissa_bit_width: int,
            exponent_bias: Optional[int] = None,
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
        if exponent_bias is None:
            exponent_bias = 2 ** (exponent_bit_width - 1) - 1
        self.exponent_bias = StatelessBuffer(
            torch.tensor(float(exponent_bias), device=device, dtype=dtype))
        self.fp_max_val = StatelessBuffer(
            max_float(self.exponent_bit_width(), self.mantissa_bit_width(), self.exponent_bias()))
        self.fp_internal_scale_min = StatelessBuffer(
            1. - self.exponent_bias() - self.mantissa_bit_width())
        if float_scaling_impl is None:
            float_scaling_impl = ConstScaling(1., device=device, dtype=dtype)
        if scaling_impl is None:
            scaling_impl = ConstScaling(1., device=device, dtype=dtype)
        # Zero-point is currently hardcoded to 0
        self.zero_point_impl = StatelessBuffer(torch.tensor(0., device=device, dtype=dtype))
        self.float_scaling_impl = float_scaling_impl
        self.scaling_impl = scaling_impl

    @brevitas.jit.script_method
    def internal_scale(self, x):
        internal_scale = floor_ste(torch.log2(torch.abs(x))) - self.mantissa_bit_width()
        internal_scale = torch.clamp_min(internal_scale, self.fp_internal_scale_min())
        internal_scale = torch.exp2(internal_scale)
        return internal_scale

    @brevitas.jit.script_method
    def quantize(self, x: torch.Tensor):
        scale = self.scaling_impl(x) / self.float_scaling_impl(x)
        scaled_x = x / scale
        internal_scale = self.internal_scale(scaled_x)
        val_fp_quant = internal_scale * self.float_to_int_impl(scaled_x / internal_scale)
        if self.signed:
            val_fp_quant = torch.clip(val_fp_quant, -1. * self.fp_max_val(), self.fp_max_val())
        else:
            val_fp_quant = torch.clip(val_fp_quant, 0., self.fp_max_val())
        return val_fp_quant, scale

    @brevitas.jit.script_method
    def dequantize(self, y, scale):
        return y * scale

    @brevitas.jit.script_method
    def forward(self, x):
        y, scale = self.quantize(x)
        y = self.dequantize(y, scale)
        # This is to respect the current interface of proxies
        return y, scale, self.zero_point_impl(), self.bit_width()
