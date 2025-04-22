# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Tuple

import torch
import torch.nn as nn

import brevitas
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.scaling import ConstScaling
from brevitas.core.utils import StatelessBuffer
from brevitas.function import compute_max_mantissa
from brevitas.utils.torch_utils import float_internal_scale


class FloatQuant(brevitas.jit.ScriptModule):
    __constants__ = ['signed', 'eps']

    def __init__(
            self,
            bit_width: int,
            signed: bool,
            exponent_bit_width: int,
            mantissa_bit_width: int,
            exponent_bias: int,
            float_clamp_impl: nn.Module,
            input_view_impl: nn.Module,
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

        self.mantissa_bit_width = StatelessBuffer(
            (torch.tensor(float(mantissa_bit_width), device=device, dtype=dtype)))
        self.exponent_bias = StatelessBuffer(
            torch.tensor(float(exponent_bias), device=device, dtype=dtype))

        self.fp_internal_scale_min = StatelessBuffer(
            1. - self.exponent_bias() - self.mantissa_bit_width())

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

        # This is more friendly for compile
        # TODO: This assumes fixed mantissa bit-width
        self.pre_compute_max_mantissa = StatelessBuffer(
            compute_max_mantissa(self.mantissa_bit_width()))

    @brevitas.jit.script_method
    def quantize(self, x: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_view_impl(x)
        scaled_x = x / scale
        internal_scale = float_internal_scale(
            scaled_x, self.mantissa_bit_width(), self.fp_internal_scale_min(), self.eps)
        val_fp_quant = internal_scale * self.float_to_int_impl(scaled_x / internal_scale)
        return val_fp_quant, scale

    @brevitas.jit.script_method
    def dequantize(self, y, scale):
        return y * scale

    @brevitas.jit.script_method
    def forward(self, x):
        if self.float_scaling_impl is not None:
            float_scaling_impl_value = self.float_scaling_impl(
                self.exponent_bit_width(), self.pre_compute_max_mantissa(), self.exponent_bias())
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
                y, self.exponent_bit_width(), self.pre_compute_max_mantissa(), self.exponent_bias())
            y = self.dequantize(y, scale)
        # This is to respect the current interface of proxies
        return y, scale, self.zero_point_impl(), self.exponent_bit_width(), self.mantissa_bit_width(), self.exponent_bias(), saturating, inf_values, nan_values
