# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
import math

import torch
from torch import Tensor
from torch.nn import Module

from brevitas.function.ops import max_int
from brevitas.function.ops import min_int

__all__ = ['BaseHandler', 'BitWidthHandlerMixin', 'ZeroPointHandlerMixin']


class BaseHandler(Module, ABC):

    def __init__(self) -> None:
        super().__init__()

    def attach_debug_info(self, module):
        pass

    @abstractmethod
    def prepare_for_export(self, module):
        pass


class QuantAxisMixin(ABC):

    @classmethod
    def quant_axis(cls, scale):
        for i, s in enumerate(scale.shape):
            if s != 1:
                return i
        return None


class ClipMixin(ABC):

    @classmethod
    def int_clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        # equality comparisons among power-of-2 floats are okay
        if narrow or bit_width != 8. and bit_width != 32.:
            if signed and (bit_width < 8. or narrow and bit_width <= 8.):
                dtype = torch.int8
            elif not signed and (bit_width < 8. or narrow and bit_width <= 8.):
                dtype = torch.uint8
            elif signed and (bit_width < 32. or narrow and bit_width <= 32.):
                dtype = torch.int32
            else:
                raise RuntimeError(
                    f"Sign {signed} and bit width {bit_width} not supported for export.")
            return {
                'min_val': min_int(signed, narrow, bit_width).to(dtype),
                'max_val': max_int(signed, narrow, bit_width).to(dtype)}
        else:
            return None

    @classmethod
    def float_clip_symbolic_kwargs(cls, narrow, signed, bit_width, scale, zero_point):
        symbolic_kwargs = cls.int_clip_symbolic_kwargs(narrow, signed, bit_width)
        if symbolic_kwargs is not None:
            symbolic_kwargs['min_val'] = (symbolic_kwargs['min_val'] - zero_point) * scale
            symbolic_kwargs['max_val'] = (symbolic_kwargs['max_val'] - zero_point) * scale
        return symbolic_kwargs


class BitWidthHandlerMixin(ABC):

    @classmethod
    def validate_bit_width(cls, bit_width: Tensor, reference: int, le_then=False):
        if bit_width is None:
            raise RuntimeError("Bit width cannot be None")
        if isinstance(bit_width, torch.Tensor):
            bit_width = bit_width.item()
        bit_width = int(bit_width)
        if bit_width > reference:
            raise RuntimeError(f"Bit width {bit_width} is not supported.")
        elif bit_width < reference and not le_then:
            raise RuntimeError(f"Bit width {bit_width} is not supported, should be {reference}b.")
        return bit_width

    @classmethod
    def validate_8b_bit_width(cls, bit_width: Tensor, le_then=False):
        return cls.validate_bit_width(bit_width, 8, le_then)

    @classmethod
    def validate_16b_bit_width(cls, bit_width: Tensor, le_then=False):
        return cls.validate_bit_width(bit_width, 16, le_then)

    @classmethod
    def validate_32b_bit_width(cls, bit_width: Tensor, le_then=False):
        return cls.validate_bit_width(bit_width, 32, le_then)


class ScaleHandlerMixin(ABC):

    @classmethod
    def validate_scalar_scale(cls, scale: Tensor):
        if scale is None:
            raise RuntimeError("Scale cannot be None.")
        if scale.view(-1).shape[0] != 1:
            raise RuntimeError("Only per-tensor scaling is supported.")
        return scale.item()

    @classmethod
    def validate_scalar_int_exponent(cls, scale: Tensor):
        cls.validate_scalar_scale(scale)
        exponent = math.log2(scale)
        if not exponent.is_integer():
            raise RuntimeError("Only power-of-two scale factors are supported.")
        exponent = int(exponent)
        return exponent

    @classmethod
    def validate_neg_scalar_int_exponent(cls, scale: Tensor):
        return -cls.validate_scalar_int_exponent(scale)


class ZeroPointHandlerMixin(ABC):

    @classmethod
    def zero_point_with_dtype(cls, signed, bit_width, zero_point):
        if not signed:
            if (zero_point < 0).any():
                raise RuntimeError("Zero points have to be positive under unsigned quantization")
            if bit_width > 8:
                raise RuntimeError("Unsigned zero-point with bit-width > 8 not supported.")
            return zero_point.type(torch.uint8)
        else:
            if bit_width <= 8:
                return zero_point.type(torch.int8)
            else:
                return zero_point.type(torch.int32)

    @classmethod
    def quant_input_zero_point(cls, module):
        signed = module.is_quant_input_signed
        zero_point = module.quant_input_zero_point()
        bit_width = module.quant_input_bit_width()
        return cls.zero_point_with_dtype(signed, bit_width, zero_point)

    @classmethod
    def quant_weight_zero_point(cls, module):
        signed = module.is_quant_weight_signed
        zero_point = module.quant_weight_zero_point()
        bit_width = module.quant_weight_bit_width()
        return cls.zero_point_with_dtype(signed, bit_width, zero_point)

    @classmethod
    def quant_output_zero_point(cls, module):
        signed = module.is_quant_output_signed
        zero_point = module.quant_output_zero_point()
        bit_width = module.quant_output_bit_width()
        return cls.zero_point_with_dtype(signed, bit_width, zero_point)
