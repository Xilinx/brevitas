# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from copy import copy

import torch
from torch import Tensor

from brevitas.export.common import to_0dim_if_scalar
from brevitas.export.common import to_item_if_0dim
from brevitas.proxy import ActQuantProxyFromInjector
from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.proxy import DecoupledWeightQuantProxyFromInjector
from brevitas.proxy import DecoupledWeightQuantWithInputProxyFromInjector
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector

from .base import BitWidthHandlerMixin
from .base import ClipMixin
from .base import QuantAxisMixin
from .base import ZeroPointHandlerMixin


class DQMixin(ABC):

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def dequantize_fn(self, x, scale, zero_point, axis):
        pass

    @property
    @abstractmethod
    def flatten_dequantize_params(self):
        pass

    @property
    @abstractmethod
    def itemize_quantize_scalar_params(self):
        pass

    def assert_ge_zero(self, *args):
        for a in args:
            bools = a >= 0.
            if isinstance(bools, torch.Tensor):
                bools = bools.all()
            assert bools


class CDQMixin(DQMixin, ABC):

    @abstractmethod
    def clip_fn(self, x, min_val, max_val):
        pass


class QMixin(BitWidthHandlerMixin, ABC):

    @classmethod
    @abstractmethod
    def uint8_dtype(cls):
        pass

    @classmethod
    @abstractmethod
    def int8_dtype(cls):
        pass

    @classmethod
    @abstractmethod
    def int32_dtype(cls):
        pass

    @abstractmethod
    def quantize_fn(self, x, scale, zero_point, dtype, axis):
        pass

    @classmethod
    def signed_dtype(cls, bit_width, is_signed):
        if bit_width is None:
            return None
        if is_signed and bit_width <= 8:
            dtype = cls.int8_dtype()
        elif not is_signed and bit_width <= 8:
            dtype = cls.uint8_dtype()
        elif is_signed and bit_width > 8:
            dtype = cls.int32_dtype()
        else:
            raise RuntimeError(
                "Unsigned quantization > 8b not supported for export, switch to signed.")
        return dtype


class CDQProxyHandlerMixin(QuantAxisMixin, ClipMixin, ZeroPointHandlerMixin, CDQMixin, ABC):

    def dequantize_symbolic_kwargs(cls, scale, zero_point, bit_width, is_signed):
        scale_orig_shape = scale.shape
        axis = cls.quant_axis(scale)
        if cls.flatten_dequantize_params:
            scale = scale.flatten()
        scale = to_0dim_if_scalar(scale)
        if cls.flatten_dequantize_params:
            zero_point = zero_point.flatten()
        zp = to_0dim_if_scalar(zero_point)
        zp = zp.expand_as(scale)
        zp = cls.zero_point_with_dtype(is_signed, bit_width, zp)
        return {
            'scale': scale,
            'zero_point': zp,
            'axis': axis,
            # We save only the scale original shape
            # as zero-point is being expanded to the same
            # size as the scale
            'scale_orig_shape': scale_orig_shape}


class QCDQWeightQuantProxyHandlerMixin(CDQProxyHandlerMixin, ABC):
    handled_layer = WeightQuantProxyFromInjector

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            int_weights = {
                tm.weight.data_ptr(): tm.quant_weight().int(float_datatype=False)
                for tm in module.tracked_module_list}
            # Get the first quant weight as representative
            quant_weight = module.tracked_module_list[0].quant_weight()
            self.symbolic_kwargs['int_weights'] = int_weights
            self.symbolic_kwargs['bit_width'] = quant_weight.bit_width
            self.symbolic_kwargs['clip_symbolic_kwargs'] = self.int_clip_symbolic_kwargs(
                module.is_narrow_range, module.is_signed, quant_weight.bit_width)
            self.symbolic_kwargs['dequantize_symbolic_kwargs'] = self.dequantize_symbolic_kwargs(
                quant_weight.scale,
                quant_weight.zero_point,
                quant_weight.bit_width,
                module.is_signed)
        else:
            self.symbolic_kwargs = None

    def symbolic_execution(self, x: Tensor):
        assert self.symbolic_kwargs is not None, 'Symbolic execution requires quant to be enabled'
        x = self.symbolic_kwargs['int_weights'][x.data_ptr()]
        clip_symbolic_kwargs = self.symbolic_kwargs['clip_symbolic_kwargs']
        # Copy dict to allow for popping kwargs even on shared quantizers
        dequantize_symbolic_kwargs = copy(self.symbolic_kwargs['dequantize_symbolic_kwargs'])
        scale = dequantize_symbolic_kwargs['scale']
        zero_point = dequantize_symbolic_kwargs['zero_point']
        bit_width = self.symbolic_kwargs['bit_width']
        scale_orig_shape = dequantize_symbolic_kwargs.pop('scale_orig_shape')
        # Workaround to trick the tracer into believing all return values are used
        self.assert_ge_zero(scale, zero_point, bit_width)
        if clip_symbolic_kwargs is not None:
            x = self.clip_fn(x, *clip_symbolic_kwargs.values())
        x = self.dequantize_fn(x, *dequantize_symbolic_kwargs.values())
        # Restore the original shapes to guarantee correct shape propagation downstream
        scale = scale.view(scale_orig_shape)
        zero_point = zero_point.view_as(scale)
        return x, scale, zero_point, bit_width


class QCDQDecoupledWeightQuantProxyHandlerMixin(QCDQWeightQuantProxyHandlerMixin, ABC):
    handled_layer = DecoupledWeightQuantProxyFromInjector

    def symbolic_execution(self, x: Tensor):
        out, scale, zero_point, bit_width = super().symbolic_execution(x)
        # Return post-rounding scale and zero-point in place of pre-rounding as a placeholder
        return out, scale, zero_point, scale, zero_point, bit_width


class QCDQDecoupledWeightQuantWithInputProxyHandlerMixin(QCDQDecoupledWeightQuantProxyHandlerMixin,
                                                         ABC):
    handled_layer = DecoupledWeightQuantWithInputProxyFromInjector

    def symbolic_execution(self, x: Tensor, input_bit_width: torch.Tensor, input_is_signed: bool):
        return super().symbolic_execution(x)


class QCDQActQuantProxyHandlerMixin(QMixin, CDQProxyHandlerMixin, ABC):
    handled_layer = ActQuantProxyFromInjector

    def quantize_symbolic_kwargs(cls, scale, zero_point, bit_width, is_signed):
        # compute axis before redefining scale
        axis = cls.quant_axis(scale)
        scale = to_0dim_if_scalar(scale.flatten())
        zp = to_0dim_if_scalar(zero_point.flatten())
        # expand_as must go after 0-dim check
        zp = zp.expand_as(scale)
        zp = cls.zero_point_with_dtype(is_signed, bit_width, zp)
        if cls.itemize_quantize_scalar_params:
            scale = to_item_if_0dim(scale)
            zp = to_item_if_0dim(zp)
        dtype = cls.signed_dtype(bit_width, is_signed)
        return {'scale': scale, 'zero_point': zp, 'dtype': dtype, 'axis': axis}

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            self.symbolic_kwargs['bit_width'] = module.bit_width()
            self.symbolic_kwargs['quantize_symbolic_kwargs'] = self.quantize_symbolic_kwargs(
                module.scale(), module.zero_point(), module.bit_width(), module.is_signed)
            self.symbolic_kwargs['dequantize_symbolic_kwargs'] = self.dequantize_symbolic_kwargs(
                module.scale(), module.zero_point(), module.bit_width(), module.is_signed)
            self.symbolic_kwargs['clip_symbolic_kwargs'] = self.int_clip_symbolic_kwargs(
                module.is_narrow_range, module.is_signed, module.bit_width())
        else:
            self.symbolic_kwargs = None

    def symbolic_execution(self, x: Tensor):
        assert self.symbolic_kwargs is not None, 'Symbolic execution requires quant to be enabled'
        quantize_symbolic_kwargs = self.symbolic_kwargs['quantize_symbolic_kwargs']
        clip_symbolic_kwargs = self.symbolic_kwargs['clip_symbolic_kwargs']
        # Copy dict to allow for popping kwargs even on shared quantizers
        dequantize_symbolic_kwargs = copy(self.symbolic_kwargs['dequantize_symbolic_kwargs'])
        scale = dequantize_symbolic_kwargs['scale']
        zero_point = dequantize_symbolic_kwargs['zero_point']
        scale_orig_shape = dequantize_symbolic_kwargs.pop('scale_orig_shape')
        bit_width = self.symbolic_kwargs['bit_width']
        # Workaround to trick the tracer into believing all return values are used
        self.assert_ge_zero(scale, zero_point, bit_width)
        x = self.quantize_fn(x, *quantize_symbolic_kwargs.values())
        if clip_symbolic_kwargs is not None:
            x = self.clip_fn(x, *clip_symbolic_kwargs.values())
        x = self.dequantize_fn(x, *dequantize_symbolic_kwargs.values())
        # Restore the original shapes to guarantee correct shape propagation downstream
        scale = scale.view(scale_orig_shape)
        zero_point = zero_point.view_as(scale)
        return x, scale, zero_point, bit_width


class QCDQBiasQuantProxyHandlerMixin(DQMixin, QuantAxisMixin, ZeroPointHandlerMixin, ABC):
    handled_layer = BiasQuantProxyFromInjector

    def validate(self, module):
        if module.bit_width() is not None:
            assert module.bit_width() > 1., 'Binary quant not supported'
        assert module.is_signed, 'Unsigned bias not supported.'
        assert module.rounding_mode == 'ROUND', 'Only round to nearest even supported.'

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            int_biases = {
                tm.bias.data_ptr(): tm.quant_bias().int(float_datatype=False)
                for tm in module.tracked_module_list}
            self.symbolic_kwargs = {
                'int_biases': int_biases,
                'scale': module.scale(),
                'zero_point': module.zero_point(),
                'bit_width': module.bit_width()}
        else:
            self.symbolic_kwargs = None

    def symbolic_execution(self, x: Tensor, input_scale=None, input_bit_width=None):
        assert self.symbolic_kwargs is not None, 'Symbolic execution requires quant to be enabled'
        int_bias = self.symbolic_kwargs['int_biases'][x.data_ptr()]
        scale = self.symbolic_kwargs['scale']
        bit_width = self.symbolic_kwargs['bit_width']
        zero_point = self.symbolic_kwargs['zero_point']
        assert scale is not None or input_scale is not None, 'Input scale required for bias export'
        assert bit_width is not None or input_bit_width is not None, 'Input bit width required for bias export'
        if input_scale is not None:
            scale = input_scale
        scale_orig_shape = scale.shape
        if input_bit_width is not None:
            bit_width = input_bit_width
        quant_axis = self.quant_axis(scale)
        if self.flatten_dequantize_params:
            scale = scale.flatten()
            zero_point = zero_point.flatten()
        scale = to_0dim_if_scalar(scale)
        zero_point = to_0dim_if_scalar(zero_point).expand_as(scale)
        zero_point = self.zero_point_with_dtype(
            True, bit_width, zero_point)  # assume signed is True
        y = self.dequantize_fn(int_bias, scale, zero_point, quant_axis)
        # Restore the original shapes to guarantee correct shape propagation downstream
        scale = scale.view(scale_orig_shape)
        zero_point = zero_point.view_as(scale)
        return y, scale, zero_point, bit_width


class QCDQTruncQuantProxyHandlerMixin(QuantAxisMixin,
                                      ClipMixin,
                                      ZeroPointHandlerMixin,
                                      QMixin,
                                      CDQMixin,
                                      ABC):
    handled_layer = TruncQuantProxyFromInjector

    def prepare_for_export(self, module: TruncQuantProxyFromInjector):
        if module.is_quant_enabled:
            self.validate(module)
            self.symbolic_kwargs = {'output_bit_width': module.bit_width()}

    def symbolic_execution(
            self, x: Tensor, scale: Tensor, zero_point: Tensor, input_bit_width: Tensor,
            signed: Tensor):
        assert self.symbolic_kwargs is not None, 'Symbolic execution requires quant to be enabled'
        output_bit_width = self.symbolic_kwargs['output_bit_width']
        dtype = self.int8_dtype() if signed else self.uint8_dtype()
        trunc_scale = 2.0 ** (input_bit_width - output_bit_width)
        pre_scale = scale * trunc_scale
        flat_pre_scale = to_0dim_if_scalar(pre_scale.flatten())
        flat_scale = to_0dim_if_scalar(scale.flatten())
        zp = to_0dim_if_scalar(zero_point.flatten()).expand_as(flat_scale)
        x = self.quantize_fn(x, flat_pre_scale, zp, dtype, self.quant_axis(pre_scale))
        clip_symbolic_kwargs = self.int_clip_symbolic_kwargs(
            signed=signed, narrow=False, bit_width=output_bit_width)
        if clip_symbolic_kwargs is not None:
            x = self.clip_fn(x, *clip_symbolic_kwargs.values())
        x = self.dequantize_fn(x, flat_scale, zp, self.quant_axis(scale))
        return x, scale, zero_point, output_bit_width
