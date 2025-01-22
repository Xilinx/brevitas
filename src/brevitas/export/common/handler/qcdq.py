# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from copy import copy

import torch
from torch import Tensor

from brevitas.export.common import to_0dim_if_scalar
from brevitas.export.common import to_item_if_0dim
from brevitas.proxy import ActFloatQuantProxyFromInjector
from brevitas.proxy import ActQuantProxyFromInjector
from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.proxy import DecoupledWeightQuantProxyFromInjector
from brevitas.proxy import DecoupledWeightQuantWithInputProxyFromInjector
from brevitas.proxy import WeightFloatQuantProxyFromInjector
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.proxy.float_parameter_quant import BiasFloatQuantProxyFromInjector
from brevitas.proxy.runtime_quant import DynamicActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector

from .base import BitWidthHandlerMixin
from .base import ClipMixin
from .base import FloatClipMixin
from .base import FloatZeroPointHandlerMixin
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


class DQCastMixin(DQMixin, ABC):

    @abstractmethod
    def cast_fn(self, x, dtype):
        pass


class CDQCastMixin(DQCastMixin, ABC):

    @abstractmethod
    def clip_fn(self, x, min_val, max_val):
        pass


class FloatQMixin(ABC):

    @abstractmethod
    def quantize_fn(self, x, scale, zero_point, dtype, axis):
        pass

    @classmethod
    def signed_dtype(cls, exponent_bit_width, mantissa_bit_width, is_ocp, is_fnuz):
        if exponent_bit_width is None or mantissa_bit_width is None:
            return None
        if is_ocp:
            if exponent_bit_width == 4 and mantissa_bit_width == 3:
                dtype = torch.float8_e4m3fn
            elif exponent_bit_width == 5 and mantissa_bit_width == 2:
                dtype = torch.float8_e5m2
        elif is_fnuz:
            if exponent_bit_width == 4 and mantissa_bit_width == 3:
                dtype = torch.float8_e4m3fnuz
            elif exponent_bit_width == 5 and mantissa_bit_width == 2:
                dtype = torch.float8_e5m2fnuz
        else:
            raise NotImplementedError
        return dtype


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


class DynamicQMixin(QMixin, ABC):

    @abstractmethod
    def quantize_fn(self, x, dtype):
        pass


class FloatCDQCastProxyHandlerMixin(QuantAxisMixin,
                                    FloatClipMixin,
                                    FloatZeroPointHandlerMixin,
                                    CDQCastMixin,
                                    ABC):

    def dequantize_symbolic_kwargs(
            cls, scale, zero_point, exponent_bit_width, mantissa_bit_width, is_ocp, is_fnuz):
        scale_orig_shape = scale.shape
        axis = cls.quant_axis(scale)
        if cls.flatten_dequantize_params:
            scale = scale.flatten()
        scale = to_0dim_if_scalar(scale)
        if cls.flatten_dequantize_params:
            zero_point = zero_point.flatten()
        zp = to_0dim_if_scalar(zero_point)
        zp = zp.expand_as(scale)
        zp = cls.zero_point_with_dtype(exponent_bit_width, mantissa_bit_width, is_ocp, is_fnuz, zp)
        return {
            'scale': scale,
            'zero_point': zp,
            'axis': axis,
            # We save only the scale original shape
            # as zero-point is being expanded to the same
            # size as the scale
            'scale_orig_shape': scale_orig_shape}


class CDQCastProxyHandlerMixin(QuantAxisMixin, ClipMixin, ZeroPointHandlerMixin, CDQCastMixin, ABC):

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


class FloatQCDQCastWeightQuantProxyHandlerMixin(FloatQMixin, FloatCDQCastProxyHandlerMixin):
    handled_layer = WeightFloatQuantProxyFromInjector

    def quantize_symbolic_kwargs(
            cls, scale, zero_point, exponent_bit_width, mantissa_bit_width, is_ocp, is_fnuz):
        # compute axis before redefining scale
        axis = cls.quant_axis(scale)
        scale = to_0dim_if_scalar(scale.flatten())
        zp = to_0dim_if_scalar(zero_point.flatten())
        # expand_as must go after 0-dim check
        zp = zp.expand_as(scale)
        zp = cls.zero_point_with_dtype(exponent_bit_width, mantissa_bit_width, is_ocp, is_fnuz, zp)
        if cls.itemize_quantize_scalar_params:
            scale = to_item_if_0dim(scale)
            zp = to_item_if_0dim(zp)
        dtype = cls.signed_dtype(exponent_bit_width, mantissa_bit_width, is_ocp, is_fnuz)
        return {'scale': scale, 'zero_point': zp, 'dtype': dtype, 'axis': axis}

    def prepare_quantize_from_floating_point(self, module):
        quant_weight = module.tracked_module_list[0].quant_weight()
        scale = quant_weight.scale
        self.scale_dtype = scale.dtype
        if self.scale_dtype == torch.bfloat16 or self.scale_dtype == torch.float16:
            scale = self.cast_fn(scale, torch.float32)
        self.symbolic_kwargs['quantize_symbolic_kwargs'] = self.quantize_symbolic_kwargs(
            scale,
            quant_weight.zero_point,
            quant_weight.exponent_bit_width,
            quant_weight.mantissa_bit_width,
            module.is_ocp,
            module.is_fnuz)

    def prepare_quantize_from_minifloat(self, module):
        raise NotImplementedError

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            if self._export_q_node:
                self.prepare_quantize_from_floating_point(module)
            else:
                self.prepare_quantize_from_minifloat(module)
            # Get the first quant weight as representative
            quant_weight = module.tracked_module_list[0].quant_weight()

            # (B)float16 is not supported with standard Q/DQ ops, thus we store the original dtype
            # of the scale and we cast it to float32.
            # The original dtype is then restored during the forward pass
            scale = quant_weight.scale
            self.scale_dtype = scale.dtype
            if self.scale_dtype == torch.bfloat16 or self.scale_dtype == torch.float16:
                scale = self.cast_fn(scale, torch.float32)

            self.symbolic_kwargs['exponent_bit_width'] = quant_weight.exponent_bit_width
            self.symbolic_kwargs['mantissa_bit_width'] = quant_weight.mantissa_bit_width
            self.symbolic_kwargs['exponent_bias'] = quant_weight.exponent_bias
            self.symbolic_kwargs['saturating'] = quant_weight.saturating
            self.symbolic_kwargs['inf_values'] = quant_weight.inf_values
            self.symbolic_kwargs['nan_values'] = quant_weight.nan_values
            self.symbolic_kwargs['clip_symbolic_kwargs'] = self.clip_symbolic_kwargs(
                module.is_narrow_range,
                module.is_signed,
                quant_weight.exponent_bit_width,
                quant_weight.mantissa_bit_width)
            self.symbolic_kwargs['dequantize_symbolic_kwargs'] = self.dequantize_symbolic_kwargs(
                scale,
                quant_weight.zero_point,
                quant_weight.exponent_bit_width,
                quant_weight.mantissa_bit_width,
                module.is_ocp,
                module.is_fnuz)
        else:
            self.symbolic_kwargs = None

    def quantize_from_floating_point(self, x: Tensor):
        # Workaround for equal_cpu RuntimeError
        quantize_symbolic_kwargs = self.symbolic_kwargs['quantize_symbolic_kwargs']
        # Before quantization, cast input to float32
        if self.scale_dtype == torch.float16 or self.scale_dtype == torch.bfloat16:
            x = self.cast_fn(x, torch.float32)
        x = self.quantize_fn(x, *quantize_symbolic_kwargs.values())
        return x

    def quantize_from_minifloat(self, x: Tensor):
        raise NotImplementedError

    def symbolic_execution(self, x: Tensor):
        assert self.symbolic_kwargs is not None, 'Symbolic execution requires quant to be enabled'

        # Copy dict to allow for popping kwargs even on shared quantizers
        dequantize_symbolic_kwargs = copy(self.symbolic_kwargs['dequantize_symbolic_kwargs'])
        scale = dequantize_symbolic_kwargs['scale']
        zero_point = dequantize_symbolic_kwargs['zero_point']

        if self._export_q_node:
            x = self.quantize_from_floating_point(x)
        else:
            x = self.quantize_from_minifloat(x)
        clip_symbolic_kwargs = self.symbolic_kwargs['clip_symbolic_kwargs']
        exponent_bit_width = self.symbolic_kwargs['exponent_bit_width']
        mantissa_bit_width = self.symbolic_kwargs['mantissa_bit_width']
        exponent_bias = self.symbolic_kwargs['exponent_bias']
        saturating = self.symbolic_kwargs['saturating']
        inf_values = self.symbolic_kwargs['inf_values']
        nan_values = self.symbolic_kwargs['nan_values']
        scale_orig_shape = dequantize_symbolic_kwargs.pop('scale_orig_shape')
        # Workaround to trick the tracer into believing all return values are used
        self.assert_ge_zero(scale, exponent_bit_width, mantissa_bit_width, exponent_bias)
        if clip_symbolic_kwargs is not None:
            x = self.clip_fn(x, *clip_symbolic_kwargs.values())
        x = self.dequantize_fn(x, *dequantize_symbolic_kwargs.values())
        # After dequantization, cast both input and scale to the correct dtype
        if self.scale_dtype == torch.float16 or self.scale_dtype == torch.bfloat16:
            x = self.cast_fn(x, self.scale_dtype)
            scale = self.cast_fn(scale, self.scale_dtype)
        # Restore the original shapes to guarantee correct shape propagation downstream
        scale = scale.view(scale_orig_shape)
        zero_point = zero_point.view_as(scale)
        return x, scale, zero_point, exponent_bit_width, mantissa_bit_width, exponent_bias, saturating, inf_values, nan_values


class QCDQCastWeightQuantProxyHandlerMixin(QMixin, CDQCastProxyHandlerMixin):
    handled_layer = WeightQuantProxyFromInjector

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

    def prepare_quantize_from_floating_point(self, module):
        quant_weight = module.tracked_module_list[0].quant_weight()
        scale = quant_weight.scale
        self.scale_dtype = scale.dtype
        if self.scale_dtype == torch.bfloat16 or self.scale_dtype == torch.float16:
            scale = self.cast_fn(scale, torch.float32)
        self.symbolic_kwargs['quantize_symbolic_kwargs'] = self.quantize_symbolic_kwargs(
            scale, quant_weight.zero_point, quant_weight.bit_width, module.is_signed)

    def prepare_quantize_from_integer(self, module):
        int_weights = {
            tm.weight.data_ptr(): tm.quant_weight().int(float_datatype=False)
            for tm in module.tracked_module_list}
        self.symbolic_kwargs['int_weights'] = int_weights

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            if self._export_q_node:
                self.prepare_quantize_from_floating_point(module)
            else:
                self.prepare_quantize_from_integer(module)
            # Get the first quant weight as representative
            quant_weight = module.tracked_module_list[0].quant_weight()

            # (B)float16 is not supported with standard Q/DQ ops, thus we store the original dtype
            # of the scale and we cast it to float32.
            # The original dtype is then restored during the forward pass
            scale = quant_weight.scale
            self.scale_dtype = scale.dtype
            if self.scale_dtype == torch.bfloat16 or self.scale_dtype == torch.float16:
                scale = self.cast_fn(scale, torch.float32)

            self.symbolic_kwargs['bit_width'] = quant_weight.bit_width
            self.symbolic_kwargs['clip_symbolic_kwargs'] = self.int_clip_symbolic_kwargs(
                module.is_narrow_range, module.is_signed, quant_weight.bit_width)
            self.symbolic_kwargs['dequantize_symbolic_kwargs'] = self.dequantize_symbolic_kwargs(
                scale, quant_weight.zero_point, quant_weight.bit_width, module.is_signed)
        else:
            self.symbolic_kwargs = None

    def quantize_from_floating_point(self, x: Tensor):
        quantize_symbolic_kwargs = self.symbolic_kwargs['quantize_symbolic_kwargs']
        # Before quantization, cast input to float32
        if self.scale_dtype == torch.float16 or self.scale_dtype == torch.bfloat16:
            x = self.cast_fn(x, torch.float32)
        x = self.quantize_fn(x, *quantize_symbolic_kwargs.values())
        return x

    def quantize_from_integer(self, x: Tensor):
        return self.symbolic_kwargs['int_weights'][x.data_ptr()]

    def symbolic_execution(self, x: Tensor):
        assert self.symbolic_kwargs is not None, 'Symbolic execution requires quant to be enabled'
        if self._export_q_node:
            x = self.quantize_from_floating_point(x)
        else:
            x = self.quantize_from_integer(x)
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
        # After dequantization, cast both input and scale to the correct dtype
        if self.scale_dtype == torch.float16 or self.scale_dtype == torch.bfloat16:
            x = self.cast_fn(x, self.scale_dtype)
            scale = self.cast_fn(scale, self.scale_dtype)
        # Restore the original shapes to guarantee correct shape propagation downstream
        scale = scale.view(scale_orig_shape)
        zero_point = zero_point.view_as(scale)
        return x, scale, zero_point, bit_width


class QCDQCastDecoupledWeightQuantProxyHandlerMixin(QCDQCastWeightQuantProxyHandlerMixin, ABC):
    handled_layer = DecoupledWeightQuantProxyFromInjector

    def symbolic_execution(self, x: Tensor):
        out, scale, zero_point, bit_width = super().symbolic_execution(x)
        # Return post-rounding scale and zero-point in place of pre-rounding as a placeholder
        # The order of arguments must match the order in the forward method of DecoupledRescalingIntQuant
        return out, scale, zero_point, bit_width, scale, zero_point


class QCDQCastDecoupledWeightQuantWithInputProxyHandlerMixin(
        QCDQCastDecoupledWeightQuantProxyHandlerMixin, ABC):
    handled_layer = DecoupledWeightQuantWithInputProxyFromInjector

    def validate(self, module):
        assert not self._export_q_node, "This proxy requires to export integer weights"

    def symbolic_execution(self, x: Tensor, input_bit_width: torch.Tensor, input_is_signed: bool):
        return super().symbolic_execution(x)


class FloatQCDQCastActQuantProxyHandlerMixin(FloatQMixin, FloatCDQCastProxyHandlerMixin, ABC):
    handled_layer = ActFloatQuantProxyFromInjector

    def quantize_symbolic_kwargs(
            cls, scale, zero_point, exponent_bit_width, mantissa_bit_width, is_ocp, is_fnuz):
        # compute axis before redefining scale
        axis = cls.quant_axis(scale)
        scale = to_0dim_if_scalar(scale.flatten())
        zp = to_0dim_if_scalar(zero_point.flatten())
        # expand_as must go after 0-dim check
        zp = zp.expand_as(scale)
        zp = cls.zero_point_with_dtype(exponent_bit_width, mantissa_bit_width, is_ocp, is_fnuz, zp)
        if cls.itemize_quantize_scalar_params:
            scale = to_item_if_0dim(scale)
            zp = to_item_if_0dim(zp)
        dtype = cls.signed_dtype(exponent_bit_width, mantissa_bit_width, is_ocp, is_fnuz)
        return {'scale': scale, 'zero_point': zp, 'dtype': dtype, 'axis': axis}

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            self.symbolic_kwargs['exponent_bit_width'] = module.exponent_bit_width()
            self.symbolic_kwargs['mantissa_bit_width'] = module.mantissa_bit_width()
            self.symbolic_kwargs['exponent_bias'] = module.exponent_bias()
            self.symbolic_kwargs['saturating'] = module.is_saturating()
            self.symbolic_kwargs['inf_values'] = module.inf_values()
            self.symbolic_kwargs['nan_values'] = module.nan_values()

            # (B)float16 is not supported with standard Q/DQ ops, thus we store the original dtype
            # of the scale and we cast it to float32.
            # The original dtype is then restored during the forward pass
            scale = module.scale()
            self.scale_dtype = scale.dtype
            if self.scale_dtype == torch.bfloat16 or self.scale_dtype == torch.float16:
                scale = self.cast_fn(scale, torch.float32)

            self.symbolic_kwargs['quantize_symbolic_kwargs'] = self.quantize_symbolic_kwargs(
                scale,
                module.zero_point(),
                module.exponent_bit_width(),
                module.mantissa_bit_width(),
                module.is_ocp,
                module.is_fnuz)
            self.symbolic_kwargs['dequantize_symbolic_kwargs'] = self.dequantize_symbolic_kwargs(
                scale,
                module.zero_point(),
                module.exponent_bit_width(),
                module.mantissa_bit_width(),
                module.is_ocp,
                module.is_fnuz)
            self.symbolic_kwargs['clip_symbolic_kwargs'] = self.clip_symbolic_kwargs(
                module.is_narrow_range,
                module.is_signed,
                module.exponent_bit_width(),
                module.mantissa_bit_width())

        else:
            self.symbolic_kwargs = None

    def symbolic_execution(self, x: Tensor):
        assert self.symbolic_kwargs is not None, 'Symbolic execution requires quant to be enabled'

        # Copy dict to allow for popping kwargs even on shared quantizers
        dequantize_symbolic_kwargs = copy(self.symbolic_kwargs['dequantize_symbolic_kwargs'])
        scale = dequantize_symbolic_kwargs['scale']
        zero_point = dequantize_symbolic_kwargs['zero_point']
        scale_orig_shape = dequantize_symbolic_kwargs.pop('scale_orig_shape')

        quantize_symbolic_kwargs = self.symbolic_kwargs['quantize_symbolic_kwargs']
        clip_symbolic_kwargs = self.symbolic_kwargs['clip_symbolic_kwargs']
        exponent_bit_width = self.symbolic_kwargs['exponent_bit_width']
        mantissa_bit_width = self.symbolic_kwargs['mantissa_bit_width']
        exponent_bias = self.symbolic_kwargs['exponent_bias']
        saturating = self.symbolic_kwargs['saturating']
        inf_values = self.symbolic_kwargs['inf_values']
        nan_values = self.symbolic_kwargs['nan_values']

        self.assert_ge_zero(scale, exponent_bit_width, mantissa_bit_width, exponent_bias)
        # If original dtype of the input is (b)float16, cast the input to float32
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = self.cast_fn(x, torch.float32)
        x = self.quantize_fn(x, *quantize_symbolic_kwargs.values())
        if clip_symbolic_kwargs is not None:
            x = self.clip_fn(x, *clip_symbolic_kwargs.values())
        x = self.dequantize_fn(x, *dequantize_symbolic_kwargs.values())
        # After dequantization, cast both output and scale to the correct dtype
        if self.scale_dtype == torch.float16 or self.scale_dtype == torch.bfloat16:
            x = self.cast_fn(x, self.scale_dtype)
            scale = self.cast_fn(scale, self.scale_dtype)
        # Restore the original shapes to guarantee correct shape propagation downstream
        scale = scale.view(scale_orig_shape)
        zero_point = zero_point.view_as(scale)
        return x, scale, zero_point, exponent_bit_width, mantissa_bit_width, exponent_bias, saturating, inf_values, nan_values


class QCDQCastActQuantProxyHandlerMixin(QMixin, CDQCastProxyHandlerMixin, ABC):
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

            # (B)float16 is not supported with standard Q/DQ ops, thus we store the original dtype
            # of the scale and we cast it to float32.
            # The original dtype is then restored during the forward pass
            scale = module.scale()
            self.scale_dtype = scale.dtype
            if self.scale_dtype == torch.bfloat16 or self.scale_dtype == torch.float16:
                scale = self.cast_fn(scale, torch.float32)

            self.symbolic_kwargs['quantize_symbolic_kwargs'] = self.quantize_symbolic_kwargs(
                scale, module.zero_point(), module.bit_width(), module.is_signed)
            self.symbolic_kwargs['dequantize_symbolic_kwargs'] = self.dequantize_symbolic_kwargs(
                scale, module.zero_point(), module.bit_width(), module.is_signed)
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
        # If original dtype of the input is (b)float16, cast the input to float32
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = self.cast_fn(x, torch.float32)
        x = self.quantize_fn(x, *quantize_symbolic_kwargs.values())
        if clip_symbolic_kwargs is not None:
            x = self.clip_fn(x, *clip_symbolic_kwargs.values())
        x = self.dequantize_fn(x, *dequantize_symbolic_kwargs.values())
        # After dequantization, cast both output and scale to the correct dtype
        if self.scale_dtype == torch.float16 or self.scale_dtype == torch.bfloat16:
            x = self.cast_fn(x, self.scale_dtype)
            scale = self.cast_fn(scale, self.scale_dtype)
        # Restore the original shapes to guarantee correct shape propagation downstream
        scale = scale.view(scale_orig_shape)
        zero_point = zero_point.view_as(scale)
        return x, scale, zero_point, bit_width


class DynamicQDQCastActQuantProxyHandlerMixin(DynamicQMixin, DQCastMixin, ABC):
    handled_layer = DynamicActQuantProxyFromInjector

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            bit_width = module.bit_width()
            is_signed = module.is_signed
            dtype = self.signed_dtype(bit_width, is_signed)
            self.symbolic_kwargs['bit_width'] = bit_width
            self.symbolic_kwargs['is_signed'] = is_signed
            self.symbolic_kwargs['dtype'] = dtype
        else:
            self.symbolic_kwargs = None

    def symbolic_execution(self, x: Tensor):
        assert self.symbolic_kwargs is not None, 'Symbolic execution requires quant to be enabled'

        bit_width = self.symbolic_kwargs['bit_width']
        int_dtype = self.symbolic_kwargs['dtype']
        # Workaround to trick the tracer into believing all return values are used
        self.assert_ge_zero(bit_width)
        # If original dtype of the input is (b)float16, cast the input to float32
        x_dtype = x.dtype
        if x_dtype == torch.float16 or x_dtype == torch.bfloat16:
            x = self.cast_fn(x, torch.float32)

        x, scale, zero_point = self.quantize_fn(x, int_dtype)

        x = self.dequantize_fn(x, scale, zero_point, None)
        # After dequantization, cast both output and scale to the correct dtype
        if x_dtype == torch.float16 or x_dtype == torch.bfloat16:
            x = self.cast_fn(x, x_dtype)
            scale = self.cast_fn(scale, x_dtype)
        return x, scale, zero_point, bit_width


class FloatCDQCastBiasQuantProxyHandlerMixin(DQCastMixin,
                                             QuantAxisMixin,
                                             FloatZeroPointHandlerMixin,
                                             ABC):
    # TODO: We do not have any bias quantizer so this is not wired to anything.
    # Currently we do not support Minifloat -> DQ export for minifloat.
    # This has to be rewritten to be QDQ
    handled_layer = BiasFloatQuantProxyFromInjector

    def validate(self, module):
        if module.bit_width() is not None:
            assert module.bit_width() > 1., 'Binary quant not supported'
        assert module.is_signed, 'Unsigned bias not supported.'
        assert module.rounding_mode == 'ROUND', 'Only round to nearest even supported.'

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            int_biases = {
                tm.bias.data_ptr(): tm.quant_bias().minifloat(float_datatype=False)
                for tm in module.tracked_module_list}
            self.symbolic_kwargs = {
                'int_biases': int_biases,
                'scale': module.scale(),
                'zero_point': module.zero_point(),
                'exponent_bit_width': module.exponent_bit_width(),
                'mantissa_bit_width': module.mantissa_bit_width(),
                'exponent_bias': module.exponent_bias(),
                'saturating': module.is_saturating(),
                'inf_values': module.inf_values(),
                'nan_values': module.nan_values()}

        else:
            self.symbolic_kwargs = None

    def symbolic_execution(self, x: Tensor, input_scale=None):
        assert self.symbolic_kwargs is not None, 'Symbolic execution requires quant to be enabled'
        int_bias = self.symbolic_kwargs['int_biases'][x.data_ptr()]
        scale = self.symbolic_kwargs['scale']
        zero_point = self.symbolic_kwargs['zero_point']
        exponent_bit_width = self.symbolic_kwargs['exponent_bit_width']
        mantissa_bit_width = self.symbolic_kwargs['mantissa_bit_width']
        exponent_bias = self.symbolic_kwargs['exponent_bias']
        saturating = self.symbolic_kwargs['saturating']
        inf_values = self.symbolic_kwargs['inf_values']
        nan_values = self.symbolic_kwargs['nan_values']

        assert scale is not None or input_scale is not None, 'Input scale required for bias export'
        if input_scale is not None:
            scale = input_scale
        scale_orig_shape = scale.shape

        quant_axis = self.quant_axis(scale)
        if self.flatten_dequantize_params:
            scale = scale.flatten()
            zero_point = zero_point.flatten()
        scale = to_0dim_if_scalar(scale)
        zero_point = to_0dim_if_scalar(zero_point).expand_as(scale)
        zero_point = self.zero_point_with_dtype(
            True, exponent_bit_width, mantissa_bit_width, zero_point)  # assume signed is True
        # If original dtype of scale is (b)float16, store the original dtype
        # and cast the scale to float32
        scale_dtype = scale.dtype
        if scale_dtype == torch.float16 or scale_dtype == torch.bfloat16:
            scale = self.cast_fn(scale, torch.float32)
        y = self.dequantize_fn(int_bias, scale, zero_point, quant_axis)
        # After dequantization, cast both output and scale to the correct dtype
        if scale_dtype == torch.float16 or scale_dtype == torch.bfloat16:
            y = self.cast_fn(y, scale_dtype)
            scale = self.cast_fn(scale, scale_dtype)
        # Restore the original shapes to guarantee correct shape propagation downstream
        scale = scale.view(scale_orig_shape)
        zero_point = zero_point.view_as(scale)
        return y, scale, zero_point, exponent_bit_width, mantissa_bit_width, exponent_bias, saturating, inf_values, nan_values


class CDQCastBiasQuantProxyHandlerMixin(DQCastMixin, QuantAxisMixin, ZeroPointHandlerMixin, ABC):
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

    def symbolic_execution(self, x: Tensor, input_scale=None):
        assert self.symbolic_kwargs is not None, 'Symbolic execution requires quant to be enabled'
        int_bias = self.symbolic_kwargs['int_biases'][x.data_ptr()]
        scale = self.symbolic_kwargs['scale']
        bit_width = self.symbolic_kwargs['bit_width']
        zero_point = self.symbolic_kwargs['zero_point']
        assert scale is not None or input_scale is not None, 'Input scale required for bias export'
        if input_scale is not None:
            scale = input_scale
        scale_orig_shape = scale.shape

        quant_axis = self.quant_axis(scale)
        if self.flatten_dequantize_params:
            scale = scale.flatten()
            zero_point = zero_point.flatten()
        scale = to_0dim_if_scalar(scale)
        zero_point = to_0dim_if_scalar(zero_point).expand_as(scale)
        zero_point = self.zero_point_with_dtype(
            True, bit_width, zero_point)  # assume signed is True
        # If original dtype of scale is (b)float16, store the original dtype
        # and cast the scale to float32
        scale_dtype = scale.dtype
        if scale_dtype == torch.float16 or scale_dtype == torch.bfloat16:
            scale = self.cast_fn(scale, torch.float32)
        y = self.dequantize_fn(int_bias, scale, zero_point, quant_axis)
        # After dequantization, cast both output and scale to the correct dtype
        if scale_dtype == torch.float16 or scale_dtype == torch.bfloat16:
            y = self.cast_fn(y, scale_dtype)
            scale = self.cast_fn(scale, scale_dtype)
        # Restore the original shapes to guarantee correct shape propagation downstream
        scale = scale.view(scale_orig_shape)
        zero_point = zero_point.view_as(scale)
        return y, scale, zero_point, bit_width


class QCDQCastTruncQuantProxyHandlerMixin(QuantAxisMixin,
                                          ClipMixin,
                                          ZeroPointHandlerMixin,
                                          QMixin,
                                          CDQCastMixin,
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
        # If original dtype of scale is (b)float16, store the original scale dtype
        # and cast the scale and the input to float32
        scale_dtype = scale.dtype
        if scale_dtype == torch.bfloat16 or scale_dtype == torch.float16:
            scale = self.cast_fn(scale, torch.float32)
        if x.dtype == torch.bfloat16 or x.dtype == torch.float16:
            x = self.cast_fn(x, torch.float32)
        pre_scale = scale * trunc_scale
        flat_pre_scale = to_0dim_if_scalar(pre_scale.flatten())
        flat_scale = to_0dim_if_scalar(scale.flatten())
        zp = to_0dim_if_scalar(zero_point.flatten()).expand_as(flat_scale)
        zp = self.zero_point_with_dtype(signed, output_bit_width, zp)
        x = self.quantize_fn(x, flat_pre_scale, zp, dtype, self.quant_axis(pre_scale))
        clip_symbolic_kwargs = self.int_clip_symbolic_kwargs(
            signed=signed, narrow=False, bit_width=output_bit_width)
        if clip_symbolic_kwargs is not None:
            x = self.clip_fn(x, *clip_symbolic_kwargs.values())
        x = self.dequantize_fn(x, flat_pre_scale, zp, self.quant_axis(scale))
        # After dequantization, cast both output and scale to the correct dtype
        if scale_dtype == torch.float16 or scale_dtype == torch.bfloat16:
            x = self.cast_fn(x, scale_dtype)
            flat_pre_scale = self.cast_fn(flat_pre_scale, scale_dtype)
        return x, flat_pre_scale, zero_point, output_bit_width
