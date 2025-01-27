# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from copy import copy

import torch
from torch import Tensor

from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.export.onnx.handler import QuantLSTMLayerHandler
from brevitas.proxy import ActQuantProxyFromInjector
from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.proxy import DecoupledWeightQuantProxyFromInjector
from brevitas.proxy import DecoupledWeightQuantWithInputProxyFromInjector
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjector
from brevitas.proxy.float_runtime_quant import ActFloatQuantProxyFromInjector
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector

from .function import BrevitasBinaryQuantFn
from .function import BrevitasFloatQuantFn
from .function import BrevitasQuantFn
from .function import BrevitasQuantLSTMCellFn
from .function import BrevitasTruncFn


class BrevitasFloatQuantProxyHandler(ONNXBaseHandler, ABC):

    def validate(self, module):
        assert not module.is_groupwise, "Export with Per Group quantization not supported"

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            self.symbolic_kwargs = {
                'scale':
                    module.scale(),
                'exponent_bit_width':
                    module.exponent_bit_width(),
                'mantissa_bit_width':
                    module.mantissa_bit_width(),
                'exponent_bias':
                    module.exponent_bias(),
                'has_inf':
                    module.inf_values() is not None,
                'has_nan':
                    module.nan_values() is not None,
                'saturating':
                    module.is_saturating(),
                'has_subnormal':
                    True,  # Currently we only support subnormal
                'rounding_mode':
                    module.rounding_mode,
                'max_float':
                    torch.tensor(module.quant_injector.max_available_float).type_as(module.scale())}
            self.return_args = {
                'scale': module.scale(),
                'zero_point': torch.zeros_like(module.scale()),
                'exponent_bit_width': module.exponent_bit_width(),
                'mantissa_bit_width': module.mantissa_bit_width(),
                'exponent_bias': module.exponent_bias(),
                'saturating': module.is_saturating(),
                'inf_values': module.inf_values(),
                'nan_values': module.nan_values(),}

    def symbolic_execution(self, x: Tensor):
        x = BrevitasFloatQuantFn.apply(x, *self.symbolic_kwargs.values())
        return_args = (x, *self.return_args.values())
        return return_args


class BrevitasWeightFloatQuantProxyHandler(BrevitasFloatQuantProxyHandler):
    handled_layer = WeightFloatQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.quant_weights = None

    def validate(self, zero_point):
        assert zero_point == 0, "Zero-point not supported for minifloat quant."

    def prepare_for_export(self, module: WeightQuantProxyFromInjector):
        if module.is_quant_enabled:
            first_qweight = module.tracked_module_list[0].quant_weight()
            self.validate(first_qweight.zero_point)
            self.symbolic_kwargs = {
                'scale':
                    first_qweight.scale,
                'exponent_bit_width':
                    first_qweight.exponent_bit_width,
                'mantissa_bit_width':
                    first_qweight.mantissa_bit_width,
                'exponent_bias':
                    first_qweight.exponent_bias,
                'has_inf':
                    first_qweight.inf_values is not None,
                'has_nan':
                    first_qweight.nan_values is not None,
                'saturating':
                    first_qweight.saturating,
                'has_subnormal':
                    True,  # Currently we only support subnormal
                'rounding_mode':
                    module.rounding_mode,
                'max_float':
                    torch.tensor(module.quant_injector.max_available_float
                                ).type_as(first_qweight.scale)}
            self.return_args = {
                'scale': first_qweight.scale,
                'zero_point': torch.zeros_like(first_qweight.scale),
                'exponent_bit_width': first_qweight.exponent_bit_width,
                'mantissa_bit_width': first_qweight.mantissa_bit_width,
                'exponent_bias': first_qweight.exponent_bias,
                'saturating': first_qweight.saturating,
                'inf_values': first_qweight.inf_values,
                'nan_values': first_qweight.nan_values,}

    def symbolic_execution(self, x: Tensor):
        return super().symbolic_execution(x)


class BrevitasActFloatQuantProxyHandler(BrevitasFloatQuantProxyHandler):
    handled_layer = ActFloatQuantProxyFromInjector


class BrevitasQuantProxyHandler(ONNXBaseHandler, ABC):

    def validate(self, module):
        if module.bit_width() == 1:
            assert module.zero_point() == 0, "Zero-point not supported for binary quant."
        assert not module.is_groupwise, "Export with Per Group quantization not supported"

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            self.symbolic_kwargs = {
                'scale': module.scale(),
                'zero_point': module.zero_point(),
                'bit_width': module.bit_width(),
                'narrow_range': module.is_narrow_range,
                'signed': module.is_signed,
                'rounding_mode': module.rounding_mode}

    def symbolic_execution(self, x: Tensor):
        scale = self.symbolic_kwargs['scale']
        zero_point = self.symbolic_kwargs['zero_point']
        bit_width = self.symbolic_kwargs['bit_width']
        if bit_width == 1:
            x = BrevitasBinaryQuantFn.apply(x, *self.symbolic_kwargs.values())
        else:
            x = BrevitasQuantFn.apply(x, *self.symbolic_kwargs.values())
        return x, scale, zero_point, bit_width


class BrevitasWeightQuantProxyHandler(BrevitasQuantProxyHandler):
    handled_layer = WeightQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.quant_weights = None

    def validate(self, bit_width, zero_point):
        if bit_width == 1:
            assert zero_point == 0, "Zero-point not supported for binary quant."

    def prepare_for_export(self, module: WeightQuantProxyFromInjector):
        if module.is_quant_enabled:
            first_qweight = module.tracked_module_list[0].quant_weight()
            self.validate(first_qweight.bit_width, first_qweight.zero_point)
            self.quant_weight_values = {
                tm.weight.data_ptr(): tm.quant_weight().value for tm in module.tracked_module_list}
            self.symbolic_kwargs = {
                'scale': first_qweight.scale,
                'zero_point': first_qweight.zero_point,
                'bit_width': first_qweight.bit_width,
                # narrow_range is not a property of the QuantTensor, take it from the proxy instead
                'narrow_range': module.is_narrow_range,
                'signed': first_qweight.signed,
                # override rounding mode since quantization has been pre-applied
                'rounding_mode': 'ROUND'}

    def symbolic_execution(self, x: Tensor):
        quant_weight = self.quant_weight_values[x.data_ptr()]
        return super().symbolic_execution(quant_weight)


class BrevitasDecoupledWeightQuantProxyHandler(BrevitasWeightQuantProxyHandler):
    handled_layer = DecoupledWeightQuantProxyFromInjector

    def symbolic_execution(
            self, x: Tensor, *args):  # args supports DecoupledWeightQuantWithInputProxyFromInjector
        out, scale, zero_point, bit_width = super().symbolic_execution(x)
        # TODO fix retrieval for DecoupledWeightQuantWithInputProxyFromInjector
        # In practice it doesn't make a difference since the pre_* values are unused during export
        pre_scale, pre_zero_point = scale, zero_point
        return out, scale, zero_point, bit_width, pre_scale, pre_zero_point


class BrevitasDecoupledWeightQuantWithInputProxyHandler(BrevitasDecoupledWeightQuantProxyHandler):
    handled_layer = DecoupledWeightQuantWithInputProxyFromInjector


class BrevitasActQuantProxyHandler(BrevitasQuantProxyHandler):
    handled_layer = ActQuantProxyFromInjector


class BrevitasBiasQuantProxyHandler(BrevitasQuantProxyHandler):
    handled_layer = BiasQuantProxyFromInjector

    def symbolic_execution(self, x: Tensor, input_scale=None, input_bit_width=None):
        # avoid in place pop in case the proxy is shared
        symbolic_kwargs = copy(self.symbolic_kwargs)
        scale = symbolic_kwargs.pop('scale')
        bit_width = symbolic_kwargs.pop('bit_width')
        zero_point = symbolic_kwargs.pop('zero_point')
        if scale is None:
            assert input_scale is not None, 'Input scale required for bias export'
            scale = input_scale
        if bit_width is None:
            assert input_bit_width is not None, 'Input bit_width required for bias export'
            bit_width = input_bit_width
        y = BrevitasQuantFn.apply(x, scale, zero_point, bit_width, *symbolic_kwargs.values())
        return y, scale, zero_point, bit_width


class BrevitasTruncQuantProxyHandler(ONNXBaseHandler):
    handled_layer = TruncQuantProxyFromInjector

    def prepare_for_export(self, module: TruncQuantProxyFromInjector):
        self.symbolic_kwargs = {
            'narrow_range': module.is_narrow_range,
            'output_scale': module.scale(),
            'output_bit_width': module.bit_width(),
            'rounding_mode': module.rounding_mode}

    def symbolic_execution(
            self, x: Tensor, scale: Tensor, zero_point: Tensor, input_bit_width: Tensor,
            signed: Tensor):
        y = BrevitasTruncFn.apply(
            x, scale, zero_point, input_bit_width, signed, *self.symbolic_kwargs.values())
        return y, self.symbolic_kwargs['output_scale'], zero_point, self.symbolic_kwargs['output_bit_width']


class BrevitasQuantLSTMLayerHandler(QuantLSTMLayerHandler):

    def quantized_cell_symbolic_execution(
            self,
            quant_input,
            quant_hidden_state,
            quant_cell_state,
            quant_weight_ii,
            quant_weight_if,
            quant_weight_ic,
            quant_weight_io,
            quant_weight_hi,
            quant_weight_hf,
            quant_weight_hc,
            quant_weight_ho,
            quant_bias_input,
            quant_bias_forget,
            quant_bias_cell,
            quant_bias_output):
        return BrevitasQuantLSTMCellFn.apply(
            quant_input,
            quant_hidden_state,
            quant_cell_state,
            quant_weight_ii,
            quant_weight_if,
            quant_weight_ic,
            quant_weight_io,
            quant_weight_hi,
            quant_weight_hf,
            quant_weight_hc,
            quant_weight_ho,
            quant_bias_input,
            quant_bias_forget,
            quant_bias_cell,
            quant_bias_output,
            *self.symbolic_kwargs.values())
