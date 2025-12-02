# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
import warnings
from warnings import warn

import torch

from brevitas.export.common.handler.qcdq import CDQCastBiasQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import CDQCastMixin
from brevitas.export.common.handler.qcdq import DQCastMixin
from brevitas.export.common.handler.qcdq import DynamicQDQCastActQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import DynamicQMixin
from brevitas.export.common.handler.qcdq import FloatQCDQCastActQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import FloatQCDQCastWeightQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import FloatQMixin
from brevitas.export.common.handler.qcdq import QCDQCastActQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQCastDecoupledWeightQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import \
    QCDQCastDecoupledWeightQuantWithInputProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQCastTruncQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQCastWeightQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QMixin
from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.export.onnx.handler import QuantLSTMLayerHandler
from brevitas.inject.enum import ScalingPerOutputType

from ..function import CastFn
from ..function import DequantizeLinearFn
from ..function import DynamicQuantizeLinearFn
from ..function import IntClipFn
from ..function import QuantizeLinearFn


class StdDQCastONNXMixin(DQCastMixin, ABC):

    def dequantize_fn(self, x, scale, zero_point, axis):
        return DequantizeLinearFn.apply(x, scale, zero_point, axis)

    def cast_fn(self, x, dtype):
        return CastFn.apply(x, dtype)

    @property
    def flatten_dequantize_params(self):
        return True

    @property
    def itemize_quantize_scalar_params(self):
        return False

    def validate(self, module):
        assert module.bit_width() > 1., 'Binary quant not supported'


class StdFloatDQCastONNXMixin(StdDQCastONNXMixin, ABC):

    def validate(self, module):
        assert module.is_ocp or module.is_fnuz, 'Only OCP/FNUZ Standard are supported for FP8 export'


class StdFloatCDQCastONNXMixin(CDQCastMixin, StdFloatDQCastONNXMixin, ABC):

    def clip_fn(self, x, min_val, max_val):
        raise NotImplementedError


class StdCDQCastONNXMixin(CDQCastMixin, StdDQCastONNXMixin, ABC):

    def clip_fn(self, x, min_val, max_val):
        return IntClipFn.apply(x, min_val, max_val)


class StdFloatQCDQCastONNXMixin(FloatQMixin, StdFloatCDQCastONNXMixin, ABC):

    def validate(self, module):
        if getattr(self, '_export_q_node', True):
            if module.rounding_mode.upper() != 'ROUND':
                warnings.warn("Exporting different rounding function than ROUND requires exporting" \
                            " and storing fake quantized weights. This could cause OOM issues.")
                self.export_fake_quantized = True
        super().validate(module)

    def quantize_fn(self, x, scale, zero_point, dtype, axis):
        return QuantizeLinearFn.apply(x, scale, zero_point, dtype, axis)


class StdQCDQCastONNXMixin(QMixin, StdCDQCastONNXMixin, ABC):

    @classmethod
    def int8_dtype(cls):
        return torch.int8

    @classmethod
    def uint8_dtype(cls):
        return torch.uint8

    @classmethod
    def int32_dtype(cls):
        return torch.int32

    def validate(self, module):
        super().validate(module)
        # ONNX QuantizeLinear supports only 8b output with round to nearest even.
        # Below 8b quantization is supported through clipping.
        if getattr(self, '_export_q_node', True):
            if module.rounding_mode.upper() != 'ROUND':
                self.export_fake_quantized = True
        assert not module.is_groupwise, "Export with Per Group quantization not supported"

        self.validate_8b_bit_width(module.bit_width(), le_then=True)

    def quantize_fn(self, x, scale, zero_point, dtype, axis):
        return QuantizeLinearFn.apply(x, scale, zero_point, dtype, axis)


class StdDynamicQDQCastONNXMixin(DynamicQMixin, StdDQCastONNXMixin, ABC):

    @classmethod
    def int8_dtype(cls):
        return torch.int8

    @classmethod
    def uint8_dtype(cls):
        return torch.uint8

    @classmethod
    def int32_dtype(cls):
        return torch.int32

    def validate(self, module):
        super().validate(module)

        assert module.is_signed == False, "Only unsigned quantization supported"
        assert module.quant_injector.scaling_stats_op == 'min_max', "Only min_max scaling op supported"
        # ONNX QuantizeLinear supports only 8b output with round to nearest even.
        # Below 8b quantization is supported through clipping.
        assert module.rounding_mode.upper() == 'ROUND', 'Only round to nearest even supported'
        # Below 8b quantization is not supported.
        self.validate_8b_bit_width(module.bit_width(), le_then=False)
        # Only per tensor quantization is supported
        assert module.quant_injector.scaling_per_output == ScalingPerOutputType.TENSOR, "Only per tensor scaling supported"

    def quantize_fn(self, x, dtype):
        return DynamicQuantizeLinearFn.apply(x, dtype)


class StdFloatQCDQCastONNXWeightQuantProxyHandler(StdFloatQCDQCastONNXMixin,
                                                  FloatQCDQCastWeightQuantProxyHandlerMixin,
                                                  ONNXBaseHandler):
    _export_q_node = False


class StdQCDQCastONNXWeightQuantProxyHandler(StdQCDQCastONNXMixin,
                                             QCDQCastWeightQuantProxyHandlerMixin,
                                             ONNXBaseHandler):
    _export_q_node = False


class StdQCDQCastONNXDecoupledWeightQuantProxyHandler(StdQCDQCastONNXMixin,
                                                      QCDQCastDecoupledWeightQuantProxyHandlerMixin,
                                                      ONNXBaseHandler):
    _export_q_node = False


class StdQCDQCastONNXDecoupledWeightQuantWithInputProxyHandler(
        StdQCDQCastONNXMixin, QCDQCastDecoupledWeightQuantWithInputProxyHandlerMixin,
        ONNXBaseHandler):
    _export_q_node = False


class StdFloatQCDQCastONNXActQuantProxyHandler(StdFloatQCDQCastONNXMixin,
                                               FloatQCDQCastActQuantProxyHandlerMixin,
                                               ONNXBaseHandler):
    pass


class StdQCDQCastONNXActQuantProxyHandler(StdQCDQCastONNXMixin,
                                          QCDQCastActQuantProxyHandlerMixin,
                                          ONNXBaseHandler):
    pass


class StdDynamicQDQCastONNXActQuantProxyHandler(StdDynamicQDQCastONNXMixin,
                                                DynamicQDQCastActQuantProxyHandlerMixin,
                                                ONNXBaseHandler):
    pass


class StdCDQCastONNXBiasQuantProxyHandler(StdDQCastONNXMixin,
                                          CDQCastBiasQuantProxyHandlerMixin,
                                          ONNXBaseHandler):
    pass


class StdQCDQCastONNXTruncQuantProxyHandler(StdQCDQCastONNXMixin,
                                            QCDQCastTruncQuantProxyHandlerMixin,
                                            ONNXBaseHandler):
    pass


class StdQCDQCastONNXQuantLSTMLayerHandler(QuantLSTMLayerHandler):

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
        raise RuntimeError(
            "Quantized LSTM cell is not supported for ONNX QCDQ "
            "(weights only quantization is). Use export_qonnx.")
