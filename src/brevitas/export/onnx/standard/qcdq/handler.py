# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC

import torch

from brevitas.export.common.handler.qcdq import CDQMixin
from brevitas.export.common.handler.qcdq import DQMixin
from brevitas.export.common.handler.qcdq import QCDQActQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQBiasQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQDecoupledWeightQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQDecoupledWeightQuantWithInputProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQTruncQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQWeightQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QMixin
from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.export.onnx.handler import QuantLSTMLayerHandler

from ..function import DequantizeLinearFn
from ..function import IntClipFn
from ..function import QuantizeLinearFn


class StdDQONNXMixin(DQMixin, ABC):

    def dequantize_fn(self, x, scale, zero_point, axis):
        return DequantizeLinearFn.apply(x, scale, zero_point, axis)

    @property
    def flatten_dequantize_params(self):
        return True

    @property
    def itemize_quantize_scalar_params(self):
        return False

    def validate(self, module):
        assert module.bit_width() > 1., 'Binary quant not supported'


class StdCDQONNXMixin(CDQMixin, StdDQONNXMixin, ABC):

    def clip_fn(self, x, min_val, max_val):
        return IntClipFn.apply(x, min_val, max_val)


class StdQCDQONNXMixin(QMixin, StdCDQONNXMixin, ABC):

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
        assert module.rounding_mode.upper() == 'ROUND', 'Only round to nearest even supported'
        self.validate_8b_bit_width(module.bit_width(), le_then=True)

    def quantize_fn(self, x, scale, zero_point, dtype, axis):
        return QuantizeLinearFn.apply(x, scale, zero_point, dtype, axis)


class StdQCDQONNXWeightQuantProxyHandler(StdCDQONNXMixin,
                                         QCDQWeightQuantProxyHandlerMixin,
                                         ONNXBaseHandler):
    pass


class StdQCDQONNXDecoupledWeightQuantProxyHandler(StdCDQONNXMixin,
                                                  QCDQDecoupledWeightQuantProxyHandlerMixin,
                                                  ONNXBaseHandler):
    pass


class StdQCDQONNXDecoupledWeightQuantWithInputProxyHandler(
        StdCDQONNXMixin, QCDQDecoupledWeightQuantWithInputProxyHandlerMixin, ONNXBaseHandler):
    pass


class StdQCDQONNXActQuantProxyHandler(StdQCDQONNXMixin,
                                      QCDQActQuantProxyHandlerMixin,
                                      ONNXBaseHandler):
    pass


class StdQCDQONNXBiasQuantProxyHandler(StdDQONNXMixin,
                                       QCDQBiasQuantProxyHandlerMixin,
                                       ONNXBaseHandler):
    pass


class StdQCDQONNXTruncQuantProxyHandler(StdQCDQONNXMixin,
                                        QCDQTruncQuantProxyHandlerMixin,
                                        ONNXBaseHandler):
    pass


class StdQCDQONNXQuantLSTMLayerHandler(QuantLSTMLayerHandler):

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
