# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from copy import copy

import torch
from torch import Tensor

from brevitas.export.common.handler.base import QuantAxisMixin
from brevitas.export.common.handler.qcdq import DQMixin
from brevitas.export.common.handler.qcdq import QCDQActQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQBiasQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQDecoupledWeightQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQMixin
from brevitas.export.common.handler.qcdq import QCDQTruncQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import QCDQWeightQuantProxyHandlerMixin
from brevitas.export.common.handler.qcdq import ZeroPointHandlerMixin
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


class StdQCDQONNXMixin(QCDQMixin, StdDQONNXMixin, ABC):

    @property
    def clip_over_integers(self):
        return True

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
        self.validate_8b_bit_width(module.bit_width(), le_then=True)
        assert module.bit_width() > 1., 'Binary quant not supported'
        assert module.rounding_mode.upper() == 'ROUND', 'Only round to nearest even supported'

    def quantize_fn(self, x, scale, zero_point, dtype, axis):
        return QuantizeLinearFn.apply(x, scale, zero_point, dtype, axis)

    def clip_fn(self, x, min_val, max_val):
        return IntClipFn.apply(x, min_val, max_val)


class StdQCDQONNXWeightQuantProxyHandler(StdQCDQONNXMixin,
                                         QCDQWeightQuantProxyHandlerMixin,
                                         ONNXBaseHandler):
    pass


class StdQCDQONNXDecoupledWeightQuantProxyHandler(StdQCDQONNXMixin,
                                                  QCDQDecoupledWeightQuantProxyHandlerMixin,
                                                  ONNXBaseHandler):
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
