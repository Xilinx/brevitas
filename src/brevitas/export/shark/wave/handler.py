# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from typing import Tuple

from iree.turbine.kernel.wave.nn import WaveQuantLinear
import torch
from torch import Tensor
import torch.nn as nn

from brevitas.export.inference.handler import FloatInferencetHandler
from brevitas.export.inference.handler import FloatWeightInferencetHandler
from brevitas.nn import QuantLinear


class InferenceHandler(torch.nn.Module, ABC):

    def attach_debug_info(self, module: nn.Module):
        pass

    @abstractmethod
    def prepare_for_export(self, module: nn.Module):
        pass

    @abstractmethod
    def quantize(self, x: Tensor):
        pass

    @abstractmethod
    def dequantize(self, x: Tensor):
        pass


class QuantLinearFp8Handler(InferenceHandler):
    handled_layer = QuantLinear

    def __init__(self):
        super().__init__()
        self.weight_quant = FloatWeightInferencetHandler()
        self.input_quant = FloatInferencetHandler()
        self.wave_linear = None

    def validate(self, module):
        # TODO: Check that we are quantizing to the correct fp8 type, etc. etc.
        pass

    def prepare_for_export(self, module):
        ## Weight export
        out_feat, input_feat = module.weight.shape[0], module.weight.shape[1]
        if module.weight_quant.is_quant_enabled:
            weight_quant = module.weight_quant
            self.weight_quant.prepare_for_export(weight_quant)
        elif module.input_quant.is_quat_enabled:
            input_quant = module.input_quant
            self.input_quant.prepare_for_export(input_quant)
        quant_params = {
            'weight_scale': self.weight_quant.scale,
            'weight_scale_shape': self.weight_quant.scale.shape,
            'input_scale': self.input_quant.scale,
            'input_scale_shape': self.input_quant.scale.shape,
            'qdtype': torch.float8_e4m3fnuz}
        self.wave_linear = WaveQuantLinear(
            input_feat, out_feat, quant_params, bias=module.bias is not None)
        self.wave_linear.load_state_dict(module.state_dict())
        del module.weight
        del module.bias

    def forward(self, input):
        output = self.wave_linear(input)
        return output
