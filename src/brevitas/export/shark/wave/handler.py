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
        if module.input_quant.is_quant_enabled:
            input_quant = module.input_quant
            self.input_quant.prepare_for_export(input_quant)
        quant_params = {
            'weight_scale': self.weight_quant.scale,
            'weight_scale_shape': self.weight_quant.scale.shape,
            'input_scale': self.input_quant.scale,
            'input_scale_shape': self.input_quant.scale.shape,
            'qdtype': torch.float8_e4m3fnuz}
        # self.wave_linear = WaveQuantLinear(
        #     input_feat, out_feat, quant_params, bias=False)
        # self.wave_linear.weight.data = module.weight.data
        # if module.bias is not None:
        #     self.wave_linear.bias.data = module.bias.data 
        self.bias = module.bias
        self.weight = module.weight
        del module.weight
        del module.bias

    def forward(self, input):
        input_q = self.input_quant.quantize(input, self.input_quant.scale.to(input.device), None)
        weight_q = self.weight_quant.quantize(self.weight, self.weight_quant.scale.to(input.device), None)

        if len(input_q.shape) > 2:
            B = input_q.shape[0]
            output_1 = torch.stack([torch._scaled_mm(
                input_q[i].to(torch.float8_e4m3fnuz),
                weight_q.t().to(torch.float8_e4m3fnuz),
                scale_a=self.input_quant.scale.to(input.device),
                scale_b=self.weight_quant.scale.to(input.device),
                # bias=self.bias,
                out_dtype=torch.float16
                ) for i in range(B)], dim=0)
        else:
            output_1 = torch._scaled_mm(
                input_q.to(torch.float8_e4m3fnuz),
                weight_q.t().to(torch.float8_e4m3fnuz),
                scale_a=self.input_quant.scale.to(input.device),
                scale_b=self.weight_quant.scale.to(input.device),
                # bias=self.bias,
                out_dtype=torch.float16
                )

        if self.bias is not None:
            output_1 += self.bias

        return output_1
