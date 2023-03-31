# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch import Tensor

from brevitas.nn import TruncAvgPool2d

from ..function.acc import TruncAvgPool2dFn
from .base import FINNQuantIOHandler


class FINNTruncAvgPool2dHandler(FINNQuantIOHandler):
    handled_layer = TruncAvgPool2d

    @staticmethod
    def quant_output_shape(module: TruncAvgPool2d):
        shape = FINNQuantIOHandler.quant_output_shape(module)
        if shape is None:
            raise RuntimeError("Caching of output shapes is required to export TruncAvgPool2d")
        return shape

    @staticmethod
    def quant_input_bit_width(module: TruncAvgPool2d):
        bit_width = FINNQuantIOHandler.quant_input_bit_width_tensor(module)
        if bit_width is None:
            raise RuntimeError("Caching of input bit width is required to export TruncAvgPool2d")
        return int(bit_width.item())

    @staticmethod
    def quant_output_bit_width(module: TruncAvgPool2d):
        bit_width = FINNQuantIOHandler.quant_output_bit_width_tensor(module)
        if bit_width is None:
            raise RuntimeError("Caching of output bit width is required to export TruncAvgPool2d")
        return int(bit_width.item())

    @staticmethod
    def quant_input_signed(module: TruncAvgPool2d) -> int:
        signed = FINNQuantIOHandler.quant_input_signed(module)
        if signed is None:
            raise RuntimeError("Output sign of TruncAvgPool2d is malformed")
        return int(signed)

    def prepare_for_export(self, module):
        self.symbolic_kwargs = {
            'out_shape': self.quant_output_shape(module),
            'kernel': module.kernel_size,
            'stride': module.stride,
            'signed': self.quant_input_signed(module),
            'ibits': self.quant_input_bit_width(module),
            'obits': self.quant_output_bit_width(module),
            'scale': self.quant_input_scale(module),
            'qnt_type': self.quant_input_type(module)}

    def symbolic_execution(self, inp: Tensor):
        ret = TruncAvgPool2dFn.apply(inp, *self.symbolic_kwargs.values())
        return ret
