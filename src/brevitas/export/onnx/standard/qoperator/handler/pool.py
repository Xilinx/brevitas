from abc import ABC
from typing import Union

import torch
from torch import Tensor

# early import of max_pool to avoid being affected by monkeypatching
from torch.nn.functional import max_pool1d, max_pool2d

from brevitas.nn import QuantMaxPool1d, QuantMaxPool2d
from .base import StdQOpONNXQuantWrapperHandler


class StdQOpONNXQuantMaxPoolNd(StdQOpONNXQuantWrapperHandler, ABC):


    @classmethod
    def op_symbolic_kwargs(cls, module: Union[QuantMaxPool1d, QuantMaxPool2d]):
        return {
            'kernel_size': module.kernel_size,
            'stride': module.stride,
            'padding': module.padding,
            'dilation': module.dilation,
            'ceil_mode': module.ceil_mode,
            'return_indices': module.return_indices}


class StdQOpONNXQuantMaxPool1d(StdQOpONNXQuantMaxPoolNd):
    handled_layer = QuantMaxPool1d

    def op_symbolic_execution(self, inp: Tensor):
        op_symbolic_kwargs = self.symbolic_kwargs['op_symbolic_kwargs']
        return max_pool1d(inp, *op_symbolic_kwargs.values())


class StdQOpONNXQuantMaxPool2d(StdQOpONNXQuantMaxPoolNd):
    handled_layer = QuantMaxPool2d

    def op_symbolic_execution(self, inp: Tensor):
        op_symbolic_kwargs = self.symbolic_kwargs['op_symbolic_kwargs']
        return max_pool2d(inp, *op_symbolic_kwargs.values())