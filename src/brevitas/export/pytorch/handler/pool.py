from abc import ABC
from typing import Union

from brevitas.nn import QuantMaxPool1d, QuantMaxPool2d
from .base import PytorchQuantLayerHandler
from . import qF


class PytorchQuantMaxPoolNd(PytorchQuantLayerHandler, ABC):

    @classmethod
    def validate(cls, module):
        # nothing to do here, pytorch's quant max pool is standard max pool
        pass

    @classmethod
    def explicit_output_dtype(cls) -> bool:
        return False

    @classmethod
    def prepare_qf_kwargs(cls, module: Union[QuantMaxPool1d, QuantMaxPool2d]):
        return {
            'kernel_size': module.kernel_size,
            'stride': module.stride,
            'padding': module.padding,
            'dilation': module.dilation,
            'ceil_mode': module.ceil_mode,
            'return_indices': module.return_indices}

    def prepare_for_export(self, module):
        self.qf_impl, self.qf_kwargs = self.prepare_qf(module)
        self.return_quant_tensor = module.return_quant_tensor

    def forward(self, inp):
        out = self.qf_impl(inp, **self.qf_kwargs)
        # We are being tolerant here to non quantized tensors
        if out.is_quantized and not self.return_quant_tensor:
            out = out.dequantize()
        return out


class PytorchQuantMaxPool1d(PytorchQuantMaxPoolNd):
    handled_layer = QuantMaxPool1d

    @classmethod
    def prepare_qf(cls, module: QuantMaxPool1d):
        return qF.max_pool1d, cls.prepare_qf_kwargs(module)


class PytorchQuantMaxPool2d(PytorchQuantMaxPoolNd):
    handled_layer = QuantMaxPool2d

    @classmethod
    def prepare_qf(cls, module: QuantMaxPool1d):
        return qF.max_pool2d, cls.prepare_qf_kwargs(module)