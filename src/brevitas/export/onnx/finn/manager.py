from typing import Tuple, Union, Optional

import torch
from torch import Tensor
from torch.nn import Module, Sequential
from torch.autograd import Function

from brevitas.export.onnx.debug import DebugMarkerFunction
from brevitas.export.onnx.manager import ONNXBaseManager, onnx
from brevitas.export.manager import _set_layer_export_handler, _set_layer_export_mode
from brevitas.quant_tensor import QuantTensor

from .transform import move_quant_attributes_into_annotations
from .transform import restore_domain
from .handler.parameter import FINNQuantConv2dHandler, FINNQuantLinearHandler
from .handler.parameter import FINNQuantConv1dHandler
from .handler.act import FINNQuantHardTanhHandler, FINNQuantReLUHandler, FINNQuantIdentityHandler
from .handler.acc import FINNQuantAvgPool2dHandler
from .utils import finn_datatype
from .function.acc import QuantAvgPool2dFn
from .function.act import QuantHardTanhFn, QuantReLUFn
from .function.parameter import QuantizedLinearFn, QuantizedConvNdFn


class _InputQuantTensorFunction(Function):
    "Account symbolically for scale and zero-point of an input quant tensor"

    @staticmethod
    def symbolic(g, x, scale, zero_point):
        # x is assumed to be an integer valued tensor here
        if zero_point is not None:
            x = g.op('Sub', x, zero_point)
        if scale is not None:
            x = g.op('Mul', x, scale)
        return x

    @staticmethod
    def forward(ctx, x, scale, zero_point):
        return x


class _InputPreprocessingModule(Module):

    def __init__(self, scale, zero_point):
        super(_InputPreprocessingModule, self).__init__()
        if scale is not None:
            self.register_buffer('scale', scale)
        else:
            self.scale = None
        if zero_point is not None:
            self.register_buffer('zero_point', zero_point)
        else:
            self.zero_point = None

    def forward(self, x):
        if torch.onnx.is_in_onnx_export():
            x = _InputQuantTensorFunction.apply(x, self.scale, self.zero_point)
        return x


def set_quant_tensor_datatype(model, tensor_name, datatype: str):
    qa = onnx.TensorAnnotation()
    dt = onnx.StringStringEntryProto()
    dt.key = "finn_datatype"
    dt.value = datatype
    qa.tensor_name = tensor_name
    qa.quant_parameter_tensor_names.append(dt)
    model.graph.quantization_annotation.append(qa)


class FINNManager(ONNXBaseManager):
    target_name = 'FINN'

    handlers = [
        FINNQuantLinearHandler,
        FINNQuantConv1dHandler,
        FINNQuantConv2dHandler,
        FINNQuantReLUHandler,
        FINNQuantIdentityHandler,
        FINNQuantHardTanhHandler,
        FINNQuantAvgPool2dHandler]

    custom_fns = [
        DebugMarkerFunction,
        QuantizedConvNdFn,
        QuantizedLinearFn,
        QuantReLUFn,
        QuantHardTanhFn,
        QuantAvgPool2dFn
    ]

    model_transforms = [
        move_quant_attributes_into_annotations,
        restore_domain]

    onnx_passes = [
        # use initializers instead of Constant nodes for fixed params
        "extract_constant_to_initializer",
        # remove unused graph inputs & initializers
        "eliminate_unused_initializer"]

    @classmethod
    def set_export_mode(cls, module: Module, enabled: bool):
        _set_layer_export_mode(module, enabled)

    @classmethod
    def set_export_handler(cls, module: Module):
        _set_layer_export_handler(cls, module)

    @classmethod
    def export(
            cls,
            module: Module,
            input_shape: Tuple[int, ...] = None,
            export_path: str = None,
            input_t: Optional[Union[Tensor, QuantTensor]] = None,
            **kwargs):
        if input_t is not None and isinstance(input_t, QuantTensor):
            if input_t.is_not_none:
                assert input_t.is_valid, 'Input QuantTensor is not properly quantized'
            training_state = module.training
            preprocessing_module = _InputPreprocessingModule(input_t.scale, input_t.zero_point)
            module = Sequential(preprocessing_module, module)
            module.train(training_state)
        onnx_model = cls.export_onnx(
            module, input_shape, export_path, input_t, **kwargs)
        if input_t is not None and isinstance(input_t, QuantTensor):
            bit_width = input_t.bit_width
            signed = input_t.signed
            if bit_width is not None and signed is not None:
                # '0' is the name of the input tensor to the model unless otherwise specified
                if 'input_names' in kwargs and kwargs['input_names']:
                    input_name = kwargs['input_names'][0]
                else:
                    input_name = '0'
                set_quant_tensor_datatype(onnx_model, input_name, finn_datatype(bit_width, signed))
                if export_path is not None:
                    onnx.save(onnx_model, export_path)
        return onnx_model