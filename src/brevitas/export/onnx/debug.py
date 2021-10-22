# -*- coding: future_annotations -*-

import torch
from torch.autograd import Function

from brevitas.nn.mixin.base import QuantLayerMixin
from brevitas.quant_tensor import QuantTensor
from brevitas.proxy.quant_proxy import QuantProxyProtocol


class DebugMarkerFunction(Function):

    @staticmethod
    def symbolic(g, input, export_debug_name):
        ret = g.op(
            'brevitas.onnx::DebugMarker', input, export_debug_name_s=export_debug_name)
        return ret

    @staticmethod
    def forward(ctx, input, export_debug_name):
        return input


class ONNXDebugHook(object):

    def __init__(self, input_enabled, output_enabled):
        self.values = {}
        self.input_enabled = input_enabled
        self.output_enabled = output_enabled

    def unpack(self, value):
        if isinstance(value, tuple) and len(value) == 1:
            return value[0]
        else:
            return value

    def __call__(self, module, module_in, module_out):
        if self.input_enabled:
            self.values[module.export_debug_name + ".input"] = self.unpack(module_in)
        if self.output_enabled:
            self.values[module.export_debug_name + ".output"] = self.unpack(module_out)

    def clear(self):
        self.values = {}


def enable_debug(
        model,
        input_enabled=True,
        output_enabled=True,
        proxy_level=False):
    base_filter_class = QuantProxyProtocol if proxy_level else QuantLayerMixin
    filter_fn = lambda x: isinstance(x, base_filter_class)
    hook = ONNXDebugHook(input_enabled, output_enabled)
    for name, module in model.named_modules():
        if hasattr(module, "export_debug_name") and filter_fn(module):
            module.export_debug_name = name
            module.export_input_debug = input_enabled
            module.export_output_debug = output_enabled
            module.register_forward_hook(hook)
    return hook
