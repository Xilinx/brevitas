# -*- coding: future_annotations -*-
from typing import TYPE_CHECKING

import torch
from torch.autograd import Function

if TYPE_CHECKING:
    from brevitas.nn.mixin.base import QuantLayerMixin


class DebugMarkerFunction(Function):

    @staticmethod
    def symbolic(g, input, export_debug_name):
        ret = g.op('DebugMarker', input, export_debug_name_s=export_debug_name, domain_s="finn.custom_op.general")
        return ret

    @staticmethod
    def forward(ctx, input, export_debug_name):
        return torch.empty(input.shape, dtype=torch.float)


class ONNXDebugHook(object):

    def __init__(self):
        self.values = {}

    def __call__(self, module: QuantLayerMixin, module_in, module_out):
        inp = module._cached_inp.quant_tensor.value
        out = module._cached_out.quant_tensor.value
        if inp is not None:
            self.values[module.export_debug_name + ".input"] = inp
        if out is not None:
            self.values[module.export_debug_name + ".output"] = out

    def clear(self):
        self.values = {}


def enable_debug(module, hook=ONNXDebugHook(), inp=True, out=True, filter_fn=lambda x: True):
    """Enable debug and set up given forward hook on all QuantLayer-derived children
    that return True when passed to filter_fn (always True by default).
    Debug-enabled nodes will create DebugMarker nodes when exported.
    """
    for child_name, child_module in module.named_modules():
        if hasattr(child_module, "export_debug_name") and filter_fn(child_module):
            child_module.cache_inference_quant_inp = inp
            child_module.cache_inference_quant_out = out
            child_module.cache_quant_io_metadata_only = False
            child_module.export_debug_name = child_name
            child_module.register_forward_hook(hook)
    return hook
