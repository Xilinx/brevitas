from brevitas.nn.mixin.base import QuantLayerMixin


class ONNXDebugHook(object):

    def __init__(self):
        self.inputs = {}
        self.outputs = {}

    def __call__(self, module: QuantLayerMixin, module_in, module_out):
        inp = module._cached_inp.quant_tensor.value
        out = module._cached_out.quant_tensor.value
        self.inputs[module.export_debug_name] = inp
        self.outputs[module.export_debug_name] = out

    def clear(self):
        self.inputs = {}
        self.outputs = {}


def enable_debug(module, hook=ONNXDebugHook(), filter_fn=lambda x: True):
    """Enable debug and set up given forward hook on all QuantLayer-derived children
    that return True when passed to filter_fn (always True by default).
    Debug-enabled nodes will create DebugMarker nodes when exported.
    """
    for child_name, child_module in module.named_modules():
        if hasattr(child_module, "export_debug_name") and filter_fn(child_module):
            child_module.cache_quant_metadata_only = False
            child_module.export_debug_name = child_name
            child_module.register_forward_hook(hook)
    return hook