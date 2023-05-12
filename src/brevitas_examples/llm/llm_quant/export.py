import torch

from brevitas.export.common.handler.base import BaseHandler
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import BaseManager
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector


class WeightBlockQuantProxyHandler(BaseHandler):
    handled_layer = WeightQuantProxyFromInjector

    def __init__(self):
        super(WeightBlockQuantProxyHandler, self).__init__()
        self.int_weight = None
        self.scale = None
        self.expanded_scaling_shape = None
        self.reshaped_scaling_shape = None

    def prepare_for_export(self, module):
        assert len(module.tracked_module_list) == 1, "Shared quantizers not supported."
        quant_layer = module.tracked_module_list[0]
        quant_weight = quant_layer.quant_weight()
        self.int_weight = quant_weight.int().to(torch.int8).detach()
        scaling_impl = module.tensor_quant.scaling_impl
        int_scaling_impl = module.tensor_quant.int_scaling_impl
        bit_width_impl = module.tensor_quant.msb_clamp_bit_width_impl
        int_threshold = int_scaling_impl(bit_width_impl())
        threshold = scaling_impl.wrapped_scaling_impl.stats_scaling_impl(
            scaling_impl.wrapped_scaling_impl.parameter_list_stats())
        self.scale = threshold / int_threshold
        self.expanded_scaling_shape = scaling_impl.expanded_scaling_shape
        self.reshaped_scaling_shape = scaling_impl.reshaped_scaling_shape

    def forward(self, x):
        scale = self.scale.expand(self.expanded_scaling_shape).contiguous()
        # contiguous above is to avoid the reshape below being mapped to a unsafe view
        scale = scale.view(self.reshaped_scaling_shape)
        quant_weight = self.int_weight * scale
        return quant_weight, scale, None, None


class BlockProxyLevelManager(BaseManager):

    handlers = [WeightBlockQuantProxyHandler]

    @classmethod
    def set_export_handler(cls, module):
        _set_proxy_export_handler(cls, module)


def brevitas_block_proxy_export_mode(model, enabled):
    if enabled:
        model.eval()
        model.apply(BlockProxyLevelManager.set_export_handler)
        _set_proxy_export_mode(model, enabled=True)
    else:
        _set_proxy_export_mode(model, enabled=False)
