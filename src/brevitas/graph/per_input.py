from functools import reduce
from operator import mul

from torch import nn

from brevitas.graph.utils import replace_module
from .base import PerInputModuleToModuleByHook

__all__ = [
    'AdaptiveAvgPoolToAvgPool',
    'AvgPoolToDepthwiseConv'
]


class AdaptiveAvgPoolToAvgPool(PerInputModuleToModuleByHook):

    SUPPORTED_LAYERS = (
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)

    def register_hooks(self, model):
        for module in model.modules():
            if isinstance(module, self.SUPPORTED_LAYERS):
                hook_handler = module.register_forward_pre_hook(self.hook_fn)
                self.hook_handlers.append(hook_handler)
    
    def get_adaptive_output_size(self, adaptive_avgpool):
        output_size = adaptive_avgpool.output_size
        if isinstance(output_size, tuple):
            return output_size
        else:
            assert isinstance(output_size, int)
            if isinstance(adaptive_avgpool, nn.AdaptiveAvgPool1d):
                return (output_size,)
            elif isinstance(adaptive_avgpool, nn.AdaptiveAvgPool2d):
                return (output_size, output_size)
            else:
                assert isinstance(adaptive_avgpool, nn.AdaptiveAvgPool3d)
                return (output_size, output_size, output_size)
    
    def replace_modules(self, model):
        for adaptive_avgpool, size in self.input_size_map.items():
            output_size = self.get_adaptive_output_size(adaptive_avgpool)
            input_size = size[-len(output_size):]
            mod = [input_size[i] % output_size[i] for i in range(0, len(output_size))]
            if mod == [0] * len(output_size):
                k = tuple(int(input_size[i] / output_size[i]) for i in range(0, len(output_size)))
                kwargs = {'kernel_size': k, 'stride': k}
                if isinstance(adaptive_avgpool, nn.AdaptiveAvgPool1d):
                    avgpool = nn.AvgPool1d(**kwargs)
                elif isinstance(adaptive_avgpool, nn.AdaptiveAvgPool2d):
                    avgpool = nn.AvgPool2d(**kwargs)
                else:
                    assert isinstance(adaptive_avgpool, nn.AdaptiveAvgPool3d)
                    avgpool = nn.AvgPool3d(**kwargs)
                replace_module(model, adaptive_avgpool, avgpool)


class AvgPoolToDepthwiseConv(PerInputModuleToModuleByHook):

    SUPPORTED_LAYERS = (
        nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)

    def register_hooks(self, model):
        for module in model.modules():
            if isinstance(module, self.SUPPORTED_LAYERS):
                hook_handler = module.register_forward_pre_hook(self.hook_fn)
                self.hook_handlers.append(hook_handler)

    def replace_modules(self, model):
        for avgpool, size in self.input_size_map.items():
            kwargs = {
                'kernel_size': avgpool.kernel_size,
                'stride': avgpool.stride,
                'padding': avgpool.padding,
                'in_channels': size[1],
                'out_channels': size[1],
                'groups': size[1],
                'bias': False}
            if isinstance(avgpool, nn.AvgPool1d):
                dw_conv = nn.Conv1d(**kwargs)
            elif isinstance(avgpool, nn.AvgPool2d):
                dw_conv = nn.Conv2d(**kwargs)
            else:
                assert isinstance(avgpool, nn.AvgPool3d)
                dw_conv = nn.Conv3d(**kwargs)
            kernel_value = 1. / reduce(mul, dw_conv.kernel_size)
            dw_conv.weight.data.fill_(kernel_value)
            replace_module(model, avgpool, dw_conv)