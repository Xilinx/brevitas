# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import reduce
from operator import mul

import torch
from torch import nn

from brevitas.graph.utils import replace_module
from brevitas.nn import QuantConv1d
from brevitas.nn import QuantConv2d

from .base import PerInputModuleToModuleByHook

__all__ = ['AdaptiveAvgPoolToAvgPool', 'AvgPoolToQuantDepthwiseConv']


class AdaptiveAvgPoolToAvgPool(PerInputModuleToModuleByHook):

    SUPPORTED_LAYERS = (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)

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

    def replace_modules(self, model, global_avgpool_unit_stride=True):
        for adaptive_avgpool, size in self.input_size_map.items():
            output_size = self.get_adaptive_output_size(adaptive_avgpool)
            input_size = size[-len(output_size):]
            mod = [input_size[i] % output_size[i] for i in range(0, len(output_size))]
            if mod == [0] * len(output_size):
                # Reference https://stackoverflow.com/a/63603993/16744139
                s = tuple(int(input_size[i] / output_size[i]) for i in range(0, len(output_size)))
                k = tuple(
                    input_size[i] - s[i] * (output_size[i] - 1) for i in range(0, len(output_size)))
                # Set stride 1 whenever the adaptive avg pool is global
                if global_avgpool_unit_stride and all(os == 1 for os in output_size):
                    s = tuple([1] * len(s))
                kwargs = {'kernel_size': k, 'stride': s}
                if isinstance(adaptive_avgpool, nn.AdaptiveAvgPool1d):
                    avgpool = nn.AvgPool1d(**kwargs)
                elif isinstance(adaptive_avgpool, nn.AdaptiveAvgPool2d):
                    avgpool = nn.AvgPool2d(**kwargs)
                else:
                    assert isinstance(adaptive_avgpool, nn.AdaptiveAvgPool3d)
                    avgpool = nn.AvgPool3d(**kwargs)
                replace_module(model, adaptive_avgpool, avgpool)


class AvgPoolToQuantDepthwiseConv(PerInputModuleToModuleByHook):

    SUPPORTED_LAYERS = (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)

    def __init__(self, **conv_kwargs):
        super().__init__()
        self.conv_kwargs = conv_kwargs

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
            kwargs.update(**self.conv_kwargs)
            if isinstance(avgpool, nn.AvgPool1d):
                dw_conv = QuantConv1d(**kwargs)
            elif isinstance(avgpool, nn.AvgPool2d):
                dw_conv = QuantConv2d(**kwargs)
            else:
                assert isinstance(avgpool, nn.AvgPool3d)
                raise RuntimeError("QuantConv3d not supported yet.")
            kernel_value = 1. / reduce(mul, dw_conv.kernel_size)
            dw_conv.register_parameter(
                'scalar_weight', torch.nn.Parameter(torch.tensor(kernel_value)))
            weight_shape = dw_conv.weight.shape
            del dw_conv.weight
            # Attach property to instance by dynamically subclassing and assigning the subclass to the instance
            # Reference https://gist.github.com/Wilfred/49b0409c6489f1bdf5a5c98a488b31b5
            class_name = dw_conv.__class__.__name__ + 'FromAvgPool'
            child_class = type(
                class_name, (dw_conv.__class__,),
                {'weight': property(lambda self: self.scalar_weight.expand(weight_shape))})
            dw_conv.__class__ = child_class
            replace_module(model, avgpool, dw_conv)
