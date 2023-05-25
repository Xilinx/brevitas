from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager

import torch
from torch.nn import Module

from brevitas.export.common.handler.base import BaseHandler
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_layer_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import _set_layer_export_mode
from brevitas.export.manager import BaseManager
from brevitas.nn import QuantLinear
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas_examples.llm.llm_quant.mlir_custom_mm import *  # noqa


class WeightBlockQuantHandlerBase(BaseHandler, ABC):
    handled_layer = WeightQuantProxyFromInjector

    def __init__(self):
        super(WeightBlockQuantHandlerBase, self).__init__()
        self.int_weight = None
        self.scale = None
        self.zero_point = None
        self.bit_width = None

    def scaling_impl(self, proxy_module):
        return proxy_module.tensor_quant.scaling_impl

    def zero_point_impl(self, proxy_module):
        return proxy_module.tensor_quant.zero_point_impl

    def bit_width_impl(self, proxy_module):
        return proxy_module.tensor_quant.msb_clamp_bit_width_impl

    def export_scale(self, proxy_module, bit_width):
        scaling_impl = self.scaling_impl(proxy_module)
        int_scaling_impl = proxy_module.tensor_quant.int_scaling_impl
        int_threshold = int_scaling_impl(bit_width)
        threshold = scaling_impl.wrapped_scaling_impl.stats_scaling_impl(
            scaling_impl.wrapped_scaling_impl.parameter_list_stats())
        return threshold / int_threshold

    def export_zero_point(self, proxy_module, scale, bit_width):
        zero_point_impl = self.zero_point_impl(proxy_module)
        return zero_point_impl.unexpanded_zero_point(scale, bit_width)

    @abstractmethod
    def prepare_for_export(self, module):
        pass

    @abstractmethod
    def forward(self, x):
        pass


class WeightBlockQuantProxyHandler(WeightBlockQuantHandlerBase):

    def __init__(self):
        super().__init__()
        self.expanded_scaling_shape = None
        self.reshaped_scaling_shape = None
        self.expanded_zero_point_shape = None
        self.reshaped_zero_point_shape = None

    def prepare_for_export(self, module):
        assert len(module.tracked_module_list) == 1, "Shared quantizers not supported."
        self.bit_width = self.bit_width_impl(module)()
        assert self.bit_width <= 8., "Only 8b or lower is supported."
        quant_layer = module.tracked_module_list[0]
        quant_weight = quant_layer.quant_weight()
        self.int_weight = quant_weight.int().detach()
        self.scale = self.export_scale(module, self.bit_width)
        self.expanded_scaling_shape = self.scaling_impl(module).expanded_scaling_shape
        self.reshaped_scaling_shape = self.scaling_impl(module).reshaped_scaling_shape
        if (quant_weight.zero_point != 0.).any():
            self.zero_point = self.export_zero_point(module, self.scale, self.bit_width)
            self.expanded_zero_point_shape = self.zero_point_impl(module).expanded_zero_point_shape
            self.reshaped_zero_point_shape = self.zero_point_impl(module).reshaped_zero_point_shape

    def forward(self, x):
        scale = self.scale.expand(self.expanded_scaling_shape).contiguous()
        # contiguous above is to avoid the reshape below being mapped to a unsafe view
        scale = scale.view(self.reshaped_scaling_shape)
        int_weight = self.int_weight
        if self.zero_point is not None:
            zero_point = self.zero_point.expand(self.expanded_zero_point_shape).contiguous()
            # contiguous above is to avoid the reshape below being mapped to a unsafe view
            zero_point = zero_point.view(self.reshaped_zero_point_shape)
            int_weight = int_weight - zero_point
        else:
            zero_point = torch.zeros_like(scale)
        quant_weight = int_weight * scale
        return quant_weight, scale, zero_point, self.bit_width


class LinearWeightBlockQuantHandler(WeightBlockQuantHandlerBase, ABC):
    handled_layer = QuantLinear

    def __init__(self):
        super(LinearWeightBlockQuantHandler, self).__init__()
        self.group_size = None

    def pack_int_weights(self, bit_width, int_weights):
        assert int_weights.dtype in [torch.int8, torch.uint8], "Packing requires (u)int8 input."
        if bit_width == 8:
            return int_weights
        elif bit_width == 4 or bit_width == 2:
            packed_int_weights = torch.zeros(
                (int_weights.shape[0], int_weights.shape[1] * bit_width // 8),
                device=int_weights.device,
                dtype=int_weights.dtype)
            i = 0
            for column in range(packed_int_weights.shape[1]):
                # Compared to the reference below we don't transpose the matrix and we pack into 8b data rather than 32b
                # https://github.com/qwopqwop200/GPTQ-for-LLaMa/blob/05781593c818d4dc8adc2d32c975e83d17d2b9a8/quant/quant_linear.py#L346
                for j in range(i, i + (8 // bit_width)):
                    shift_factor = (bit_width * (j - i))
                    packed_int_weights[:, column] |= int_weights[:, j] << shift_factor
                i += 8 // bit_width
            return packed_int_weights
        else:
            raise ValueError(f"Bit width {bit_width} not supported.")

    def prepare_for_export(self, module):
        self.bit_width = self.bit_width_impl(module.weight_quant)()
        assert self.bit_width <= 8., "Only 8b or lower is supported."
        quant_weight = module.quant_weight()
        self.bias = module.bias
        self.scale = self.export_scale(module.weight_quant, self.bit_width)
        if (quant_weight.zero_point != 0.).any():
            self.zero_point = self.export_zero_point(
                module.weight_quant, self.scale, self.bit_width)
        else:
            # if there is no zero-point, export zeroes in the shape of scale
            self.zero_point = torch.zeros_like(self.scale)
        self.group_size = module.weight_quant.quant_injector.block_size
        self.bit_width = int(self.bit_width.cpu().item())
        self.int_weights = self.pack_int_weights(self.bit_width, quant_weight.int().detach())

    @abstractmethod
    def forward(self, x):     
        pass


class BlockQuantProxyLevelManager(BaseManager):

    handlers = [WeightBlockQuantProxyHandler]

    @classmethod
    def set_export_handler(cls, module):
        _set_proxy_export_handler(cls, module)
        
        
def block_quant_layer_level_manager(export_handlers):
    
    class BlockQuantLayerLevelManager(BaseManager):
        handlers = export_handlers

        @classmethod
        def set_export_handler(cls, module):
            _set_layer_export_handler(cls, module)

    return BlockQuantLayerLevelManager


@contextmanager
def brevitas_proxy_export_mode(model, export_manager=BlockQuantProxyLevelManager):
    is_training = model.training
    model.eval()
    model.apply(export_manager.set_export_handler)
    _set_proxy_export_mode(model, enabled=True)
    try:
        yield model
    finally:
        _set_proxy_export_mode(model, enabled=False)
        model.train(is_training)
        
        
@contextmanager
def brevitas_layer_export_mode(model, export_manager):
    is_training = model.training
    model.eval()
    model.apply(export_manager.set_export_handler)
    _set_layer_export_mode(model, enabled=True)
    try:
        yield model
    finally:
        _set_layer_export_mode(model, enabled=False)
        model.train(is_training)
        
        
def replace_call_fn_target(graph_model, src, target):
    for node in graph_model.graph.nodes:
        if node.op == "call_function" and node.target is src:
            node.target = target
    graph_model.graph.lint()
    graph_model.recompile()



