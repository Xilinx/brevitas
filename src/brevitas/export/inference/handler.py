import torch

from brevitas.export.common.handler.base import BaseHandler
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector


class InferenceWeightProxyHandler(BaseHandler):
    handled_layer = WeightQuantProxyFromInjector

    def __init__(self):
        super(InferenceWeightProxyHandler, self).__init__()
        self.int_weight = None
        self.scale = None
        self.zero_point = None
        self.bit_width = None
        self.dtype = None
        self.float_weight = None

    def prepare_for_export(self, module):
        assert len(module.tracked_module_list) == 1, "Shared quantizers not supported."
        quant_layer = module.tracked_module_list[0]
        quant_weight = quant_layer.quant_weight()
        self.int_weight = quant_weight.int().detach()
        self.scale = module.scale()
        self.zero_point = module.zero_point()
        self.bit_width = module.bit_width()
        self.float_weight = self.int_weight * self.scale

    def forward(self, x):
        return self.float_weight, self.scale, self.zero_point, self.bit_width
