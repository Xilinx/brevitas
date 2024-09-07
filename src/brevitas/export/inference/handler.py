from typing import Tuple

import torch

from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector


class IntInferencetHandler(torch.nn.Module):
    handled_layer = (
        ActQuantProxyFromInjector, WeightQuantProxyFromInjector, BiasQuantProxyFromInjector)

    def attach_debug_info(self, module):
        pass

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.scale = module.scale()
            self.zero_point = module.zero_point().to(self.scale.device)
            self.bit_width = module.bit_width()
            self.min_int = min_int(module.is_signed, module.is_narrow_range, self.bit_width)
            self.max_int = max_int(module.is_signed, module.is_narrow_range, self.bit_width)

    def quant(self, x):
        return torch.clamp(
            torch.round(x / self.scale + self.zero_point), self.min_int, self.max_int)

    def dequant(self, x):
        return (x - self.zero_point) * self.scale

    def forward(self, x, unused_scale=None) -> Tuple[torch.Tensor]:
        return self.dequant(self.quant(x)), self.scale, self.zero_point, self.bit_width
