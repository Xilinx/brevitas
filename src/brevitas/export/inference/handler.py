from typing import Tuple

import torch

from brevitas.function.ops import max_float, max_int
from brevitas.function.ops import min_int
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjector
from brevitas.proxy.float_runtime_quant import ActFloatQuantProxyFromInjector, ActFloatQuantProxyFromInjectorBase
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.utils.torch_utils import float_internal_scale


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
            self.min_clamp = min_int(module.is_signed, module.is_narrow_range, self.bit_width)
            self.max_clamp = max_int(module.is_signed, module.is_narrow_range, self.bit_width)

    def quant(self, x):
        return torch.clamp(
            torch.round(x / self.scale + self.zero_point), self.min_clamp, self.max_clamp)

    def dequant(self, x):
        return (x - self.zero_point) * self.scale

    def forward(self, x, unused_scale=None) -> Tuple[torch.Tensor]:
        return self.dequant(self.quant(x)), self.scale, self.zero_point, self.bit_width



class FloatInferencetHandler(IntInferencetHandler):
    handled_layer = (
        ActFloatQuantProxyFromInjector, WeightFloatQuantProxyFromInjector)

    def attach_debug_info(self, module):
        pass

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.scale = module.scale()
            self.zero_point = module.zero_point().to(self.scale.device)
            self.exponent_bit_width = module.exponent_bit_width()
            self.mantissa_bit_width = module.mantissa_bit_width()
            self.exponent_bias = module.exponent_bias()
            self.saturating = module.is_saturating()
            self.inf_values = module.inf_values()
            self.nan_values = module.nan_values()
            self.eps = torch.finfo(self.scale.dtype).tiny
            if hasattr(module.tensor_quant, 'float_to_int_impl'):
                self.float_to_int_impl = module.tensor_quant.float_to_int_impl
                self.float_clamp_impl = module.tensor_quant.float_clamp_impl
            elif hasattr(module, 'fused_activation_quant_proxy'):
                self.float_to_int_impl = module.fused_activation_quant_proxy.tensor_quant.float_to_int_impl
                self.float_clamp_impl = module.fused_activation_quant_proxy.tensor_quant.float_clamp_impl

            self.max_clamp = max_float(self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias)
            self.min_clamp = -self.max_clamp
            self.fp_internal_scale_min = 1. - self.exponent_bias - self.mantissa_bit_width
            self.max_value = max_float(self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias)
            self.min_value = torch.tensor(0.) if not module.is_signed else -self.max_value

    def quant(self, x):
        x = x/self.scale
        internal_scale = float_internal_scale(
            x, self.mantissa_bit_width, self.fp_internal_scale_min, self.eps)
        x = internal_scale * self.float_to_int_impl(x / internal_scale)
        x = self.float_clamp_impl.saturating_clamp(x, self.max_value, self.min_value)
        if not self.saturating:
            x = self.float_clamp_impl.inf_nan_clamp(x, self.max_value)
        
        return x

    def forward(self, x, unused_scale=None) -> Tuple[torch.Tensor]:
        return self.dequant(self.quant(x)), self.scale, self.zero_point, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias, self.saturating, self.inf_values, self.nan_values