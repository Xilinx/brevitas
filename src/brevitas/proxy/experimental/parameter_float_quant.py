from typing import Union

import torch
from torch import Tensor

from brevitas.proxy.experimental.base_float_quant import QuantFloatProxyFromInjector
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from brevitas.proxy.parameter_quant import ParameterQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.quant_tensor import QuantTensor


class WeightFloatQuantProxyFromInjector(ParameterQuantProxyFromInjector,
                                        QuantFloatProxyFromInjector):

    @property
    def tracked_parameter_list(self):
        return [m.weight for m in self.tracked_module_list if m.weight is not None]

    @property
    def requires_quant_input(self):
        return False

    def scale(self):
        if not self.is_quant_enabled:
            return None
        scale = self.__call__(self.tracked_parameter_list[0]).scale
        return scale

    def zero_point(self):
        if not self.is_quant_enabled:
            return None
        zero_point = self.__call__(self.tracked_parameter_list[0]).zero_point
        return zero_point

    def bit_width(self):
        if not self.is_quant_enabled:
            return None
        bit_width = self.__call__(self.tracked_parameter_list[0]).bit_width
        return bit_width

    def forward(self, x: torch.Tensor) -> Union[Tensor, QuantTensor]:
        if self.is_quant_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            out, scale, zero_point, bit_width = impl(x)
            return QuantTensor(out, scale, zero_point, bit_width, self.is_signed, self.training)
        else:  # quantization disabled
            return x


class BiasFloatQuantProxyFromInjector(BiasQuantProxyFromInjector, QuantFloatProxyFromInjector):
    pass
