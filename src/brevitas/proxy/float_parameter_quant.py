from abc import ABC
from typing import Any, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn

from brevitas.core.function_wrapper.misc import Identity
from brevitas.inject import BaseInjector as Injector
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjectorBase
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
from brevitas.quant_tensor import FloatQuantTensor
from brevitas.utils.quant_utils import _CachedIOFloat


class WeightFloatQuantProxyFromInjectorBase(WeightQuantProxyFromInjectorBase, ABC):

    def bit_width(self):
        if not self.is_quant_enabled:
            return None
        x = self.__call__(self.tracked_parameter_list[0])
        bit_width = x.mantissa_bit_width + x.exponent_bit_width + 1
        return bit_width

    def scale(self):
        return self.retrieve_attribute('scale')

    def zero_point(self):
        return self.retrieve_attribute('zero_point')

    def exponent_bit_width(self):
        return self.retrieve_attribute('exponent_bit_width')

    def mantissa_bit_width(self):
        return self.retrieve_attribute('mantissa_bit_width')

    def exponent_bias(self):
        return self.retrieve_attribute('exponent_bias')

    def is_saturating(self):
        return self.retrieve_attribute('saturating')

    def inf_values(self):
        return self.retrieve_attribute('inf_values')

    def nan_values(self):
        return self.retrieve_attribute('nan_values')

    @property
    def is_ocp(self):
        is_e4m3 = self.mantissa_bit_width() == 3 and self.exponent_bit_width() == 4
        is_ocp_e4m3 = is_e4m3 and self.inf_values() is None and self.nan_values() == (('111',))

        is_e5m2 = self.mantissa_bit_width() == 2 and self.exponent_bit_width() == 5
        is_ocp_e5m2 = is_e5m2 and self.inf_values() == (
            ('00',)) and self.nan_values() == ('01', '11', '10')

        return is_ocp_e4m3 or is_ocp_e5m2

    @property
    def is_fnuz(self):
        is_e4m3 = self.mantissa_bit_width() == 3 and self.exponent_bit_width() == 4
        is_fnuz_e4m3 = is_e4m3 and self.inf_values() is None and self.nan_values(
        ) is None and self.exponent_bias() == 8

        is_e5m2 = self.mantissa_bit_width() == 2 and self.exponent_bit_width() == 5
        is_fnuz_e5m2 = is_e5m2 and self.inf_values() is None and self.nan_values(
        ) is None and self.exponent_bias() == 16
        return is_fnuz_e4m3 or is_fnuz_e5m2

    @property
    def input_view_impl(self):
        if self.tensor_quant is not None:
            return self.tensor_quant.input_view_impl
        else:
            return Identity()


class WeightFloatQuantProxyFromInjector(WeightFloatQuantProxyFromInjectorBase):

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        super().__init__(quant_layer, quant_injector)
        self.cache_class = _CachedIOFloat

    def create_quant_tensor(self, qt_args: Tuple[Any]) -> FloatQuantTensor:
        out, scale, zero_point, exponent_bit_width, mantissa_bit_width, exponent_bias, saturating, inf_values, nan_values = qt_args
        return FloatQuantTensor(
            out,
            scale,
            zero_point,
            exponent_bit_width,
            mantissa_bit_width,
            exponent_bias,
            saturating,
            inf_values,
            nan_values,
            self.is_signed,
            self.training)


class BiasFloatQuantProxyFromInjector(BiasQuantProxyFromInjectorBase):

    def scale(self):
        if not self.is_quant_enabled:
            return None
        if self.requires_input_scale:
            cache = self.get_cached('scale')
            return cache
        zhs = self._zero_hw_sentinel()
        scale = self.__call__(self.tracked_parameter_list[0], zhs).scale
        return scale

    def zero_point(self):
        if not self.is_quant_enabled:
            return None
        zhs = self._zero_hw_sentinel()
        zero_point = self.__call__(self.tracked_parameter_list[0], zhs).zero_point
        return zero_point

    def exponent_bit_width(self):
        if not self.is_quant_enabled:
            return None
        exponent_bit_width = self.__call__(self.tracked_parameter_list[0]).exponent_bit_width
        return exponent_bit_width

    def mantissa_bit_width(self):
        if not self.is_quant_enabled:
            return None
        mantissa_bit_width = self.__call__(self.tracked_parameter_list[0]).mantissa_bit_width
        return mantissa_bit_width

    def exponent_bias(self):
        if not self.is_quant_enabled:
            return None
        exponent_bias = self.__call__(self.tracked_parameter_list[0]).exponent_bias
        return exponent_bias

    def is_saturating(self):
        if not self.is_quant_enabled:
            return None
        saturating = self.__call__(self.tracked_parameter_list[0]).saturating
        return saturating

    def inf_values(self):
        if not self.is_quant_enabled:
            return None
        inf_values = self.__call__(self.tracked_parameter_list[0]).inf_values
        return inf_values

    def nan_values(self):
        if not self.is_quant_enabled:
            return None
        nan_values = self.__call__(self.tracked_parameter_list[0]).nan_values
        return nan_values

    def forward(self,
                x: Tensor,
                input_scale: Optional[Tensor] = None) -> Union[Tensor, FloatQuantTensor]:
        out = x
        if self.is_quant_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            if self.requires_input_scale and input_scale is None:
                input_scale = self.scale()
                if input_scale is None:
                    raise RuntimeError("Input scale required")

            if self.requires_input_scale:
                input_scale = input_scale.view(-1)
                out, scale, zero_point, exponent_bit_width, mantissa_bit_width, exponent_bias, saturating, inf_values, nan_values = impl(x, input_scale)
            else:
                out, scale, zero_point, exponent_bit_width, mantissa_bit_width, exponent_bias, saturating, inf_values, nan_values = impl(x)

            out = FloatQuantTensor(
                out,
                scale,
                zero_point,
                exponent_bit_width,
                mantissa_bit_width,
                exponent_bias,
                saturating,
                inf_values,
                nan_values,
                self.is_signed,
                self.training)
        else:
            out = x
        if isinstance(out,
                      FloatQuantTensor) and not self.training and self.cache_inference_quant_bias:
            cached_bias = _CachedIOFloat(out.detach(), metadata_only=False)
            self._cached_bias = cached_bias
        return out
