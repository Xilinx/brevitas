from abc import ABC
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from brevitas.core.function_wrapper.misc import Identity
from brevitas.inject import BaseInjector as Injector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase
from brevitas.quant_tensor import FloatQuantTensor
from brevitas.utils.quant_utils import _CachedIOFloat


class ActFloatQuantProxyFromInjectorBase(ActQuantProxyFromInjectorBase, ABC):

    def scale(self, force_eval=True):
        return self.retrieve_attribute('scale', force_eval)

    def zero_point(self, force_eval=True):
        return self.retrieve_attribute('zero_point', force_eval)

    def exponent_bit_width(self, force_eval=True):
        return self.retrieve_attribute('exponent_bit_width', force_eval)

    def mantissa_bit_width(self, force_eval=True):
        return self.retrieve_attribute('mantissa_bit_width', force_eval)

    def exponent_bias(self, force_eval=True):
        return self.retrieve_attribute('exponent_bias', force_eval)

    def is_saturating(self, force_eval=True):
        return self.retrieve_attribute('saturating', force_eval)

    def inf_values(self, force_eval=True):
        return self.retrieve_attribute('inf_values', force_eval)

    def nan_values(self, force_eval=True):
        return self.retrieve_attribute('nan_values', force_eval)

    @property
    def input_view_impl(self):
        if self.fused_activation_quant_proxy.tensor_quant is not None:
            return self.fused_activation_quant_proxy.tensor_quant.input_view_impl
        else:
            return Identity()

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


class ActFloatQuantProxyFromInjector(ActFloatQuantProxyFromInjectorBase):

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector):
        super().__init__(quant_layer, quant_injector)
        self.cache_class = _CachedIOFloat

    def create_quant_tensor(
            self,
            qt_args: Union[torch.Tensor, Tuple[Any]],
            x: Optional[FloatQuantTensor] = None) -> FloatQuantTensor:
        if isinstance(qt_args, tuple):
            out = FloatQuantTensor(*qt_args, signed=self.is_signed, training=self.training)
        else:
            out = FloatQuantTensor(
                qt_args,
                x.scale,
                x.zero_point,
                x.mantissa_bit_width,
                x.exponent_bit_width,
                x.exponent_bias,
                x.saturating,
                x.inf_values,
                x.nan_values,
                x.signed,
                self.training)
        return out


class DynamicActFloatQuantProxyFromInjector(ActFloatQuantProxyFromInjector):

    def scale(self, force_eval=True):
        raise RuntimeError("Scale for Dynamic Act Quant is input-dependant")

    def zero_point(self, force_eval=True):
        raise RuntimeError("Zero point for Dynamic Act Quant is input-dependant")
