from typing import Optional, Union
from warnings import warn

import torch
from torch import Tensor
import torch.nn as nn

from brevitas.inject import BaseInjector as Injector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase
from brevitas.quant_tensor import FloatQuantTensor
from brevitas.quant_tensor.base_quant_tensor import QuantTensor
from brevitas.utils.quant_utils import _CachedIOFloat


class ActFloatQuantProxyFromInjectorBase(ActQuantProxyFromInjectorBase):

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

    def saturating(self, force_eval=True):
        return self.retrieve_attribute('saturating', force_eval)

    def inf_values(self, force_eval=True):
        return self.retrieve_attribute('inf_values', force_eval)

    def nan_values(self, force_eval=True):
        return self.retrieve_attribute('nan_values', force_eval)

    @property
    def is_ocp(self):
        is_e4m3 = self.mantissa_bit_width() == 3 and self.exponent_bit_width() == 4
        is_ocp_e4m3 = is_e4m3 and self.inf_values() is None and self.nan_values() == (('111',))

        is_e5m2 = self.mantissa_bit_width() == 5 and self.exponent_bit_width() == 2
        is_ocp_e5m2 = is_e5m2 and self.inf_values() == (
            ('00',)) and self.nan_values() == ('01', '11', '10')

        return is_ocp_e4m3 or is_ocp_e5m2

    @property
    def is_fnuz(self):
        is_e4m3 = self.mantissa_bit_width() == 3 and self.exponent_bit_width() == 4
        is_fnuz_e4m3 = is_e4m3 and self.inf_values() is None and self.nan_values(
        ) is None and self.exponent_bias() == 8

        is_e5m2 = self.mantissa_bit_width() == 5 and self.exponent_bit_width() == 2
        is_fnuz_e5m2 = is_e5m2 and self.inf_values() is None and self.nan_values(
        ) is None and self.exponent_bias() == 16
        return is_fnuz_e4m3 or is_fnuz_e5m2

    def forward(self, x: Union[Tensor, QuantTensor]) -> Union[Tensor, FloatQuantTensor]:
        out = x
        if self.fused_activation_quant_proxy is not None:
            y = x
            if isinstance(y, QuantTensor):
                y = y.value

            if self.export_mode:
                y = self.fused_activation_quant_proxy.activation_impl(y)
                y = self.export_handler(y)
            elif not self.is_quant_enabled:
                y = self.fused_activation_quant_proxy.activation_impl(y)
            else:
                y = self.fused_activation_quant_proxy(y)
            # If y is an empty FloatQuantTensor, we need to check if this is a passthrough proxy,
            # otherwise return a simple Tensor
            # We exclude the last two values (inf_values and nan_values)
            if isinstance(y, tuple) and not any(map(lambda f: f is None, y[:-2])):
                out = FloatQuantTensor(*y, signed=self.is_signed, training=self.training)
            elif self.is_passthrough_act:  # preserve scale/zp/bit/sign even without output quant
                if isinstance(y, tuple):
                    y = y[0]
                if isinstance(x, FloatQuantTensor):
                    out = FloatQuantTensor(
                        y,
                        x.scale,
                        x.zero_point,
                        x.mantissa_bit_width,
                        x.exponent_bit_width,
                        x.exponent_bias,
                        x.signed,
                        self.training,
                        x.saturating,
                        x.inf_values,
                        x.nan_values)
                else:
                    out = y
            else:
                if isinstance(y, tuple):
                    y = y[0]
                out = y
        else:
            # If fused activation quant proxy is not enabled, return the input
            out = x
        if not self.training and self.cache_inference_quant_act and isinstance(out,
                                                                               FloatQuantTensor):
            cached_out = _CachedIOFloat(out.detach(), self.cache_quant_io_metadata_only)
            self._cached_act = cached_out
        return out


class ActFloatQuantProxyFromInjector(ActFloatQuantProxyFromInjectorBase):

    def forward(self, x: Union[Tensor, FloatQuantTensor]) -> Union[Tensor, FloatQuantTensor]:
        out = x
        if self.fused_activation_quant_proxy is not None:
            y = x
            if isinstance(y, FloatQuantTensor):
                y = y.value

            if self.export_mode:
                y = self.fused_activation_quant_proxy.activation_impl(y)
                y = self.export_handler(y)
            elif not self.is_quant_enabled:
                y = self.fused_activation_quant_proxy.activation_impl(y)
            else:
                y = self.fused_activation_quant_proxy(y)
            # If y is an empty FloatQuantTensor, we need to check if this is a passthrough proxy,
            # otherwise return a simple Tensor
            # We exclude the last two values (inf_values and nan_values)
            if isinstance(y, tuple) and not any(map(lambda f: f is None, y[:-2])):
                out = FloatQuantTensor(*y, signed=self.is_signed, training=self.training)
            elif self.is_passthrough_act:  # preserve scale/zp/bit/sign even without output quant
                if isinstance(y, tuple):
                    y = y[0]
                if isinstance(x, FloatQuantTensor):
                    out = FloatQuantTensor(
                        y,
                        x.scale,
                        x.zero_point,
                        x.mantissa_bit_width,
                        x.exponent_bit_width,
                        x.exponent_bias,
                        x.signed,
                        self.training,
                        x.saturating,
                        x.inf_values,
                        x.nan_values)
                else:
                    out = y
            else:
                if isinstance(y, tuple):
                    y = y[0]
                out = y
        else:
            # If fused activation quant proxy is not enabled, return the input
            out = x
        if not self.training and self.cache_inference_quant_act and isinstance(out,
                                                                               FloatQuantTensor):
            cached_out = _CachedIOFloat(out.detach(), self.cache_quant_io_metadata_only)
            self._cached_act = cached_out
        return out
