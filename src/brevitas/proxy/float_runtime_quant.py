from typing import Optional, Union
from warnings import warn

import torch
from torch import Tensor
import torch.nn as nn

from brevitas.inject import BaseInjector as Injector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjectorBase
from brevitas.quant_tensor import FloatQuantTensor
from brevitas.utils.quant_utils import _CachedIOFloat


class ActFloatQuantProxyFromInjector(ActQuantProxyFromInjectorBase):

    def scale(self, force_eval=True):
        if self.is_quant_enabled:
            current_status = self.training
            if force_eval:
                self.eval()
            out = self.__call__(self._zero_hw_sentinel())
            self.train(current_status)
            return out.scale
        elif self._cached_act is not None:
            return self._cached_act.scale
        elif self._cached_act is None:
            return None

    def zero_point(self, force_eval=True):
        if self.is_quant_enabled:
            current_status = self.training
            if force_eval:
                self.eval()
            out = self.__call__(self._zero_hw_sentinel())
            self.train(current_status)
            return out.zero_point
        elif self._cached_act is not None:
            return self._cached_act.zero_point
        elif self._cached_act is None:
            return None

    def bit_width(self, force_eval=True):
        if self.is_quant_enabled:
            current_status = self.training
            if force_eval:
                self.eval()
            out = self.__call__(self._zero_hw_sentinel())
            self.train(current_status)
            return out.bit_width
        elif self._cached_act is not None:
            return self._cached_act.bit_width
        elif self._cached_act is None:
            return None

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
            if isinstance(y, tuple) and not any(map(lambda f: f is None, y)):
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
