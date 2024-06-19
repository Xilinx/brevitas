from typing import Union

from torch import Tensor

from brevitas.proxy.float_runtime_quant import ActFloatQuantProxyFromInjectorBase
from brevitas.quant_tensor import GroupwiseFloatQuantTensor
from brevitas.utils.quant_utils import _CachedIOGroupwiseFloat


class GroupwiseActFloatQuantProxyFromInjector(ActFloatQuantProxyFromInjectorBase):

    @property
    def group_dim(self):
        return self.quant_injector.group_dim

    @property
    def group_size(self):
        return self.quant_injector.group_size

    def forward(
            self, x: Union[Tensor,
                           GroupwiseFloatQuantTensor]) -> Union[Tensor, GroupwiseFloatQuantTensor]:
        out = x
        if self.fused_activation_quant_proxy is not None:
            y = x
            if isinstance(y, GroupwiseFloatQuantTensor):
                y = y.value

            if self.export_mode:
                y = self.fused_activation_quant_proxy.activation_impl(y)
                y = self.export_handler(y)
            elif not self.is_quant_enabled:
                y = self.fused_activation_quant_proxy.activation_impl(y)
            else:
                y = self.fused_activation_quant_proxy(y)
            # If y is an empty GroupwiseFloatQuantTensor, we need to check if this is a passthrough proxy,
            # otherwise return a simple Tensor
            # We exclude the last two values (inf_values and nan_values)
            if isinstance(y, tuple) and not any(map(lambda f: f is None, y[:-2])):
                out = GroupwiseFloatQuantTensor(*y, signed=self.is_signed, training=self.training)
            elif self.is_passthrough_act:  # preserve scale/zp/bit/sign even without output quant
                if isinstance(y, tuple):
                    y = y[0]
                if isinstance(x, GroupwiseFloatQuantTensor):
                    out = GroupwiseFloatQuantTensor(
                        y,
                        x.scale,
                        x.zero_point,
                        self.group_dim,
                        self.group_size,
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
        if not self.training and self.cache_inference_quant_act and isinstance(
                out, GroupwiseFloatQuantTensor):
            cached_out = _CachedIOGroupwiseFloat(out.detach(), self.cache_quant_io_metadata_only)
            self._cached_act = cached_out
        return out
