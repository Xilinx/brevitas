from typing import Any, Optional, Tuple

from brevitas.proxy.float_runtime_quant import ActFloatQuantProxyFromInjectorBase
from brevitas.quant_tensor import GroupwiseFloatQuantTensor
from brevitas.utils.quant_utils import _CachedIOGroupwiseFloat


class GroupwiseActFloatQuantProxyFromInjector(ActFloatQuantProxyFromInjectorBase):

    def __init__(self, quant_layer, quant_injector):
        super().__init__(quant_layer, quant_injector)
        self.cache_class = _CachedIOGroupwiseFloat

    @property
    def group_dim(self):
        return self.quant_injector.group_dim

    @property
    def group_size(self):
        return self.quant_injector.group_size

    def create_quant_tensor(
            self,
            qt_args: Tuple[Any],
            x: Optional[GroupwiseFloatQuantTensor] = None) -> GroupwiseFloatQuantTensor:
        if x is None:
            value, scale, zero_point, exponent_bit_width, mantissa_bit_width, exponent_bias, saturating, inf_values, nan_values = qt_args
            out = GroupwiseFloatQuantTensor(
                value,
                scale,
                zero_point,
                self.group_size,
                self.group_dim,
                exponent_bit_width,
                mantissa_bit_width,
                exponent_bias,
                saturating,
                inf_values,
                nan_values,
                self.is_signed,
                self.training)
        else:
            out = GroupwiseFloatQuantTensor(
                qt_args,
                x.scale,
                x.zero_point,
                self.group_size,
                self.group_dim,
                x.exponent_bit_width,
                x.mantissa_bit_width,
                x.exponent_bias,
                x.saturating,
                x.inf_values,
                x.nan_values,
                x.signed,
                self.training)
        return out
