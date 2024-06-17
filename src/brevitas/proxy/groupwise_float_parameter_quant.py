from typing import Optional, Union

import torch
from torch import Tensor

from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjectorBase
from brevitas.quant_tensor import GroupwiseFloatQuantTensor


class GroupwiseWeightFloatQuantProxyFromInjector(WeightFloatQuantProxyFromInjectorBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Is this always generated?
        self.view_impl = self.quant_injector.scaling_stats_input_view_shape_impl

    @property
    def group_dim(self):
        return self.quant_injector.group_dim

    @property
    def group_size(self):
        return self.quant_injector.group_size

    def forward(self, x: torch.Tensor) -> Union[Tensor, GroupwiseFloatQuantTensor]:
        if self.is_quant_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            x = self.view_impl(x)
            out, scale, zero_point, exponent_bit_width, mantissa_bit_width, exponent_bias, saturating, inf_values, nan_values = impl(x)
            return GroupwiseFloatQuantTensor(
                out,
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
        else:  # quantization disabled
            return x
