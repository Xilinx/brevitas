from typing import Optional, Union

import torch
from torch import Tensor

from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.quant_tensor import GroupwiseIntQuantTensor


class GroupwiseWeightQuantProxyFromInjector(WeightQuantProxyFromInjector):

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

    def forward(self, x: torch.Tensor) -> Union[Tensor, GroupwiseIntQuantTensor]:
        if self.is_quant_enabled:
            impl = self.export_handler if self.export_mode else self.tensor_quant
            out, scale, zero_point, bit_width = impl(x)
            return GroupwiseIntQuantTensor(
                out,
                scale,
                zero_point,
                self.group_size,
                self.group_dim,
                bit_width,
                self.is_signed,
                self.training)
        else:  # quantization disabled
            return x
