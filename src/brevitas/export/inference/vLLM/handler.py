import torch

from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.function_wrapper.shape import dynamic_over_sub_channel_block_view
from brevitas.utils.quant_utils import groupwise_dequant_expand

from ..handler import DynamicScaleZeroPointMixin
from ..handler import GroupwiseFloatInferenceHandler
from ..handler import GroupwiseIntInferenceHandler


class StandaloneGroupwiseQuantMixin(DynamicScaleZeroPointMixin):

    def compute_scale(self, x, group_dim):
        scale = torch.clamp(torch.max(torch.abs(x), dim=group_dim, keepdim=True)[0], 1e-4)
        threshold = self.threshold
        if self.scaling_restriction == RestrictValueType.POWER_OF_TWO:
            scale = torch.clamp(torch.pow(2, torch.floor(torch.log2(scale))), 1e-7)
        if self.threshold_restriction == RestrictValueType.POWER_OF_TWO:
            threshold = torch.clamp(torch.pow(2, torch.floor(torch.log2(threshold))), 1e-7)
        scale = scale / threshold
        return scale


class vLLMGroupwiseIntInferenceHandler(GroupwiseIntInferenceHandler, StandaloneGroupwiseQuantMixin):

    def forward(self, x):
        inp_shape = x.shape
        x = dynamic_over_sub_channel_block_view(x, self.group_size, self.group_dim)
        scale = self.compute_scale(x, self.group_dim)
        zero_point = torch.zeros(()).type_as(x)
        out = self.inner_forward(x, scale, zero_point)
        out = groupwise_dequant_expand(out, scale, zero_point, self.group_dim, inp_shape)[0]
        return out


class vLLMGroupwiseFloatInferenceHandler(GroupwiseFloatInferenceHandler,
                                         StandaloneGroupwiseQuantMixin):

    def forward(self, x):
        inp_shape = x.shape
        x = dynamic_over_sub_channel_block_view(x, self.group_size, self.group_dim)
        scale = self.compute_scale(x, self.group_dim)
        zero_point = torch.zeros(()).type_as(x)
        out = self.inner_forward(x, scale, zero_point)
        out = groupwise_dequant_expand(out, scale, zero_point, self.group_dim, inp_shape)[0]
        return out
