import torch

from brevitas.core.function_wrapper.misc import Identity
from brevitas.core.function_wrapper.shape import dynamic_over_sub_channel_block_view
from brevitas.core.function_wrapper.shape import OverOutputFeaturesView
from brevitas.core.function_wrapper.shape import PermuteDims
from brevitas.core.restrict_val import RestrictValueType
from brevitas.function.shape import over_output_features
from brevitas.utils.quant_utils import groupwise_dequant_expand

from ..handler import DynamicIntInferenceHandler
from ..handler import DynamicScaleZeroPointMixin
from ..handler import GroupwiseFloatInferenceHandler
from ..handler import GroupwiseIntInferenceHandler

EPS = 1e-6


class StandaloneGroupwiseQuantMixin(DynamicScaleZeroPointMixin):

    def compute_scale(self, x, group_dim):
        scale = torch.clamp(torch.max(torch.abs(x), dim=group_dim, keepdim=True)[0], EPS)
        threshold = self.threshold
        if self.scaling_restriction == RestrictValueType.POWER_OF_TWO:
            scale = torch.pow(2, torch.floor(torch.log2(scale)))
        if self.threshold_restriction == RestrictValueType.POWER_OF_TWO:
            threshold = torch.clamp(torch.pow(2, torch.floor(torch.log2(threshold))), EPS)
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


class vLLMDynamicPerRowIntInferenceHandler(DynamicScaleZeroPointMixin, DynamicIntInferenceHandler):

    def __init__(self):
        super().__init__()
        self.register_buffer("permute_dims", torch.ones(()))

    def prepare_for_export(self, module):
        super().prepare_for_export(module)
        for name, submodule in module.named_submodules():
            if 'scaling_stats_input_view_shape_impl' in name:
                assert type(submodule) == OverOutputFeaturesView, "Only per-row dynamic quantization is supported"
                if hasattr(submodule, 'permute_dims'):
                    self.permute_dims = submodule.permute_dims
                    self.permute_impl = PermuteDims(submodule.permute_dims)
                else:
                    self.permute_impl = Identity()

    def dynamic_broadcast(self, x, shape):
        return x.view(*shape[:-1], 1)

    def forward(self, x):
        x = self.permute_impl(x)
        x_shape = over_output_features(x)
        scale = self.compute_scale(x.reshape(x_shape), self.group_dim)
        scale = self.dynamic_broadcast(scale, x.shape)
        zero_point = torch.zeros(()).type_as(x)
        out = self.forward(x, scale, zero_point)
        return out
