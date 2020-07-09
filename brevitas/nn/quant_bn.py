from typing import Optional

import torch
from torch import nn

import brevitas.config as config
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.stats import StatsOp
#from brevitas.proxy.config import SCALING_MIN_VAL
from .quant_scale_bias import QuantScaleBias


def mul_add_from_bn(bn_mean, bn_var, bn_eps, bn_weight, bn_bias, affine_only):
    mul_factor = bn_weight
    add_factor = bn_bias * torch.sqrt(bn_var + bn_eps)
    add_factor = add_factor - bn_mean * (bn_weight - 1.0)
    if not affine_only:
        mul_factor = mul_factor / torch.sqrt(bn_var + bn_eps)
        add_factor = add_factor - bn_mean
        add_factor = add_factor / torch.sqrt(bn_var + bn_eps)
    return mul_factor, add_factor


class BatchNorm2dToQuantScaleBias(QuantScaleBias):

    def __init__(self,
                 num_features,
                 eps: float = 1e-5,
                 bias_quant_type: QuantType = QuantType.FP,
                 bias_narrow_range: bool = False,
                 bias_bit_width: int = None,
                 weight_quant_type: QuantType = QuantType.FP,
                 weight_quant_override: nn.Module = None,
                 weight_narrow_range: bool = False,
                 weight_scaling_override: Optional[nn.Module] = None,
                 weight_bit_width: int = 32,
                 weight_scaling_impl_type: ScalingImplType = ScalingImplType.STATS,
                 weight_scaling_const: Optional[float] = None,
                 weight_scaling_stats_op: StatsOp = StatsOp.MAX,
                 weight_scaling_per_output_channel: bool = False,
                 weight_restrict_scaling_type: RestrictValueType = RestrictValueType.LOG_FP,
                 weight_scaling_stats_sigma: float = 3.0,
          #       weight_scaling_min_val: float = SCALING_MIN_VAL,
                 compute_output_scale: bool = False,
                 compute_output_bit_width: bool = False,
                 return_quant_tensor: bool = False):
        super(BatchNorm2dToQuantScaleBias, self).__init__(num_features,
                                                          bias_quant_type,
                                                          bias_narrow_range,
                                                          bias_bit_width,
                                                          weight_quant_type,
                                                          weight_quant_override,
                                                          weight_narrow_range,
                                                          weight_scaling_override,
                                                          weight_bit_width,
                                                          weight_scaling_impl_type,
                                                          weight_scaling_const,
                                                          weight_scaling_stats_op,
                                                          weight_scaling_per_output_channel,
                                                          weight_restrict_scaling_type,
                                                          weight_scaling_stats_sigma,
                                                          weight_scaling_min_val,
                                                          compute_output_scale,
                                                          compute_output_bit_width,
                                                          return_quant_tensor)
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        running_mean_key = prefix + 'running_mean'
        running_var_key = prefix + 'running_var'
        num_batches_tracked_key = prefix + 'num_batches_tracked'

        if running_mean_key in state_dict and running_var_key in state_dict:
            weight_init, bias_init = mul_add_from_bn(bn_bias=state_dict[bias_key],
                                                     bn_weight=state_dict[weight_key],
                                                     bn_mean=state_dict[running_mean_key],
                                                     bn_var=state_dict[running_var_key],
                                                     bn_eps=self.eps,
                                                     affine_only=False)
            self.weight.data = weight_init
            self.bias.data = bias_init
            del state_dict[bias_key]
            del state_dict[weight_key]
            del state_dict[running_mean_key]
            del state_dict[running_var_key]
            del state_dict[num_batches_tracked_key]
        super(BatchNorm2dToQuantScaleBias, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                                       missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and bias_key in missing_keys:
            missing_keys.remove(bias_key)
        if config.IGNORE_MISSING_KEYS and weight_key in missing_keys:
            missing_keys.remove(weight_key)
        if num_batches_tracked_key in unexpected_keys:
            unexpected_keys.remove(num_batches_tracked_key)
