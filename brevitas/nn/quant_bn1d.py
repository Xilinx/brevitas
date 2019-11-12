import torch
import torch.nn as nn

import brevitas.config as config
from brevitas.core import ZERO_HW_SENTINEL_NAME
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType, RestrictValue, RestrictValueOpImplType, FloatToIntImplType
from brevitas.core.scaling import ScalingImplType, StandaloneScaling
from brevitas.proxy.parameter_quant import BiasQuantProxy, OVER_BATCH_OVER_CHANNELS_SHAPE
from .quant_layer import QuantLayer
from brevitas.nn.quant_bn import  mul_add_from_bn

__all__ = ['QuantBatchNorm1d']
# def mul_add_from_bn(bn_mean, bn_var, bn_eps, bn_weight, bn_bias, affine_only):
#     mul_factor = bn_weight
#     add_factor = bn_bias * torch.sqrt(bn_var + bn_eps)
#     add_factor = add_factor - bn_mean * (bn_weight - 1.0)
#     if not affine_only:
#         mul_factor = mul_factor / torch.sqrt(bn_var + bn_eps)
#         add_factor = add_factor - bn_mean
#         add_factor = add_factor / torch.sqrt(bn_var + bn_eps)
#     return mul_factor, add_factor



class QuantBatchNorm1d(QuantLayer, nn.Module):

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 restrict_value_type: RestrictValueType = RestrictValueType.FP,
                 impl_type: ScalingImplType = ScalingImplType.STATS,
                 bias_quant_type: QuantType = QuantType.FP,
                 bias_narrow_range: bool = False,
                 bias_bit_width: int = None):
        QuantLayer.__init__(self,
                            compute_output_scale=False,
                            compute_output_bit_width=False,
                            return_quant_tensor=False)
        nn.Module.__init__(self)

        if bias_quant_type != QuantType.FP and not (self.compute_output_scale and self.compute_output_bit_width):
            raise Exception("Quantizing bias requires to compute output scale and output bit width")

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        if impl_type == ScalingImplType.PARAMETER_FROM_STATS:
            self.running_mean = None
            self.running_var = None
        elif impl_type == ScalingImplType.STATS:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            raise Exception("Scaling mode not supported")

        self.eps = eps
        self.momentum = momentum
        self.impl_type = impl_type
        self.num_features = num_features
        self.restrict_value_type = restrict_value_type
        self.bias_quant = BiasQuantProxy(quant_type=bias_quant_type,
                                         narrow_range=bias_narrow_range,
                                         bit_width=bias_bit_width)
        self.restrict_weight = RestrictValue(restrict_value_type=restrict_value_type,
                                             float_to_int_impl_type=FloatToIntImplType.ROUND,
                                             min_val=None)
        self.restrict_scaling_preprocess = RestrictValue.restrict_value_op(restrict_value_type,
                                                                           restrict_value_op_impl_type=
                                                                           RestrictValueOpImplType.TORCH_MODULE)

    def stats(self, input_tensor):
        input_size = input_tensor.size()
        if input_size == 2:
            batch_size, channels = input_size
            height = 0
            input_tensor = input_tensor.permute(1, 0, 2).contiguous().view(channels, -1)
        elif input_size == 3:
            batch_size, channels, height = input_size
            input_tensor = input_tensor.permute(1, 0).contiguous().view(channels, -1)
        num_elems = input_tensor.shape[1]
        sum_ = input_tensor.sum(1)
        sum_of_square = input_tensor.pow(2).sum(1)
        mean = sum_ / num_elems
        sumvar = sum_of_square - sum_ * mean
        unbias_var = sumvar / (num_elems - 1)
        bias_var = sumvar / num_elems
        return mean, unbias_var, bias_var

    def forward(self, quant_tensor):
        output_scale = None
        output_bit_width = None
        input_tensor, input_scale, input_bit_width = self.unpack_input(quant_tensor)
        if self.impl_type == ScalingImplType.STATS:
            if self.training:
                mean, unbias_var, var = self.stats(input_tensor)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * unbias_var.detach()
            else:
                mean = self.running_mean
                var = self.running_var
            weight, bias = mul_add_from_bn(bn_mean=mean, bn_var=var, bn_weight=self.weight,
                                           bn_bias=self.bias, bn_eps=self.eps, affine_only=False)
            weight = weight.view(OVER_BATCH_OVER_CHANNELS_SHAPE)
            sign = weight.sign()
            weight = self.restrict_scaling_preprocess(weight.abs())
            bias = bias.view(OVER_BATCH_OVER_CHANNELS_SHAPE)
        else:  # ScalingImplType.PARAMETER_FROM_STATS
            weight = self.weight.view(OVER_BATCH_OVER_CHANNELS_SHAPE)
            bias = self.bias.view(OVER_BATCH_OVER_CHANNELS_SHAPE)
        weight = self.restrict_weight(weight)
        if self.impl_type == ScalingImplType.STATS:
            weight = sign * weight
        bias = self.bias_quant(bias, output_scale, output_bit_width)
        output_tensor = input_tensor * weight + bias
        return self.pack_output(output_tensor, output_scale, output_bit_width)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        running_mean_key = prefix + 'running_mean'
        running_var_key = prefix + 'running_var'
        num_batches_tracked_key = prefix + 'num_batches_tracked'

        # If it's converting a FP BN into weight/bias impl
        if self.impl_type == ScalingImplType.PARAMETER_FROM_STATS \
                and running_mean_key in state_dict and running_var_key in state_dict:
            weight_init, bias_init = mul_add_from_bn(bn_bias=state_dict[bias_key],
                                                     bn_weight=state_dict[weight_key],
                                                     bn_mean=state_dict[running_mean_key],
                                                     bn_var=state_dict[running_var_key],
                                                     bn_eps=self.eps,
                                                     affine_only=False)
            restrict_op = RestrictValue.restrict_value_op(restrict_value_type=self.restrict_value_type,
                                                          restrict_value_op_impl_type=RestrictValueOpImplType.TORCH_FN)
            self.weight_sign = torch.sign(weight_init.data)
            weight_init = weight_init.detach().clone().abs().data
            self.weight.data = restrict_op(weight_init)
            self.bias.data = bias_init.detach().clone().data
            del state_dict[bias_key]
            del state_dict[weight_key]
            del state_dict[running_mean_key]
            del state_dict[running_var_key]
            del state_dict[num_batches_tracked_key]
        super(QuantBatchNorm1d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and bias_key in missing_keys:
            missing_keys.remove(bias_key)
        if config.IGNORE_MISSING_KEYS and weight_key in missing_keys:
            missing_keys.remove(weight_key)
        if num_batches_tracked_key in unexpected_keys:
            unexpected_keys.remove(num_batches_tracked_key)

if __name__ == '__main__':
    import random
    SEED=123456
    random.seed(SEED)

