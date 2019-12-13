from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.nn import QuantSigmoid, QuantTanh
import torch.nn as nn
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType
from brevitas.core.scaling import ScalingImplType, SCALING_SCALAR_SHAPE
from brevitas.core.stats import StatsInputViewShapeImpl, StatsOp
from brevitas.proxy.parameter_quant import WeightQuantProxy, BiasQuantProxy, WeightReg, _weight_quant_init_impl
from brevitas.proxy.runtime_quant import _activation_quant_init_impl
from brevitas.core.quant import RescalingIntQuant, IdentityQuant
from brevitas.core.restrict_val import RestrictValueType, RestrictValue, FloatToIntImplType, RestrictValueOpImplType

from brevitas.nn.quant_layer import QuantLayer, SCALING_MIN_VAL
import torch

from typing import Union, Optional, Tuple, List, Callable
from torch import Tensor
from brevitas.quant_tensor import QuantTensor
from brevitas.core import ZERO_HW_SENTINEL_NAME, ZERO_HW_SENTINEL_VALUE
OVER_BATCH_OVER_CHANNELS_SHAPE = (1, -1, 1, 1)

__all__=['QuantLSTMCELL']



class QuantLSTMCELL(torch.jit.ScriptModule):
        def __init__(self, input_size, hidden_size, weight_config, activation_config,
                     compute_output_scale=False, compute_output_bit_width=False,
                     return_quant_tensor=False):

            super(QuantLSTMCELL, self).__init__()
            self.register_buffer(ZERO_HW_SENTINEL_NAME, torch.tensor(ZERO_HW_SENTINEL_VALUE))
            self.return_quant_tensor = return_quant_tensor
            self.weight_config = weight_config
            self.activation_config = activation_config
            self.quant_sigmoid = self.configure_activation(self.activation_config, QuantSigmoid)
            self.quant_tanh = self.configure_activation(self.activation_config, QuantTanh)

            self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
            self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
            self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

            if self.weight_config.get('weight_quant_type', QuantType.FP) == QuantType.FP and compute_output_bit_width:
                raise Exception("Computing output bit width requires enabling quantization")
            if self.weight_config.get('bias_quant_type', QuantType.FP) != QuantType.FP and not (compute_output_scale and compute_output_bit_width):
                raise Exception("Quantizing bias requires to compute output scale and output bit width")


            self.weight_config['weight_scaling_shape'] = SCALING_SCALAR_SHAPE
            self.weight_config['weight_stats_input_view_shape_impl'] = StatsInputViewShapeImpl.OVER_TENSOR
            self.weight_config['weight_scaling_stats_input_concat_dim'] = 1
            self.weight_config['weight_scaling_stats_reduce_dim'] = None
            self.weight_quant_hh, self.bias_quant_hh = self.configure_weight(self.weight_hh, self.weight_config)
            self.weight_quant_ih, self.bias_quant_ih = self.configure_weight(self.weight_ih, self.weight_config)


        @torch.jit.script_method
        def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]



            output_scale = None
            output_bit_width = None
            input, input_scale, input_bit_width = self.unpack_input(input)
            # quant_weight_ih, quant_weight_ih_scale, quant_weight_ih_bit_width = self.forward_weights(self.weight_ih, self.weight_quant_hh)
            # quant_weight_hh, quant_weight_hh_scale, quant_weight_hh_bit_width = self.forward_weights(self.weight_hh, self.weight_quant_ih)
            OBOCS = torch.tensor([1,-1,1,])
            zero_hw_sentinel = getattr(self, 'zero_hw_sentinel')
            out, scale, bit_width = self.weight_quant_hh(self.weight_ih, zero_hw_sentinel)
            quant_weight_ih, quant_weight_ih_scale, quant_weight_ih_bit_width = out, scale, bit_width

            out, scale, bit_width = self.weight_quant_ih(self.weight_hh, zero_hw_sentinel)
            quant_weight_hh, quant_weight_hh_scale, quant_weight_hh_bit_width = out, scale, bit_width


            # quant_bias_ih, quant_bias_ih_scale, quant_bias_ih_bit_width = self.bias_quant_ih(self.bias_ih)
            # quant_bias_hh, quant_bias_hh_scale, quant_bias_hh_bit_width = self.bias_quant_hh(self.bias_hh)


            # if self.compute_output_bit_width:
            #     output_ih_bit_width = self.max_output_bit_width(input_bit_width, quant_weight_ih_bit_width)
            #     output__hh_bit_width = self.max_output_bit_width(input_bit_width, quant_weight_hh_bit_width)
            # if self.compute_output_scale:
            #     output_scale_ih = input_scale * quant_weight_ih_scale
            #     output_scale_hh = input_scale * quant_weight_hh_scale
            # if self.weight_config.get('bias'):
            #     pass

            zero_hw_sentinel = getattr(self, 'zero_hw_sentinel')
            hx, cx = state
            gates = (torch.mm(input, quant_weight_ih.t()) + self.bias_ih +
                     torch.mm(hx, quant_weight_hh.t()) + self.bias_hh)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ingate,_,_ = self.quant_sigmoid(ingate,zero_hw_sentinel)
            forgetgate,_,_ = self.quant_sigmoid(forgetgate,zero_hw_sentinel)
            cellgate,_,_ = self.quant_tanh(cellgate,zero_hw_sentinel)
            outgate,_,_ = self.quant_sigmoid(outgate,zero_hw_sentinel)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * self.quant_tanh(cy,zero_hw_sentinel)[0]

            return hy, (hy, cy)



        def max_output_bit_width(self, input_bit_width, weight_bit_width):
            pass
            #
            # max_uint_input = max_uint(bit_width=input_bit_width, narrow_range=False)
            # max_kernel_val = self.weight_quant.tensor_quant.int_quant.max_uint(weight_bit_width)
            # group_size = self.out_channels // self.groups
            # max_uint_output = max_uint_input * max_kernel_val * self.kernel_size[0] * group_size
            # max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
            # return max_output_bit_width

        def unpack_input(self, input):
            if isinstance(input, QuantTensor):
                return input
            else:
                return input, None, None

        def pack_output(self,
                        output,
                        output_scale,
                        output_bit_width):
            if self.return_quant_tensor:
                return QuantTensor(tensor=output, scale=output_scale, bit_width=output_bit_width)
            else:
                return output

        def configure_weight(self, weight, weight_config):
            zero_hw_sentinel = getattr(self, 'zero_hw_sentinel')
            wqp: IdentityQuant = _weight_quant_init_impl(bit_width=weight_config.get('weight_bit_width',8),
                                     quant_type=weight_config.get('weight_quant_type', QuantType.FP),
                                     narrow_range=weight_config.get('weight_narrow_range', True),
                                     scaling_override=weight_config.get('weight_scaling_override', None),
                                     restrict_scaling_type=weight_config.get('weight_restrict_scaling_type', RestrictValueType.LOG_FP),
                                     scaling_const=weight_config.get('weight_scaling_const', None),
                                     scaling_stats_op=weight_config.get('weight_scaling_stats_op', StatsOp.MAX),
                                     scaling_impl_type=weight_config.get('weight_scaling_impl_type', ScalingImplType.STATS),
                                     scaling_stats_reduce_dim=weight_config.get('weight_scaling_stats_reduce_dim', None),
                                     scaling_shape=weight_config.get('weight_scaling_shape', SCALING_SCALAR_SHAPE),
                                     bit_width_impl_type=weight_config.get('weight_bit_width_impl_type', BitWidthImplType.CONST),
                                     bit_width_impl_override=weight_config.get('weight_bit_width_impl_override', None),
                                     restrict_bit_width_type=weight_config.get('weight_restrict_bit_width_type', RestrictValueType.INT),
                                     min_overall_bit_width=weight_config.get('weight_min_overall_bit_width', 2),
                                     max_overall_bit_width=weight_config.get('weight_max_overall_bit_width', None),
                                     ternary_threshold=weight_config.get('weight_ternary_threshold', 0.5),
                                     scaling_stats_input_view_shape_impl=weight_config.get('weight_stats_input_view_shape_impl', StatsInputViewShapeImpl.OVER_TENSOR),
                                     scaling_stats_input_concat_dim=weight_config.get('weight_scaling_stats_input_concat_dim', 1),
                                     scaling_stats_sigma=weight_config.get('weight_scaling_stats_sigma', 3.0),
                                     scaling_min_val=weight_config.get('weight_scaling_min_val', SCALING_MIN_VAL),
                                     override_pretrained_bit_width=weight_config.get('weight_override_pretrained_bit_width', False),
                                     tracked_parameter_list=weight,
                                     zero_hw_sentinel=zero_hw_sentinel)
            bqp = BiasQuantProxy(quant_type=weight_config.get('bias_quant_type', QuantType.FP) ,
                                     bit_width=weight_config.get('bias_bit_width', 8),
                                     narrow_range=weight_config.get('bias_narrow_range', True))
            return wqp, bqp

        def configure_activation(self, activation_config, activation_func = QuantSigmoid):
            signed = True
            min_val = -1
            max_val = 1
            if activation_func == QuantTanh:
                activation_impl = nn.Tanh()
                min_val = -1
                signed = True
            elif activation_func==QuantSigmoid:
                activation_impl = nn.Sigmoid()
                min_val = 0
                signed = False

            activation_object = _activation_quant_init_impl(activation_impl=activation_impl,
                                       bit_width=activation_config.get('bit_width', 8),
                                       narrow_range=activation_config.get('narrow_range', True),
                                       quant_type=activation_config.get('quant_type', QuantType.FP),
                                       float_to_int_impl_type=activation_config.get('float_to_int_impl_type', FloatToIntImplType.ROUND),
                                       min_overall_bit_width=activation_config.get('min_overall_bit_width', 2),
                                       max_overall_bit_width=activation_config.get('max_overall_bit_width', None),
                                       bit_width_impl_override=activation_config.get('bit_width_impl_override', None),
                                       bit_width_impl_type=activation_config.get('bit_width_impl_type', BitWidthImplType.CONST),
                                       restrict_bit_width_type=activation_config.get('restrict_bit_width_type', RestrictValueType.INT),
                                       restrict_scaling_type=activation_config.get('restrict_scaling_type', RestrictValueType.LOG_FP),
                                       scaling_min_val=activation_config.get('scaling_min_val', SCALING_MIN_VAL),
                                       override_pretrained_bit_width=activation_config.get('override_pretrained_bit_width', False),
                                       min_val=activation_config.get('min_val',min_val),
                                       max_val=activation_config.get('max_val', max_val),
                                       signed=activation_config.get('signed', signed),
                                       per_channel_broadcastable_shape= activation_config.get('per_channel_broadcastable_shape',None),
                                       scaling_per_channel=activation_config.get('scaling_per_channel',False),
                                       scaling_override=activation_config.get('scaling_override',None),
                                       scaling_impl_type=activation_config.get('scaling_impl_type',ScalingImplType.CONST),
                                       scaling_stats_sigma=activation_config.get('scaling_stats_sigma',None),
                                       scaling_stats_input_view_shape_impl=activation_config.get('scaling_stats_input_view_shape_impl',None),
                                       scaling_stats_op= activation_config.get('scaling_stats_op',None),
                                       scaling_stats_buffer_momentum=activation_config.get('scaling_stats_buffer_momentum',None),
                                       scaling_stats_permute_dims=activation_config.get('scaling_stats_permute_dims',None))

            if activation_config.get('bit_width_impl_type', BitWidthImplType.CONST) == BitWidthImplType.PARAMETER:
                return activation_object
            else:
                return torch.jit.script(activation_object)

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

            ### TO DO : To be fixed cause self.scaling_impl_type ###
            scaling_impl_key = prefix + 'fused_activation_quant_proxy.tensor_quant.scaling_impl'
            runtime_stats_key = scaling_impl_key + '.runtime_stats'
            running_stats_key = scaling_impl_key + '.runtime_stats.running_stats'
            scaling_parameter_key = scaling_impl_key + '.learned_value'
            scaling_affine_weight_key = prefix + '.stats_scaling_impl.affine_rescaling.affine_weight'
            scaling_affine_bias_key = prefix + '.stats_scaling_impl.affine_rescaling.affine_bias'

            if not isinstance(self.fused_activation_quant_proxy.tensor_quant, IdentityQuant) and \
                self.scaling_impl_type == ScalingImplType.PARAMETER:
                scaling_impl = self.fused_activation_quant_proxy.tensor_quant.scaling_impl

                # If it's retrained directly from statistics, i.e. there isn't a preexisting parameter
                if running_stats_key in state_dict and not scaling_parameter_key in state_dict:
                    scaling_init = state_dict[running_stats_key]
                    if scaling_affine_weight_key in state_dict:
                        scaling_init *= state_dict[scaling_affine_weight_key]
                    if scaling_affine_bias_key in state_dict:
                        scaling_init += state_dict[scaling_affine_bias_key]

                    scaling_init = scaling_init.abs()

                    # Preprocess scaling init, which is always in FP range, based on current value restrictions
                    restrict_value_type = scaling_impl.restrict_value.restrict_value_type
                    restrict_value_init_op = scaling_impl.restrict_value.restrict_value_op(restrict_value_type,
                                                                                           RestrictValueOpImplType.TORCH_FN)
                    scaling_init = restrict_value_init_op(scaling_init)

                    # Put scaling init in place in the dict for parameter
                    if self.scaling_impl_type == ScalingImplType.PARAMETER:
                        state_dict[scaling_parameter_key] = scaling_init

                # Get rid of statistics after using them or in case there is already a parameter
                for k in list(state_dict.keys()):
                    if k.startswith(runtime_stats_key):
                        del state_dict[k]
            super(QuantLSTMCELL, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs)
            zero_hw_sentinel_key = prefix + ZERO_HW_SENTINEL_NAME
            if zero_hw_sentinel_key in missing_keys:
                missing_keys.remove(zero_hw_sentinel_key)
            if zero_hw_sentinel_key in unexpected_keys:  # for retrocompatibility with when it wasn't removed
                unexpected_keys.remove(zero_hw_sentinel_key)

        def state_dict(self, destination=None, prefix='', keep_vars=False):
            output_dict = super(QuantLSTMCELL, self).state_dict(destination, prefix, keep_vars)
            del output_dict[prefix + ZERO_HW_SENTINEL_NAME]
            return output_dict

        def forward_weights(self, weight: torch.Tensor, tensor_quant: Callable[[torch.Tensor, torch.Tensor], Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            zero_hw_sentinel = getattr(self, 'zero_hw_sentinel')
            out, scale, bit_width = tensor_quant(weight, zero_hw_sentinel)
            reshaped_scale = scale.view(OVER_BATCH_OVER_CHANNELS_SHAPE)
            return out, reshaped_scale, bit_width

