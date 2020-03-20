# From PyTorch:
#
# Copyright (c) 2019-     Xilinx, Inc              (Giuseppe Franco)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# From Caffe2:
#
# Copyright (c) 2016-present, Facebook Inc. All rights reserved.
#
# All contributions by Facebook:
# Copyright (c) 2016 Facebook Inc.
#
# All contributions by Google:
# Copyright (c) 2015 Google Inc.
# All rights reserved.
#
# All contributions by Yangqing Jia:
# Copyright (c) 2015 Yangqing Jia
# All rights reserved.
#
# All contributions from Caffe:
# Copyright(c) 2013, 2014, 2015, the respective contributors
# All rights reserved.
#
# All other contributions:
# Copyright(c) 2015, 2016 the respective contributors
# All rights reserved.
#
# Caffe2 uses a copyright model similar to Caffe: each contributor holds
# copyright over their contributions to Caffe2. The project versioning records
# all such contribution and copyright details. If a contributor wants to further
# mark their specific copyright on a particular contribution, they should
# indicate their copyright solely in the commit message of the change when it is
# committed.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
# and IDIAP Research Institute nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from brevitas.core.bit_width import BitWidthImplType
from brevitas.nn import QuantSigmoid, QuantTanh, QuantHardTanh
import torch.nn as nn
from brevitas.core.scaling import ScalingImplType, SCALING_SCALAR_SHAPE
from brevitas.core.stats import StatsInputViewShapeImpl, StatsOp
from brevitas.proxy.parameter_quant import _weight_quant_init_impl
from brevitas.proxy.runtime_quant import _activation_quant_init_impl
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType
from brevitas.quant_tensor import QuantTensor
from brevitas.core.function_wrapper import ConstScalarClamp

from brevitas.nn.quant_layer import SCALING_MIN_VAL
import torch

from typing import Tuple, List
from torch import Tensor
from brevitas.core import ZERO_HW_SENTINEL_NAME, ZERO_HW_SENTINEL_VALUE
from collections import namedtuple, OrderedDict

OVER_BATCH_OVER_CHANNELS_SHAPE = (1, -1, 1, 1)
LSTMState = namedtuple('LSTMState', ['hx', 'cx'])

__all__ = ['QuantLSTMLayer', 'BidirLSTMLayer']

@torch.jit.script
def reverse(lst):
    # type: (List[Tensor]) -> List[Tensor]
    out = torch.jit.annotate(List[Tensor], [])
    start = len(lst) - 1
    end = -1
    step = -1
    for i in range(start, end, step):
        out += [lst[i]]
    return out


class QuantLSTMLayer(torch.jit.ScriptModule):
    __constants__ = ['reverse_input']

    def __init__(self, input_size, hidden_size, weight_config, activation_config, norm_scale_hidden_config,
                 norm_scale_out_config, reverse_input=False, compute_output_scale=False,
                 compute_output_bit_width=False, return_quant_tensor=False,
                 recurrent_quant_config=None):

        super(QuantLSTMLayer, self).__init__()
        self.register_buffer(ZERO_HW_SENTINEL_NAME, torch.tensor(ZERO_HW_SENTINEL_VALUE))
        self.return_quant_tensor = return_quant_tensor
        self.weight_config = weight_config
        self.activation_config = activation_config

        weight_ii = nn.Parameter(torch.randn(hidden_size, input_size), requires_grad=True)
        weight_fi = nn.Parameter(torch.randn(hidden_size, input_size), requires_grad=True)
        weight_ai = nn.Parameter(torch.randn(hidden_size, input_size), requires_grad=True)
        weight_oi = nn.Parameter(torch.randn(hidden_size, input_size), requires_grad=True)

        weight_ih = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        weight_fh = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        weight_ah = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)
        weight_oh = nn.Parameter(torch.randn(hidden_size, hidden_size), requires_grad=True)

        self.weight_ii = weight_ii
        self.weight_fi = weight_fi
        self.weight_ai = weight_ai
        self.weight_oi = weight_oi

        self.weight_ih = weight_ih
        self.weight_fh = weight_fh
        self.weight_ah = weight_ah
        self.weight_oh = weight_oh

        self.bias_i = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
        self.bias_f = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
        self.bias_a = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
        self.bias_o = nn.Parameter(torch.randn(hidden_size), requires_grad=True)
        self.reverse_input = reverse_input

        self.weight_config['weight_scaling_shape'] = SCALING_SCALAR_SHAPE
        self.weight_config['weight_stats_input_view_shape_impl'] = StatsInputViewShapeImpl.OVER_TENSOR
        self.weight_config['weight_scaling_stats_input_concat_dim'] = 0
        self.weight_config['weight_scaling_stats_reduce_dim'] = None

        self.weight_proxy_i = self.configure_weight([weight_ii, weight_ih], self.weight_config)
        self.weight_proxy_f = self.configure_weight([weight_fi, weight_fh], self.weight_config)
        self.weight_proxy_a = self.configure_weight([weight_ai, weight_ah], self.weight_config)
        self.weight_proxy_o = self.configure_weight([weight_oi, weight_oh], self.weight_config)

        self.quant_sigmoid = self.configure_activation(self.activation_config, QuantSigmoid)
        self.quant_tanh = self.configure_activation(self.activation_config, QuantTanh)
        self.normalize_hidden_state = self.configure_activation(norm_scale_hidden_config, QuantHardTanh)

        self.out_quant = self.configure_activation(norm_scale_out_config, QuantHardTanh)
        if recurrent_quant_config is None:
            self.rec_quant = self.configure_activation(norm_scale_out_config, QuantHardTanh)
        else:
            self.rec_quant = self.configure_activation(recurrent_quant_config, QuantHardTanh)

        if self.weight_config.get('weight_quant_type', 'QuantType.FP') == 'QuantType.FP' and compute_output_bit_width:
            raise Exception("Computing output bit width requires enabling quantization")
        if self.weight_config.get('bias_quant_type', 'QuantType.FP') != 'QuantType.FP' and not (
                compute_output_scale and compute_output_bit_width):
            raise Exception("Quantizing bias requires to compute output scale and output bit width")

    @torch.jit.script_method
    def forward_iteration(self, input, hx, cx,
                          quant_weight_ii, quant_weight_fi, quant_weight_ai, quant_weight_oi,
                          quant_weight_ih, quant_weight_fh, quant_weight_ah, quant_weight_oh):

        zero_hw_sentinel = getattr(self, 'zero_hw_sentinel')

        igates_ii = torch.mm(input, quant_weight_ii.t())
        hgates_ih = torch.mm(hx, quant_weight_ih.t())

        igates_fi = torch.mm(input, quant_weight_fi.t())
        hgates_fh = torch.mm(hx, quant_weight_fh.t())

        igates_ai = torch.mm(input, quant_weight_ai.t())
        hgates_ah = torch.mm(hx, quant_weight_ah.t())

        igates_oi = torch.mm(input, quant_weight_oi.t())
        hgates_oh = torch.mm(hx, quant_weight_oh.t())

        ingate = (igates_ii + hgates_ih) + self.bias_i
        forgetgate = (igates_fi + hgates_fh) + self.bias_f
        cellgate = (igates_ai + hgates_ah) + self.bias_a
        outgate = (igates_oi + hgates_oh) + self.bias_o

        ingate = self.quant_sigmoid(ingate, zero_hw_sentinel)[0]
        forgetgate = self.quant_sigmoid(forgetgate, zero_hw_sentinel)[0]
        cellgate = self.quant_tanh(cellgate, zero_hw_sentinel)[0]
        outgate = self.quant_sigmoid(outgate, zero_hw_sentinel)[0]

        cx = self.normalize_hidden_state(cx, zero_hw_sentinel)[0]
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * self.quant_tanh(cy, zero_hw_sentinel)[0]
        hy1 = self.out_quant(hy, zero_hw_sentinel)[0]
        hy2 = self.rec_quant(hy, zero_hw_sentinel)[0]

        return hy1, (hy2, cy)

    @torch.jit.script_method
    def forward(self, inputs, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        inputs, input_scale, input_bit_width = self.unpack_input(inputs)
        zero_hw_sentinel = getattr(self, 'zero_hw_sentinel')

        quant_weight_ii, quant_weight_ii_scale, quant_weight_ii_bit_width = self.weight_proxy_i(self.weight_ii,
                                                                                                zero_hw_sentinel)
        quant_weight_fi, quant_weight_fi_scale, quant_weight_fi_bit_width = self.weight_proxy_f(self.weight_fi,
                                                                                                zero_hw_sentinel)
        quant_weight_ai, quant_weight_ai_scale, quant_weight_ai_bit_width = self.weight_proxy_a(self.weight_ai,
                                                                                                zero_hw_sentinel)
        quant_weight_oi, quant_weight_oi_scale, quant_weight_oi_bit_width = self.weight_proxy_o(self.weight_oi,
                                                                                                zero_hw_sentinel)
        quant_weight_ih, quant_weight_ih_scale, quant_weight_ih_bit_width = self.weight_proxy_i(self.weight_ih,
                                                                                                zero_hw_sentinel)
        quant_weight_fh, quant_weight_fh_scale, quant_weight_fh_bit_width = self.weight_proxy_f(self.weight_fh,
                                                                                                zero_hw_sentinel)
        quant_weight_ah, quant_weight_ah_scale, quant_weight_ah_bit_width = self.weight_proxy_a(self.weight_ah,
                                                                                                zero_hw_sentinel)
        quant_weight_oh, quant_weight_oh_scale, quant_weight_oh_bit_width = self.weight_proxy_o(self.weight_oh,
                                                                                                zero_hw_sentinel)

        inputs = inputs.unbind(0)

        start = 0
        end = len(inputs)
        step = 1
        if self.reverse_input:
            start = end - 1
            end = -1
            step = -1

        hx, cx = state
        outputs = torch.jit.annotate(List[Tensor], [])
        hx = self.out_quant(hx, zero_hw_sentinel)[0]
        for i in range(start, end, step):
            input_quant = self.out_quant(inputs[i], zero_hw_sentinel)[0]
            output, state = self.forward_iteration(input_quant, hx, cx,
                                                   quant_weight_ii, quant_weight_fi, quant_weight_ai, quant_weight_oi,
                                                   quant_weight_ih, quant_weight_fh, quant_weight_ah, quant_weight_oh)
            hx, cx = state
            outputs += [output]

        if self.reverse_input:
            return torch.stack(reverse(outputs)), state
        else:
            return torch.stack(outputs), state

    def max_output_bit_width(self, input_bit_width, weight_bit_width):
        raise Exception("Not supported yet")

    @torch.jit.script_method
    def unpack_input(self, input):
        if isinstance(input, QuantTensor):
            return input
        else:
            return input, None, None

    def configure_weight(self, weight, weight_config):
        zero_hw_sentinel = getattr(self, 'zero_hw_sentinel')
        wqp = _weight_quant_init_impl(bit_width=weight_config.get('weight_bit_width', 8),
                                      quant_type=weight_config.get('weight_quant_type', 'FP'),
                                      narrow_range=weight_config.get('weight_narrow_range', True),
                                      scaling_override=weight_config.get('weight_scaling_override',
                                                                         None),
                                      restrict_scaling_type=weight_config.get(
                                          'weight_restrict_scaling_type', RestrictValueType.LOG_FP),
                                      scaling_const=weight_config.get('weight_scaling_const', None),
                                      scaling_stats_op=weight_config.get('weight_scaling_stats_op',
                                                                         StatsOp.MAX),
                                      scaling_impl_type=weight_config.get('weight_scaling_impl_type',
                                                                          ScalingImplType.STATS),
                                      scaling_stats_reduce_dim=weight_config.get(
                                          'weight_scaling_stats_reduce_dim', None),
                                      scaling_shape=weight_config.get('weight_scaling_shape',
                                                                      SCALING_SCALAR_SHAPE),
                                      bit_width_impl_type=weight_config.get('weight_bit_width_impl_type',
                                                                            BitWidthImplType.CONST),
                                      bit_width_impl_override=weight_config.get(
                                          'weight_bit_width_impl_override', None),
                                      restrict_bit_width_type=weight_config.get(
                                          'weight_restrict_bit_width_type', RestrictValueType.INT),
                                      min_overall_bit_width=weight_config.get(
                                          'weight_min_overall_bit_width', 2),
                                      max_overall_bit_width=weight_config.get(
                                          'weight_max_overall_bit_width', None),
                                      ternary_threshold=weight_config.get('weight_ternary_threshold',
                                                                          0.5),
                                      scaling_stats_input_view_shape_impl=weight_config.get(
                                          'weight_stats_input_view_shape_impl',
                                          StatsInputViewShapeImpl.OVER_TENSOR),
                                      scaling_stats_input_concat_dim=weight_config.get(
                                          'weight_scaling_stats_input_concat_dim', 0),
                                      scaling_stats_sigma=weight_config.get('weight_scaling_stats_sigma',
                                                                            3.0),
                                      scaling_min_val=weight_config.get('weight_scaling_min_val',
                                                                        SCALING_MIN_VAL),
                                      override_pretrained_bit_width=weight_config.get(
                                          'weight_override_pretrained_bit_width', False),
                                      tracked_parameter_list=weight,
                                      zero_hw_sentinel=zero_hw_sentinel)

        return wqp

    def configure_activation(self, activation_config, activation_func=QuantSigmoid):
        signed = True
        max_val = 1
        min_val = -1
        if activation_func == QuantTanh:
            activation_impl = nn.Tanh()
        elif activation_func == QuantSigmoid:
            activation_impl = nn.Sigmoid()
            min_val = 0
            signed = False
        else:
            min_val = activation_config.get('min_val')
            max_val = activation_config.get('max_val')
            if activation_config.get('quant_type') == 'FP':
                activation_impl = ConstScalarClamp(min_val=min_val, max_val=max_val)
            else:
                activation_impl = nn.Identity()

        activation_object = _activation_quant_init_impl(activation_impl=activation_impl,
                                                        bit_width=activation_config.get('bit_width', 8),
                                                        narrow_range=activation_config.get('narrow_range', True),
                                                        quant_type=activation_config.get('quant_type', 'FP'),
                                                        float_to_int_impl_type=activation_config.get(
                                                            'float_to_int_impl_type', FloatToIntImplType.ROUND),
                                                        min_overall_bit_width=activation_config.get(
                                                            'min_overall_bit_width', 2),
                                                        max_overall_bit_width=activation_config.get(
                                                            'max_overall_bit_width', None),
                                                        bit_width_impl_override=activation_config.get(
                                                            'bit_width_impl_override', None),
                                                        bit_width_impl_type=activation_config.get('bit_width_impl_type',
                                                                                                  BitWidthImplType.CONST),
                                                        restrict_bit_width_type=activation_config.get(
                                                            'restrict_bit_width_type', RestrictValueType.INT),
                                                        restrict_scaling_type=activation_config.get(
                                                            'restrict_scaling_type', RestrictValueType.LOG_FP),
                                                        scaling_min_val=activation_config.get('scaling_min_val',
                                                                                              SCALING_MIN_VAL),
                                                        override_pretrained_bit_width=activation_config.get(
                                                            'override_pretrained_bit_width', False),
                                                        min_val=activation_config.get('min_val', min_val),
                                                        max_val=activation_config.get('max_val', max_val),
                                                        signed=activation_config.get('signed', signed),
                                                        per_channel_broadcastable_shape=None,
                                                        scaling_per_channel=False,
                                                        scaling_override=activation_config.get('scaling_override',
                                                                                               None),
                                                        scaling_impl_type=ScalingImplType.CONST,
                                                        scaling_stats_sigma=None,
                                                        scaling_stats_op=None,
                                                        scaling_stats_buffer_momentum=None,
                                                        scaling_stats_permute_dims=None)

        return activation_object

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        state_dict = self.fix_state_dict(state_dict)
        super(QuantLSTMLayer, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                          missing_keys, unexpected_keys, error_msgs)

        zero_hw_sentinel_key = prefix + "zero_hw_sentinel"

        if zero_hw_sentinel_key in missing_keys:
            missing_keys.remove(zero_hw_sentinel_key)
        if zero_hw_sentinel_key in unexpected_keys:  # for retrocompatibility with when it wasn't removed
            unexpected_keys.remove(zero_hw_sentinel_key)

    def fix_state_dict(self, state_dict):
        newstate = OrderedDict()
        hidden = self.weight_fh.shape[0]
        bias_i = torch.zeros(hidden)
        bias_f = torch.zeros(hidden)
        bias_a = torch.zeros(hidden)
        bias_o = torch.zeros(hidden)
        for name, value in state_dict.items():
            if name[:7] == 'bias_ih':
                bias_i = bias_i + value[:hidden]
                bias_f = bias_f + value[hidden:hidden * 2]
                bias_a = bias_a + value[2 * hidden:hidden * 3]
                bias_o = bias_o + value[3 * hidden:]
            elif name[:7] == 'bias_hh':
                bias_i = bias_i + value[:hidden]
                bias_f = bias_f + value[hidden:hidden * 2]
                bias_a = bias_a + value[2 * hidden:hidden * 3]
                bias_o = bias_o + value[3 * hidden:]
            elif name[:9] == 'weight_ih':
                newstate['weight_ii'] = value[:hidden, :]
                newstate['weight_fi'] = value[hidden:hidden * 2, :]
                newstate['weight_ai'] = value[2 * hidden:hidden * 3, :]
                newstate['weight_oi'] = value[3 * hidden:, :]
            elif name[:9] == 'weight_hh':
                newstate['weight_ih'] = value[:hidden, :]
                newstate['weight_fh'] = value[hidden:hidden * 2, :]
                newstate['weight_ah'] = value[2 * hidden:hidden * 3, :]
                newstate['weight_oh'] = value[3 * hidden:, :]
            else:
                newstate[name] = value

        newstate['bias_i'] = bias_i
        newstate['bias_f'] = bias_f
        newstate['bias_a'] = bias_a
        newstate['bias_o'] = bias_o

        return newstate


class BidirLSTMLayer(torch.jit.ScriptModule):
    __constants__ = ['directions']

    def __init__(self, input_size, hidden_size, weight_config, activation_config, norm_scale_out_config,
                 norm_scale_hidden_config,
                 compute_output_scale=False, compute_output_bit_width=False,
                 return_quant_tensor=False):
        super(BidirLSTMLayer, self).__init__()
        self.directions = nn.ModuleList([
            QuantLSTMLayer(input_size=input_size, hidden_size=hidden_size, weight_config=weight_config,
                           activation_config=activation_config,
                           norm_scale_out_config=norm_scale_out_config,
                           norm_scale_hidden_config=norm_scale_hidden_config,
                           reverse_input=False, compute_output_scale=compute_output_scale,
                           compute_output_bit_width=compute_output_bit_width,
                           return_quant_tensor=return_quant_tensor),
            QuantLSTMLayer(input_size=input_size, hidden_size=hidden_size, weight_config=weight_config,
                           activation_config=activation_config,
                           norm_scale_out_config=norm_scale_out_config,
                           norm_scale_hidden_config=norm_scale_hidden_config,
                           reverse_input=True, compute_output_scale=compute_output_scale,
                           compute_output_bit_width=compute_output_bit_width,
                           return_quant_tensor=return_quant_tensor)
        ])

    @torch.jit.script_method
    def forward(self, input, states):
        # type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
        outputs = torch.jit.annotate(List[Tensor], [])
        output_states = torch.jit.annotate(List[Tuple[Tensor, Tensor]], [])
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for direction in self.directions:
            state = states[i]
            out, out_state = direction(input, state)
            outputs += [out]
            output_states += [out_state]
            i += 1

        return torch.cat(outputs, -1), output_states

    def load_state_dict(self, state_dict, strict=True):
        direct = OrderedDict()
        reverse = OrderedDict()
        for name, value in state_dict.items():
            if name[-7:] == 'reverse':
                reverse[name] = value
            else:
                direct[name] = value

        self.directions[0].load_state_dict(direct, strict)
        self.directions[1].load_state_dict(reverse, strict)
