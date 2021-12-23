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

from abc import ABCMeta
from typing import Tuple, List, Optional
from collections import OrderedDict
import math

from torch import Tensor
from torch import nn
import torch

from .quant_recurrent import QuantWeightNMixin, QuantWeightRMixin, QuantWeightCMixin, reverse


__all__ = ['QuantGRULayer', 'BiGRULayer']


class QuantGRULayer(
    nn.Module,
    QuantWeightNMixin,
    QuantWeightRMixin,
    QuantWeightCMixin):
    __constants__ = ['reverse_input', 'batch_first', 'hidden_size']

    def __init__(
            self,
            input_size,
            hidden_size,
            weight_quant,
            act_quant,
            state_quant,
            norm_scale_hidden_quant,
            batch_first=False,
            reverse_input=False,
            **kwargs):
        QuantWeightCMixin.__init__(weight_quant=weight_quant, **kwargs)
        QuantWeightRMixin.__init__(weight_quant=weight_quant, **kwargs)
        QuantWeightNMixin.__init__(weight_quant=weight_quant, **kwargs)
        nn.Module.__init__(self)

        self.weight_ri = nn.Parameter(torch.randn(input_size, hidden_size))
        self.weight_ci = nn.Parameter(torch.randn(input_size, hidden_size))
        self.weight_ni = nn.Parameter(torch.randn(input_size, hidden_size))

        self.weight_rh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.weight_ch = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.weight_nh = nn.Parameter(torch.randn(hidden_size, hidden_size))

        self.bias_r = nn.Parameter(torch.zeros(hidden_size))
        self.bias_c = nn.Parameter(torch.zeros(hidden_size))
        self.bias_ni = nn.Parameter(torch.zeros(hidden_size))
        self.bias_nh = nn.Parameter(torch.zeros(hidden_size))

        self.reverse_input = reverse_input
        self.batch_first = batch_first
        self.hidden_size = hidden_size

        self.quant_sigmoid_r = self.configure_activation(self.activation_config, QuantSigmoid)
        self.quant_sigmoid_c = self.configure_activation(self.activation_config, QuantSigmoid)
        self.quant_tanh = self.configure_activation(self.activation_config, QuantTanh)

        self.norm_scale_newgate = self.configure_activation(self.norm_scale_hidden_config, QuantIdentity)
        self.norm_scale_out = self.configure_activation(self.norm_scale_out_config, QuantIdentity)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, param in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.uniform_(param, -stdv, stdv)

    def per_output_channel_broadcastable_shape(self):
        output_dim = 1
        per_channel_size = [1] * len(self.weight_ci.size())
        per_channel_size[output_dim] = self.hidden_size
        per_channel_size = tuple(per_channel_size)
        return per_channel_size

    @torch.jit.script_method
    def forward_iteration(self, input, state, quant_wih, quant_whh, quant_wni, quant_wnh):
        gates = (torch.mm(input, quant_wih) + torch.mm(state, quant_whh))
        rgate, cgate = gates.chunk(2, 1)
        gates_ni = torch.mm(input, quant_wni) + self.bias_ni
        gates_nh = torch.mm(state, quant_wnh) + self.bias_nh
        rgate = rgate + self.bias_r
        cgate = cgate + self.bias_c
        rgate = self.quant_sigmoid_r(rgate)
        cgate = self.quant_sigmoid_c(cgate)
        gates_ni = self.norm_scale_newgate(gates_ni) + self.norm_scale_newgate(rgate * gates_nh)
        ngate = self.quant_tanh(gates_ni)
        state = self.norm_scale_out(state) - self.norm_scale_out(ngate)
        hy = self.norm_scale_out(ngate) + self.norm_scale_out(cgate * state)
        return hy, hy

    @torch.jit.script_method
    def forward_cycle(self, inputs, state, quant_wih, quant_whh, quant_wni, quant_wnh):
        end = len(inputs)
        step = 1
        index = 0
        if self.reverse_input:
            end = len(inputs)
            index = end - 1
            step = -1
        outputs = torch.jit.annotate(List[Tensor], [])
        state = self.norm_scale_out(state)
        for i in range(end):
            input_quant = self.norm_scale_out(inputs[index])
            output, state = self.forward_iteration(
                input_quant, state, quant_wih, quant_whh, quant_wni, quant_wnh)
            index = index + step
            outputs += [output]
        return outputs, state

    def forward(self, inputs, state=None):
        quant_wri = self.weight_r_quant(self.weight_ri)
        quant_wci = self.weight_c_quant(self.weight_ci)
        quant_wni = self.weight_n_quant(self.weight_ni)
        quant_wrh = self.weight_r_quant(self.weight_rh)
        quant_wch = self.weight_c_quant(self.weight_ch)
        quant_wnh = self.weight_n_quant(self.weight_nh)
        quant_wih = torch.cat([quant_wri, quant_wci], dim=1)
        quant_whh = torch.cat([quant_wrh, quant_wch], dim=1)

        if self.batch_first:
            dim = 1
            inputs_unbinded = inputs.unbind(1)
        else:
            dim = 0
            inputs_unbinded = inputs.unbind(0)
        batch_size = inputs_unbinded[0].shape[0]

        if state is None:
            device = inputs_unbinded[0].device
            state = torch.zeros(batch_size, self.hidden_size, device=device)

        outputs, state = self.forward_cycle(
            inputs_unbinded, state, quant_wih, quant_whh, quant_wni, quant_wnh)

        if self.reverse_input:
            return torch.stack(reverse(outputs), dim=dim), outputs[-1]
        else:
            return torch.stack(outputs, dim=dim), outputs[-1]

    def map_state_dict(self, prefix, state_dict):
        new_state = OrderedDict()
        hidden = self.hidden_size
        bias_r = torch.zeros(hidden)
        bias_c = torch.zeros(hidden)
        prefix_len = len(prefix)
        for name, value in state_dict.items():
            if name[:prefix_len + 7] == prefix + 'bias_ih':
                bias_r = bias_r + value[:hidden]
                bias_c = bias_c + value[hidden:hidden * 2]
                new_state[prefix + 'bias_ni'] = value[2 * hidden:hidden * 3]
            elif name[:prefix_len + 7] == prefix + 'bias_hh':
                bias_r = bias_r + value[:hidden]
                bias_c = bias_c + value[hidden:hidden * 2]
                new_state[prefix + 'bias_nh'] = value[2 * hidden:hidden * 3]
            elif name[:prefix_len + 9] == prefix + 'weight_ih':
                new_state[prefix + 'weight_ri'] = (value[:hidden, :]).t()
                new_state[prefix + 'weight_ci'] = (value[hidden:hidden * 2, :]).t()
                new_state[prefix + 'weight_ni'] = (value[2 * hidden:hidden * 3, :]).t()
            elif name[:prefix_len + 9] == prefix + 'weight_hh':
                new_state[prefix + 'weight_rh'] = (value[:hidden, :]).t()
                new_state[prefix + 'weight_ch'] = (value[hidden:hidden * 2, :]).t()
                new_state[prefix + 'weight_nh'] = (value[2 * hidden:hidden * 3, :]).t()
            else:
                new_state[name] = value
        new_state[prefix + 'bias_r'] = bias_r
        new_state[prefix + 'bias_c'] = bias_c
        return new_state

    def _load_from_state_dict(
            self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        dict_to_change = dict()
        for k, v in state_dict.items():
            if k.startswith(prefix):
                dict_to_change[k] = v
        for k in list(state_dict.keys()):
            if k.startswith(prefix):
                del state_dict[k]
        dict_changed = self.map_state_dict(prefix, dict_to_change)
        for k, v in dict_changed.items():
            state_dict[k] = v
        super(QuantGRULayer, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs)


class BiGRULayer(nn.Module):
    __constants__ = ['directions']

    def __init__(self, input_size, hidden_size, weight_config, activation_config, norm_scale_out_config,
                 norm_scale_hidden_config,
                 compute_output_scale=False, compute_output_bit_width=False,
                 return_quant_tensor=False):
        super(BiGRULayer, self).__init__()
        self.directions = nn.ModuleList([
            QuantGRULayer(input_size=input_size, hidden_size=hidden_size, weight_config=weight_config,
                          activation_config=activation_config,
                          norm_scale_out_config=norm_scale_out_config,
                          norm_scale_hidden_config=norm_scale_hidden_config,
                          reverse_input=False, compute_output_scale=compute_output_scale,
                          compute_output_bit_width=compute_output_bit_width,
                          return_quant_tensor=return_quant_tensor),
            QuantGRULayer(input_size=input_size, hidden_size=hidden_size, weight_config=weight_config,
                          activation_config=activation_config,
                          norm_scale_out_config=norm_scale_out_config,
                          norm_scale_hidden_config=norm_scale_hidden_config,
                          reverse_input=True, compute_output_scale=compute_output_scale,
                          compute_output_bit_width=compute_output_bit_width,
                          return_quant_tensor=return_quant_tensor),
        ])

    def forward(self, input, states):
        outputs = torch.jit.annotate(List[Tensor], [])
        output_states = torch.jit.annotate(List[Tensor], [])
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
