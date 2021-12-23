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

import math
from typing import Optional, Tuple, List
from collections import OrderedDict

from torch import Tensor
import torch.nn as nn
import torch

import brevitas
from brevitas.nn.mixin.base import QuantLayerMixin
from brevitas.quant import Int8WeightPerTensorFloat, Int8ActPerTensorFloat
from .quant_recurrent import QuantWeightRMixin, QuantIOMixin, reverse
from .utils import compute_channel_view_shape

__all__ = ['QuantRNNLayer', 'BiRNNLayer']


class _QuantRNNCell(brevitas.jit.ScriptModule):

    def __init__(self, io_quant):
        super(_QuantRNNCell, self).__init__()
        self.io_quant = io_quant

    @brevitas.jit.script_method
    def forward_iter(
            self, quant_input_val, quant_state_val, quant_wri_val, quant_wrh_val, quant_br_val):
        gates_ri = torch.mm(quant_input_val, quant_wri_val)
        gates_rh = torch.mm(quant_state_val, quant_wrh_val)
        rgate = (gates_ri + gates_rh) + quant_br_val
        rgate = torch.tanh(rgate)
        quant_state = self.io_quant(rgate)
        return quant_state

    @brevitas.jit.script_method
    def forward(self, inputs, state, quant_wri, quant_wrh, quant_br):
        end = len(inputs)
        step = 1
        index = 0
        if self.reverse_input:
            end = len(inputs)
            index = end - 1
            step = - 1
        outputs = torch.jit.annotate(List[Tensor], [])
        quant_state = self.io_quant(state)
        for i in range(end):
            quant_input = self.io_quant(inputs[index])
            quant_state = self.forward_iter(
                quant_input[0], quant_state[0], quant_wri, quant_wrh, quant_br)
            index = index + step
            outputs += [quant_state]
        return outputs


class QuantRNNLayer(
    QuantWeightRMixin,
    QuantIOMixin,
    QuantLayerMixin,
    nn.Module):
    __constants__ = ['reverse_input', 'batch_first', 'hidden_size']

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            weight_quant = Int8WeightPerTensorFloat,
            io_quant = Int8ActPerTensorFloat,
            reverse_input=False,
            batch_first=False,
            return_quant_tensor=False,
            **kwargs):
        nn.Module.__init__(self)
        QuantWeightRMixin.__init__(self, weight_quant=weight_quant, **kwargs)
        QuantIOMixin.__init__(self, io_quant, **kwargs)
        QuantLayerMixin.__init__(self, return_quant_tensor=return_quant_tensor, **kwargs)
        self.cell_impl = _QuantRNNCell(self.io_quant)
        self.weight_ri = nn.Parameter(torch.randn(input_size, hidden_size))
        self.weight_rh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bias_r = nn.Parameter(torch.randn(hidden_size))
        self.reverse_input = reverse_input
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in [self.weight_rh, self.weight_ri]:
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def unpack_inputs_state(self, inputs, state):
        if isinstance(inputs, QuantTensor):
            inputs = inputs.value
        if self.batch_first:
            dim = 1
            inputs_unbinded = inputs.unbind(1)
        else:
            dim = 0
            inputs_unbinded = inputs.unbind(0)
        if state is None:
            batch_size = inputs_unbinded[0].shape[dim]
            state = torch.zeros(batch_size, self.hidden_size, device=inputs_unbinded[0].device)
        elif isinstance(state, QuantTensor):
            state = state.value
        return inputs, state

    def forward(self, inp, state=None):
        output_scale = None
        output_bit_width = None
        output_zero_point = None
        output_signed = None

        inp = self.unpack_input(inp)
        quant_input = self.io_quant(inp)
        quant_wri = self.weight_r_quant(self.weight_ri)
        quant_wrh = self.weight_r_quant(self.weight_rh)

        if quant_input.bit_width is not None:
            output_bit_width = self.max_acc_bit_width(quant_input.bit_width, quant_weight.bit_width)
        if quant_input.scale is not None:
            output_scale_shape = compute_channel_view_shape(inp, channel_dim=1)
            output_scale = quant_wri.scale.view(output_scale_shape)
            output_scale = output_scale * quant_input.scale.view(output_scale_shape)
        if quant_input.signed is not None:
            output_signed = inp.signed or quant_wri.signed

        quant_br = self.bias_r_quant(self.bias_r, output_scale, output_bit_width)

        inputs, state = self.unpack_inputs_state(quant_input, state)

        outputs = self.cell_impl(inputs, state, quant_wri, quant_wrh, quant_br)
        # unpack outputs
        outputs = [o[0] for o in outputs]
        if self.reverse_input:
            return torch.stack(reverse(outputs), dim=dim), outputs[-1]
        else:
            return torch.stack(outputs, dim=dim), outputs[-1]

    def _map_state_dict(self, prefix, state_dict):
        new_state = OrderedDict()
        hidden = self.hidden_size
        bias_r = torch.zeros(hidden)
        prefix_len = len(prefix)
        for name, value in state_dict.items():
            if name[:prefix_len + 7] == prefix + 'bias_ih':
                bias_r = bias_r + value
            elif name[:prefix_len + 7] == prefix + 'bias_hh':
                bias_r = bias_r + value
            elif name[:prefix_len + 9] == prefix + 'weight_ih':
                new_state[prefix + 'weight_ri'] = value.t()
            elif name[:prefix_len + 9] == prefix + 'weight_hh':
                new_state[prefix + 'weight_rh'] = value.t()
            else:
                new_state[name] = value
        new_state[prefix + 'bias_r'] = bias_r
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
        dict_changed = self._map_state_dict(prefix, dict_to_change)
        for k, v in dict_changed.items():
            state_dict[k] = v
        super(QuantRNNLayer, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs)


class BiQuantRNNLayer(nn.Module):
    __constants__ = ['directions']

    def __init__(
            self,
            input_size,
            hidden_size,
            weight_quant = Int8WeightPerTensorFloat,
            io_quant = Int8ActPerTensorFloat,
            return_quant_tensor=False,
            **kwargs):
        super(BiQuantRNNLayer, self).__init__()
        self.directions = nn.ModuleList([
            QuantRNNLayer(
                input_size=input_size,
                hidden_size=hidden_size,
                weight_quant=weight_quant,
                io_quant=io_quant,
                reverse_input=False,
                return_quant_tensor=return_quant_tensor,
                **kwargs),
            QuantRNNLayer(
                input_size=input_size,
                hidden_size=hidden_size,
                weight_quant=weight_quant,
                io_quant=io_quant,
                reverse_input=True,
                return_quant_tensor=return_quant_tensor,
                **kwargs),
        ])

    def forward(self, input, states):
        # type: (Tensor, List[Tensor]) -> Tuple[Tensor, List[Tensor]]
        # List[LSTMState]: [forward LSTMState, backward LSTMState]
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
