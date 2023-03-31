# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from functools import partial
import math
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

import brevitas
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantSigmoid
from brevitas.nn import QuantTanh
from brevitas.nn.mixin import QuantBiasMixin
from brevitas.nn.mixin import QuantWeightMixin
from brevitas.nn.mixin.base import QuantRecurrentLayerMixin
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import Uint8ActPerTensorFloat
from brevitas.quant_tensor import QuantTensor

QuantTupleShortEnabled = List[Tuple[Tensor, Tensor, Tensor, Tensor]]
QuantTupleShortDisabled = List[Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]]
QuantTupleLongEnabled = List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]
QuantTupleLongDisabled = List[Tuple[Tensor,
                                    Optional[Tensor],
                                    Optional[Tensor],
                                    Optional[Tensor],
                                    Optional[Tensor],
                                    Optional[Tensor]]]


class GateWeight(QuantWeightMixin, nn.Module):

    def __init__(self, input_features, output_features, weight_quant, **kwargs):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.randn(output_features, input_features))
        QuantWeightMixin.__init__(self, weight_quant=weight_quant, **kwargs)

    @property
    def output_channel_dim(self):
        return 0

    @property
    def out_channels(self):
        return self.weight.size(self.output_channel_dim)

    def forward(self):
        return self.weight_quant(self.weight)


class GateParams(QuantBiasMixin, nn.Module):

    def __init__(
            self, input_size, hidden_size, bias, weight_quant, bias_quant, input_weight, **kwargs):
        nn.Module.__init__(self)
        if bias:
            self.bias = nn.Parameter(torch.randn(hidden_size))
        else:
            self.bias = None
        QuantBiasMixin.__init__(self, bias_quant, **kwargs)
        if input_weight is None:
            input_weight = GateWeight(input_size, hidden_size, weight_quant=weight_quant, **kwargs)
        self.input_weight = input_weight
        # The quantizer is shared among input-to-hidden and hidden-to-hidden weights
        self.hidden_weight = GateWeight(
            hidden_size, hidden_size, weight_quant=input_weight.weight_quant)


# Simply putting the forward in a standalone script function is not enough
# for the compiler to realize that the bools being passed in are all constants
# so we resort to a module instead
class _QuantStatesInit(nn.Module):
    __constants__ = ['fast_impl', 'quant_enabled']

    def __init__(self, fast_impl: bool, quant_enabled: bool):
        super(_QuantStatesInit, self).__init__()
        self.fast_impl = fast_impl
        self.quant_enabled = quant_enabled

    def forward(self):
        if self.fast_impl:
            if self.quant_enabled:
                quant_states = torch.jit.annotate(QuantTupleShortEnabled, [])
            else:
                quant_states = torch.jit.annotate(QuantTupleShortDisabled, [])
        else:
            if self.quant_enabled:
                quant_states = torch.jit.annotate(QuantTupleLongEnabled, [])
            else:
                quant_states = torch.jit.annotate(QuantTupleLongDisabled, [])
        return quant_states


class _QuantRNNCell(nn.Module):
    __constants__ = ['reverse_input', 'batch_first']

    def __init__(
            self,
            act_fn: nn.Module,
            gate_acc_quant: nn.Module,
            output_quant: nn.Module,
            reverse_input: bool,
            batch_first: bool,
            output_quant_enabled: bool,
            fast_impl: bool):
        super(_QuantRNNCell, self).__init__()
        self.act_fn = act_fn
        self.gate_acc_quant = gate_acc_quant
        self.output_quant = output_quant
        self.reverse_input = reverse_input
        self.batch_first = batch_first
        self.hidden_states_init = _QuantStatesInit(fast_impl, output_quant_enabled)

    def forward_iter(self, quant_input, quant_state, quant_weight_ih, quant_weight_hh, quant_bias):
        quant_gate_ih = F.linear(quant_input, quant_weight_ih)
        quant_gate_hh = F.linear(quant_state, quant_weight_hh)
        quant_gate = self.gate_acc_quant(quant_gate_ih + quant_gate_hh + quant_bias)[0]
        quant_gate = self.act_fn(quant_gate)
        quant_state_tuple = self.output_quant(quant_gate)
        return quant_state_tuple

    def forward(
            self,
            quant_input: Tensor,
            quant_state: Tensor,
            quant_weight_ih: Tensor,
            quant_weight_hh: Tensor,
            quant_bias: Tensor):
        if self.batch_first:
            quant_inputs = quant_input.unbind(1)
        else:
            quant_inputs = quant_input.unbind(0)
        end = len(quant_inputs)
        step = 1
        index = 0
        quant_outputs = self.hidden_states_init()
        if self.reverse_input:
            index = end - 1
            step = -1
        for _ in range(end):
            quant_input = quant_inputs[index]
            quant_state_tuple = self.forward_iter(
                quant_input, quant_state, quant_weight_ih, quant_weight_hh, quant_bias)
            index = index + step
            quant_outputs += [quant_state_tuple]
            quant_state = quant_state_tuple[0]
        return quant_outputs


class _QuantLSTMCell(nn.Module):
    __constants__ = ['reverse_input', 'batch_first', 'cifg']

    def __init__(
            self,
            output_quant,
            cell_state_quant,
            input_acc_quant,
            forget_acc_quant,
            cell_acc_quant,
            output_acc_quant,
            input_sigmoid_quant,
            forget_sigmoid_quant,
            cell_tanh_quant,
            output_sigmoid_quant,
            hidden_state_tanh_quant,
            reverse_input: bool,
            batch_first: bool,
            cifg: bool,
            output_quant_enabled: bool,
            cell_state_quant_enabled: bool,
            fast_impl: bool):
        super(_QuantLSTMCell, self).__init__()
        self.output_quant = output_quant
        self.cell_state_quant = cell_state_quant
        self.input_acc_quant = input_acc_quant
        self.forget_acc_quant = forget_acc_quant
        self.cell_acc_quant = cell_acc_quant
        self.output_acc_quant = output_acc_quant
        self.input_sigmoid_quant = input_sigmoid_quant
        self.forget_sigmoid_quant = forget_sigmoid_quant
        self.cell_tanh_quant = cell_tanh_quant
        self.output_sigmoid_quant = output_sigmoid_quant
        self.hidden_state_tanh_quant = hidden_state_tanh_quant
        self.reverse_input = reverse_input
        self.batch_first = batch_first
        self.cifg = cifg
        self.hidden_states_init = _QuantStatesInit(fast_impl, output_quant_enabled)
        self.cell_states_init = _QuantStatesInit(fast_impl, cell_state_quant_enabled)

    def forward_iter(
            self,
            quant_input: Tensor,
            quant_hidden_state: Tensor,
            quant_cell_state: Tensor,
            quant_weight_ii: Tensor,
            quant_weight_if: Tensor,
            quant_weight_ic: Tensor,
            quant_weight_io: Tensor,
            quant_weight_hi: Tensor,
            quant_weight_hf: Tensor,
            quant_weight_hc: Tensor,
            quant_weight_ho: Tensor,
            quant_bias_input: Tensor,
            quant_bias_forget: Tensor,
            quant_bias_cell: Tensor,
            quant_bias_output: Tensor):
        # Input gate
        quant_ii_gate = F.linear(quant_input, quant_weight_ii)
        quant_hi_gate = F.linear(quant_hidden_state, quant_weight_hi)
        quant_input_gate = self.input_acc_quant(quant_ii_gate + quant_hi_gate + quant_bias_input)[0]
        quant_input_gate = self.input_sigmoid_quant(quant_input_gate)[0]
        # Forget gate
        if self.cifg:
            quant_ones = self.input_sigmoid_quant.tensor_quant(torch.ones_like(quant_input_gate))[0]
            # CIFG is defined as 1 - input_gate, in line with ONNXRuntime
            quant_forget_gate = quant_ones - quant_input_gate
        else:
            quant_if_gate = F.linear(quant_input, quant_weight_if)
            quant_hf_gate = F.linear(quant_hidden_state, quant_weight_hf)
            quant_forget_gate = self.forget_acc_quant(
                quant_if_gate + quant_hf_gate + quant_bias_forget)[0]
            quant_forget_gate = self.forget_sigmoid_quant(quant_forget_gate)[0]
        # Cell gate
        quant_ic_gate = F.linear(quant_input, quant_weight_ic)
        quant_hc_gate = F.linear(quant_hidden_state, quant_weight_hc)
        quant_cell_gate = self.cell_acc_quant(quant_ic_gate + quant_hc_gate + quant_bias_cell)[0]
        quant_cell_gate = self.cell_tanh_quant(quant_cell_gate)[0]
        # Output gate
        quant_io_gate = F.linear(quant_input, quant_weight_io)
        quant_ho_gate = F.linear(quant_hidden_state, quant_weight_ho)
        quant_out_gate = self.output_acc_quant(quant_io_gate + quant_ho_gate + quant_bias_output)[0]
        quant_out_gate = self.output_sigmoid_quant(quant_out_gate)[0]
        quant_forget_cell = self.cell_state_quant(quant_forget_gate * quant_cell_state)[0]
        quant_inp_cell = self.cell_state_quant(quant_input_gate * quant_cell_gate)[0]
        quant_cell_state_tuple = self.cell_state_quant(quant_forget_cell + quant_inp_cell)
        quant_hidden_state_tanh = self.hidden_state_tanh_quant(quant_cell_state_tuple[0])[0]
        quant_hidden_state = quant_out_gate * quant_hidden_state_tanh
        quant_hidden_state_tuple = self.output_quant(quant_hidden_state)
        return quant_hidden_state_tuple, quant_cell_state_tuple

    def forward(
            self,
            quant_input: Tensor,
            quant_hidden_state: Tensor,
            quant_cell_state: Tensor,
            quant_weight_ii: Tensor,
            quant_weight_if: Tensor,
            quant_weight_ic: Tensor,
            quant_weight_io: Tensor,
            quant_weight_hi: Tensor,
            quant_weight_hf: Tensor,
            quant_weight_hc: Tensor,
            quant_weight_ho: Tensor,
            quant_bias_input: Tensor,
            quant_bias_forget: Tensor,
            quant_bias_cell: Tensor,
            quant_bias_output: Tensor):
        if self.batch_first:
            seq_dim = 1
        else:
            seq_dim = 0
        quant_inputs = quant_input.unbind(seq_dim)
        end = len(quant_inputs)
        step = 1
        index = 0
        if self.reverse_input:
            index = end - 1
            step = -1
        quant_hidden_states = self.hidden_states_init()
        quant_cell_states = self.cell_states_init()
        for _ in range(end):
            quant_input = quant_inputs[index]
            quant_hidden_state_tuple, quant_cell_state_tuple = self.forward_iter(
                quant_input,
                quant_hidden_state,
                quant_cell_state,
                quant_weight_ii,
                quant_weight_if,
                quant_weight_ic,
                quant_weight_io,
                quant_weight_hi,
                quant_weight_hf,
                quant_weight_hc,
                quant_weight_ho,
                quant_bias_input,
                quant_bias_forget,
                quant_bias_cell,
                quant_bias_output)
            index = index + step
            quant_hidden_states += [quant_hidden_state_tuple]
            quant_hidden_state = quant_hidden_state_tuple[0]
            quant_cell_states += [quant_cell_state_tuple]
            quant_cell_state = quant_cell_state_tuple[0]
        return quant_hidden_states, quant_hidden_states[-1], quant_cell_states[-1]


class _QuantRNNLayer(QuantRecurrentLayerMixin, nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool,
            batch_first: bool,
            weight_quant,
            bias_quant,
            io_quant,
            gate_acc_quant,
            reverse_input: bool,
            quantize_output_only: bool,
            shared_input_hidden_weights: bool,
            return_quant_tensor: bool,
            nonlinearity: str,
            input_weight: GateWeight = None,
            **kwargs):
        nn.Module.__init__(self)
        io_quant = QuantIdentity(io_quant, act_kwargs_prefix='io_', **kwargs)
        gate_acc_quant = QuantIdentity(
            gate_acc_quant, act_kwargs_prefix='gate_acc_quant_', **kwargs)
        if nonlinearity == 'tanh':
            act_fn = nn.Tanh()
        elif nonlinearity == 'relu':
            act_fn = nn.ReLU()
        else:
            raise RuntimeError(f"{nonlinearity} not supported.")
        cell = _QuantRNNCell(
            act_fn,
            gate_acc_quant.act_quant,
            io_quant.act_quant,
            reverse_input,
            batch_first,
            io_quant.act_quant.is_quant_enabled,
            fast_impl=False)
        QuantRecurrentLayerMixin.__init__(
            self,
            cell=cell,
            io_quant=io_quant.act_quant,
            input_size=input_size,
            hidden_size=hidden_size,
            reverse_input=reverse_input,
            quantize_output_only=quantize_output_only,
            shared_input_hidden_weights=shared_input_hidden_weights,
            return_quant_tensor=return_quant_tensor)
        self.gate_params = GateParams(
            input_size, hidden_size, bias, weight_quant, bias_quant, input_weight, **kwargs)
        self.reset_parameters()

    @property
    def weights_to_share(self):
        if self.shared_input_hidden_weights:
            return {'input_weight': self.gate_params.input_weight}
        else:
            return {}

    @property
    def quantizers_to_share(self):
        shared_quantizers = {'io_quant': self.io_quant}
        return shared_quantizers

    @property
    def fast_cell(self):
        if self._fast_cell is not None:
            return self._fast_cell
        else:
            # lazy late init to make sure the correct fused act quant proxy is captured
            # since on every sharing among layers at init time it is re-initialized
            self._fast_cell = _QuantRNNCell(
                self.cell.act_fn,
                self._wrap_act_proxy('gate_acc_quant'),
                self._wrap_act_proxy('output_quant'),
                self.cell.reverse_input,
                self.cell.batch_first,
                self.cell.output_quant.is_quant_enabled,
                fast_impl=True)
            if brevitas.config.JIT_ENABLED:
                self._fast_cell = torch.jit.script(self._fast_cell)
            return self._fast_cell

    def forward(self, inp, state):
        quant_input = self.maybe_quantize_input(inp)
        quant_weight_ih, quant_weight_hh, quant_bias = self.gate_params_fwd(
            self.gate_params, quant_input)
        if quant_bias.value is None:
            quant_bias = torch.tensor(0., device=quant_input.value.device)
        else:
            quant_bias = quant_bias.value
        quant_state = self.maybe_quantize_state(quant_input.value, state, self.cell.output_quant)
        if self.export_mode:
            cell = self.export_handler
        elif self.fast_mode:
            cell = self.fast_cell
        else:
            cell = self.cell
        quant_outputs = cell(
            quant_input.value,
            quant_state.value,
            quant_weight_ih.value,
            quant_weight_hh.value,
            quant_bias)
        quant_output = self.pack_quant_outputs(quant_outputs)
        quant_state = self.pack_quant_state(quant_outputs[-1], self.cell.output_quant)
        return quant_output, quant_state

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        hs = self.hidden_size

        def bias():
            bias_name = f'{prefix}gate_params.bias'
            if not bias_name in state_dict.keys():
                state_dict[bias_name] = torch.zeros(hs)
            return state_dict[bias_name]

        def _set_weight(value, input_name):
            key = f'{prefix}gate_params.{input_name}_weight.weight'
            state_dict[key] = value

        set_input_weight = partial(_set_weight, input_name='input')
        set_hidden_weight = partial(_set_weight, input_name='hidden')

        for name, value in list(state_dict.items()):
            if prefix + 'bias_ih' in name or prefix + 'bias_hh' in name:
                bias().add_(value[:hs])
                del state_dict[name]
            elif prefix + 'weight_ih' in name:
                set_input_weight(value[:hs, :])
                del state_dict[name]
            elif prefix + 'weight_hh' in name:
                set_hidden_weight(value[:hs, :])
                del state_dict[name]

        super(_QuantRNNLayer, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class _QuantLSTMLayer(QuantRecurrentLayerMixin, nn.Module):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool,
            batch_first: bool,
            weight_quant,
            bias_quant,
            io_quant,
            gate_acc_quant,
            sigmoid_quant,
            tanh_quant,
            cell_state_quant,
            reverse_input: bool,
            quantize_output_only: bool,
            cifg: bool,
            shared_input_hidden_weights: bool,
            shared_cell_state_quant: bool,
            shared_intra_layer_weight_quant: bool,
            shared_intra_layer_gate_acc_quant: bool,
            return_quant_tensor: bool,
            input_input_weight: GateWeight = None,
            input_forget_weight: GateWeight = None,
            input_cell_weight: GateWeight = None,
            input_output_weight: GateWeight = None,
            **kwargs):
        nn.Module.__init__(self)
        io_quant = QuantIdentity(io_quant, act_kwargs_prefix='io_', **kwargs)
        cell_state_quant = QuantIdentity(
            cell_state_quant, act_kwargs_prefix='cell_state_', **kwargs)

        # Quantizers for the output accumulator of each gate
        input_acc_quant = QuantIdentity(gate_acc_quant, act_kwargs_prefix='gate_acc_', **kwargs)
        if shared_intra_layer_gate_acc_quant:
            forget_acc_quant = input_acc_quant
            cell_acc_quant = input_acc_quant
            output_acc_quant = input_acc_quant
        else:
            cell_acc_quant = QuantIdentity(gate_acc_quant, act_kwargs_prefix='gate_acc_', **kwargs)
            output_acc_quant = QuantIdentity(
                gate_acc_quant, act_kwargs_prefix='gate_acc_', **kwargs)
            if cifg:
                # Avoid dealing with None and set it to the input one
                forget_acc_quant = input_acc_quant
            else:
                forget_acc_quant = QuantIdentity(
                    gate_acc_quant, act_kwargs_prefix='gate_acc_', **kwargs)

        # Internal quantized activations
        input_sigmoid_quant = QuantSigmoid(sigmoid_quant, act_kwargs_prefix='sigmoid_', **kwargs)
        cell_tanh_quant = QuantTanh(tanh_quant, act_kwargs_prefix='tanh_', **kwargs)
        output_sigmoid_quant = QuantSigmoid(sigmoid_quant, act_kwargs_prefix='sigmoid_', **kwargs)
        hidden_state_tanh_quant = QuantTanh(tanh_quant, act_kwargs_prefix='tanh_', **kwargs)
        if cifg:
            # Avoid dealing with None and set it to the input one
            forget_sigmoid_quant = input_sigmoid_quant
        else:
            forget_sigmoid_quant = QuantSigmoid(
                sigmoid_quant, act_kwargs_prefix='sigmoid_', **kwargs)

        cell = _QuantLSTMCell(
            output_quant=io_quant.act_quant,
            cell_state_quant=cell_state_quant.act_quant,
            input_acc_quant=input_acc_quant.act_quant,
            forget_acc_quant=forget_acc_quant.act_quant,
            cell_acc_quant=cell_acc_quant.act_quant,
            output_acc_quant=output_acc_quant.act_quant,
            input_sigmoid_quant=input_sigmoid_quant.act_quant,
            forget_sigmoid_quant=forget_sigmoid_quant.act_quant,
            cell_tanh_quant=cell_tanh_quant.act_quant,
            output_sigmoid_quant=output_sigmoid_quant.act_quant,
            hidden_state_tanh_quant=hidden_state_tanh_quant.act_quant,
            reverse_input=reverse_input,
            batch_first=batch_first,
            cifg=cifg,
            output_quant_enabled=io_quant.act_quant.is_quant_enabled,
            cell_state_quant_enabled=cell_state_quant.act_quant.is_quant_enabled,
            fast_impl=False)
        QuantRecurrentLayerMixin.__init__(
            self,
            cell=cell,
            io_quant=io_quant.act_quant,
            input_size=input_size,
            hidden_size=hidden_size,
            reverse_input=reverse_input,
            quantize_output_only=quantize_output_only,
            shared_input_hidden_weights=shared_input_hidden_weights,
            return_quant_tensor=return_quant_tensor)

        self.input_gate_params = GateParams(
            input_size, hidden_size, bias, weight_quant, bias_quant, input_forget_weight, **kwargs)
        if shared_intra_layer_weight_quant:
            # Share the input-to-hidden input weight quantizer, which is also shared with hidden-to-hidden
            weight_quant = self.input_gate_params.input_weight.weight_quant
        if cifg:
            self.forget_gate_params = None
        else:
            self.forget_gate_params = GateParams(
                input_size,
                hidden_size,
                bias,
                weight_quant,
                bias_quant,
                input_input_weight,
                **kwargs)
        self.cell_gate_params = GateParams(
            input_size, hidden_size, bias, weight_quant, bias_quant, input_cell_weight, **kwargs)
        self.output_gate_params = GateParams(
            input_size, hidden_size, bias, weight_quant, bias_quant, input_output_weight, **kwargs)
        self.shared_cell_state_quant = shared_cell_state_quant
        self.cifg = cifg
        self.reset_parameters()

    @property
    def weights_to_share(self):
        if self.shared_input_hidden_weights:
            out = {
                'input_input_weight': self.input_gate_params.input_weight,
                'input_cell_weight': self.cell_gate_params.input_weight,
                'input_output_weight': self.output_gate_params.input_weight}
            if not self.cifg:
                out['input_forget_weight'] = self.forget_gate_params.input_weight
            return out
        else:
            return {}

    @property
    def quantizers_to_share(self):
        shared_quantizers = {'io_quant': self.io_quant}
        if self.shared_cell_state_quant:
            shared_quantizers['cell_state_quant'] = self.cell.cell_state_quant
        return shared_quantizers

    @property
    def fast_cell(self):
        if self._fast_cell is not None:
            return self._fast_cell
        else:
            # lazy late init to make sure the correct fused act quant proxy is captured
            # since on every sharing among layers at init time it is re-initialized
            self._fast_cell = _QuantLSTMCell(
                # Wrap a None or act_fn only proxy to still return a tuple
                # whenever quantization is disabled
                output_quant=self._wrap_act_proxy('output_quant'),
                cell_state_quant=self._wrap_act_proxy('cell_state_quant'),
                input_acc_quant=self._wrap_act_proxy('input_acc_quant'),
                forget_acc_quant=self._wrap_act_proxy('forget_acc_quant'),
                cell_acc_quant=self._wrap_act_proxy('cell_acc_quant'),
                output_acc_quant=self._wrap_act_proxy('output_acc_quant'),
                input_sigmoid_quant=self._wrap_act_proxy('input_sigmoid_quant'),
                forget_sigmoid_quant=self._wrap_act_proxy('forget_sigmoid_quant'),
                cell_tanh_quant=self._wrap_act_proxy('cell_tanh_quant'),
                output_sigmoid_quant=self._wrap_act_proxy('output_sigmoid_quant'),
                hidden_state_tanh_quant=self._wrap_act_proxy('hidden_state_tanh_quant'),
                reverse_input=self.cell.reverse_input,
                batch_first=self.cell.batch_first,
                cifg=self.cell.cifg,
                output_quant_enabled=self.cell.output_quant.is_quant_enabled,
                cell_state_quant_enabled=self.cell.cell_state_quant.is_quant_enabled,
                fast_impl=True)
            if brevitas.config.JIT_ENABLED:
                self._fast_cell = torch.jit.script(self._fast_cell)
            return self._fast_cell

    def forward(self, inp, hidden_state, cell_state):
        quant_input = self.maybe_quantize_input(inp)
        quant_weight_ii, quant_weight_hi, quant_bias_input = self.gate_params_fwd(
            self.input_gate_params, quant_input)
        quant_weight_ic, quant_weight_hc, quant_bias_cell = self.gate_params_fwd(
            self.cell_gate_params, quant_input)
        quant_weight_io, quant_weight_ho, quant_bias_output = self.gate_params_fwd(
            self.output_gate_params, quant_input)
        if self.cifg:
            # Avoid dealing with None and set it the same as the forget one
            quant_weight_if = quant_weight_ii
            quant_weight_hf = quant_weight_hi
            quant_bias_forget = quant_bias_input
        else:
            quant_weight_if, quant_weight_hf, quant_bias_forget = self.gate_params_fwd(
                self.forget_gate_params, quant_input)
        # Handle None bias by setting it 0.
        if quant_bias_input.value is None:
            quant_bias_input = torch.tensor(0., device=quant_input.value.device)
        else:
            quant_bias_input = quant_bias_input.value
        if quant_bias_forget.value is None:
            quant_bias_forget = torch.tensor(0., device=quant_input.value.device)
        else:
            quant_bias_forget = quant_bias_forget.value
        if quant_bias_cell.value is None:
            quant_bias_cell = torch.tensor(0., device=quant_input.value.device)
        else:
            quant_bias_cell = quant_bias_cell.value
        if quant_bias_output.value is None:
            quant_bias_output = torch.tensor(0., device=quant_input.value.device)
        else:
            quant_bias_output = quant_bias_output.value
        quant_hidden_state = self.maybe_quantize_state(
            quant_input.value, hidden_state, self.cell.output_quant)
        quant_cell_state = self.maybe_quantize_state(
            quant_input.value, cell_state, self.cell.cell_state_quant)
        # Pick cell impl
        if self.export_mode:
            cell = self.export_handler
        elif self.fast_mode:
            cell = self.fast_cell
        else:
            cell = self.cell
        quant_outputs, quant_hidden_state, quant_cell_state = cell(
            quant_input.value,
            quant_hidden_state.value,
            quant_cell_state.value,
            quant_weight_ii=quant_weight_ii.value,
            quant_weight_if=quant_weight_if.value,
            quant_weight_ic=quant_weight_ic.value,
            quant_weight_io=quant_weight_io.value,
            quant_weight_hi=quant_weight_hi.value,
            quant_weight_hf=quant_weight_hf.value,
            quant_weight_hc=quant_weight_hc.value,
            quant_weight_ho=quant_weight_ho.value,
            quant_bias_input=quant_bias_input,
            quant_bias_forget=quant_bias_forget,
            quant_bias_cell=quant_bias_cell,
            quant_bias_output=quant_bias_output)
        quant_outputs = self.pack_quant_outputs(quant_outputs)
        quant_hidden_state = self.pack_quant_state(quant_hidden_state, self.cell.output_quant)
        quant_cell_state = self.pack_quant_state(quant_cell_state, self.cell.cell_state_quant)
        return quant_outputs, quant_hidden_state, quant_cell_state

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        hs = self.hidden_size

        def bias(gate_name):
            bias_name = f'{prefix}{gate_name}_gate_params.bias'
            if not bias_name in state_dict.keys():
                state_dict[bias_name] = torch.zeros(hs)
            return state_dict[bias_name]

        def _set_weight(gate_name, value, input_name):
            key = f'{prefix}{gate_name}_gate_params.{input_name}_weight.weight'
            state_dict[key] = value

        set_input_weight = partial(_set_weight, input_name='input')
        set_hidden_weight = partial(_set_weight, input_name='hidden')

        for name, value in list(state_dict.items()):
            if prefix + 'bias_ih' in name or prefix + 'bias_hh' in name:
                bias('input').add_(value[:hs])
                bias('forget').add_(value[hs:hs * 2])
                bias('cell').add_(value[2 * hs:hs * 3])
                bias('output').add_(value[3 * hs:])
                del state_dict[name]
            elif prefix + 'weight_ih' in name:
                set_input_weight('input', value[:hs, :])
                set_input_weight('forget', value[hs:hs * 2, :])
                set_input_weight('cell', value[2 * hs:hs * 3, :])
                set_input_weight('output', value[3 * hs:, :])
                del state_dict[name]
            elif prefix + 'weight_hh' in name:
                set_hidden_weight('input', value[:hs, :])
                set_hidden_weight('forget', value[hs:hs * 2, :])
                set_hidden_weight('cell', value[2 * hs:hs * 3, :])
                set_hidden_weight('output', value[3 * hs:, :])
                del state_dict[name]

        super(_QuantLSTMLayer, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class QuantRecurrentStackBase(nn.Module):

    def __init__(
            self,
            layer_impl,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            bidirectional: bool,
            io_quant,
            shared_input_hidden_weights: bool,
            return_quant_tensor: bool,
            **kwargs):
        super(QuantRecurrentStackBase, self).__init__()
        if shared_input_hidden_weights and not bidirectional:
            raise RuntimeError("Shared input-hidden weights requires bidirectional=True.")
        if return_quant_tensor and io_quant is None:
            raise RuntimeError("return_quant_tensor=True requires io_quant != None.")

        self.num_directions = 2 if bidirectional else 1
        layers = []
        # Add io_quant to kwargs. This allows easy overwriting during sharing
        kwargs['io_quant'] = io_quant
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            quantize_output_only = bool(layer)
            # return_quant_tensor is required for bias quantization of internal layers
            layer_return_quant_tensor = return_quant_tensor or layer < num_layers - 1
            directions = []
            left_to_right = layer_impl(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                reverse_input=False,
                quantize_output_only=quantize_output_only,
                shared_input_hidden_weights=shared_input_hidden_weights,
                return_quant_tensor=layer_return_quant_tensor,
                **kwargs)
            directions.append(left_to_right)
            # Update kwargs with shared quantizers. Overwrite io_quant and any
            # other quantizer that should be shared. The quantizers of the
            # first left-to-right layer are shared to all directions and layers
            kwargs.update(**left_to_right.quantizers_to_share)
            if bidirectional:
                shared_weights = left_to_right.weights_to_share
                right_to_left = layer_impl(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    reverse_input=True,
                    quantize_output_only=quantize_output_only,
                    shared_input_hidden_weights=shared_input_hidden_weights,
                    return_quant_tensor=layer_return_quant_tensor,
                    **shared_weights,
                    **kwargs)
                directions.append(right_to_left)
            layers.append(nn.ModuleList(directions))
        self.layers = nn.ModuleList(layers)

    def forward(self, inp, hx=None):
        output_states = []
        for l, layer in enumerate(self.layers):
            dir_outputs, dir_states = [], []
            for d, direction in enumerate(layer):
                layer_state = hx[2 * l + d] if hx is not None else hx
                out, out_state = direction(inp, layer_state)
                dir_outputs += [out]
                dir_states += [out_state]
            if len(dir_outputs) > 1:
                out = torch.cat(dir_outputs, dim=-1)
                output_states += [torch.cat(dir_states, dim=0)]
            else:
                out = dir_outputs[0]
                output_states += [dir_states[0]]
            inp = out
        if len(output_states) > 1:
            output_states = torch.cat(output_states, dim=0)
        else:
            output_states = output_states[0]
        return out, output_states

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
        for name, value in list(state_dict.items()):
            for index in range(len(self.layers)):
                layer_name = f'_l{index}'
                reverse_layer_name = layer_name + '_reverse'
                if reverse_layer_name in name:
                    param_name = name[len(prefix):-len(reverse_layer_name)]
                    state_dict[f'{prefix}layers.{index}.1.{param_name}'] = value
                    del state_dict[name]
                elif layer_name in name:
                    param_name = name[len(prefix):-len(layer_name)]
                    state_dict[f'{prefix}layers.{index}.0.{param_name}'] = value
                    del state_dict[name]
        super(QuantRecurrentStackBase, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class QuantRNN(QuantRecurrentStackBase):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            nonlinearity: str = 'tanh',
            bias: bool = True,
            batch_first: bool = False,
            bidirectional: bool = False,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            io_quant=Int8ActPerTensorFloat,
            gate_acc_quant=Int8ActPerTensorFloat,
            shared_input_hidden_weights=False,
            return_quant_tensor: bool = False,
            **kwargs):
        super(QuantRNN, self).__init__(
            layer_impl=_QuantRNNLayer,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            io_quant=io_quant,
            gate_acc_quant=gate_acc_quant,
            shared_input_hidden_weights=shared_input_hidden_weights,
            return_quant_tensor=return_quant_tensor,
            **kwargs)


class QuantLSTM(QuantRecurrentStackBase):

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bias: bool = True,
            batch_first: bool = False,
            bidirectional: bool = False,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            io_quant=Int8ActPerTensorFloat,
            gate_acc_quant=Int8ActPerTensorFloat,
            sigmoid_quant=Uint8ActPerTensorFloat,
            tanh_quant=Int8ActPerTensorFloat,
            cell_state_quant=Int8ActPerTensorFloat,
            coupled_input_forget_gates: bool = False,
            cat_output_cell_states=True,
            shared_input_hidden_weights=False,
            shared_intra_layer_weight_quant=False,
            shared_intra_layer_gate_acc_quant=False,
            shared_cell_state_quant=True,
            return_quant_tensor: bool = False,
            **kwargs):
        super(QuantLSTM, self).__init__(
            layer_impl=_QuantLSTMLayer,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            bidirectional=bidirectional,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            io_quant=io_quant,
            gate_acc_quant=gate_acc_quant,
            sigmoid_quant=sigmoid_quant,
            tanh_quant=tanh_quant,
            cell_state_quant=cell_state_quant,
            cifg=coupled_input_forget_gates,
            shared_input_hidden_weights=shared_input_hidden_weights,
            shared_intra_layer_weight_quant=shared_intra_layer_weight_quant,
            shared_intra_layer_gate_acc_quant=shared_intra_layer_gate_acc_quant,
            shared_cell_state_quant=shared_cell_state_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        if cat_output_cell_states and cell_state_quant is not None and not shared_cell_state_quant:
            raise RuntimeError("Concatenating cell states requires shared cell quantizers.")
        self.cat_output_cell_states = cat_output_cell_states

    def forward(self, inp, hx=None, cx=None):
        output_hidden_states, output_cell_states = [], []
        for l, layer in enumerate(self.layers):
            dir_outputs, dir_hidden_states, dir_cell_states = [], [], []
            for d, direction in enumerate(layer):
                layer_hidden_state = hx[2 * l + d] if hx is not None else hx
                layer_cell_state = cx[2 * l + d] if cx is not None else cx
                out, out_hidden_state, out_cell_state = direction(inp, layer_hidden_state, layer_cell_state)
                dir_outputs += [out]
                dir_hidden_states += [out_hidden_state]
                dir_cell_states += [out_cell_state]
            if len(dir_outputs) > 1:
                out = torch.cat(dir_outputs, dim=-1)
                output_hidden_states += [torch.cat(dir_hidden_states, dim=0)]
                if self.cat_output_cell_states:
                    output_cell_states += [torch.cat(dir_cell_states, dim=0)]
                else:
                    output_cell_states.extend(dir_cell_states)
            else:
                out = dir_outputs[0]
                output_hidden_states += [dir_hidden_states[0]]
                if self.cat_output_cell_states:
                    output_cell_states += [dir_cell_states[0]]
                else:
                    output_cell_states.extend(dir_cell_states)
            inp = out
        if len(output_hidden_states) > 1:
            output_hidden_states = torch.cat(output_hidden_states, dim=0)
            if self.cat_output_cell_states:
                output_cell_states = torch.cat(output_cell_states, dim=0)
        else:
            output_hidden_states = output_hidden_states[0]
            if self.cat_output_cell_states:
                output_cell_states = output_cell_states[0]
        return out, (output_hidden_states, output_cell_states)
