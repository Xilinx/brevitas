# Copyright (c) 2018-     Xilinx, Inc             

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx nor the names of its contributors 
#    may be used to endorse or promote products derived from this 
#    software without specific prior written permission.

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

from functools import partial

import torch
from quantization.function.quantized_linear import quantized_linear
from torch.autograd import Variable
from torch.autograd.function import Function, InplaceFunction, once_differentiable
import torch.nn.functional as F
import xilinx.torch

class QuantizedFusedGRU(Function):
    @staticmethod
    def forward(ctx, input_gate, hidden_gate, hx, q_ibias=False, q_hbias=None):

        hy = input_gate.new()
        workspace = input_gate.new(hx.numel() * 5)

        ctx.has_bias = False
        if q_ibias is not None:
            ctx.has_bias = True
            if q_ibias.dim() == 1:
                q_ibias = q_ibias.unsqueeze(0)
            if q_hbias.dim() == 1:
                q_hbias = q_hbias.unsqueeze(0)

            if isinstance(input_gate, torch.cuda.FloatTensor):
                with torch.cuda.device_of(input_gate):
                    xilinx.torch.quantized_fused_gru_Cudaforward(
                        input_gate, hidden_gate, q_ibias, q_hbias, hx, hy, workspace)

            elif isinstance(input_gate, torch.cuda.DoubleTensor):
                with torch.cuda.device_of(input_gate):
                    xilinx.torch.quantized_fused_gru_CudaDoubleforward(
                        input_gate, hidden_gate, q_ibias, q_hbias, hx, hy, workspace)
            else:
                raise RuntimeError('Unimplemented Tensor type.')

        else:
            if isinstance(input_gate, torch.cuda.FloatTensor):
                with torch.cuda.device_of(input_gate):        
                    xilinx.torch.quantized_fused_gru_nobias_Cudaforward(
                        input_gate, hidden_gate, hx, hy, workspace)

            elif isinstance(input_gate, torch.cuda.DoubleTensor):
                with torch.cuda.device_of(input_gate):        
                    xilinx.torch.quantized_fused_gru_nobias_CudaDoubleforward(
                        input_gate, hidden_gate, hx, hy, workspace)
            else:
                raise RuntimeError('Unimplemented Tensor type.')

        ctx.workspace = workspace
        ctx.igate_size = input_gate.size()
        ctx.hgate_size = hidden_gate.size()

        return hy

    @staticmethod
    @once_differentiable
    def backward(ctx, gradOutput):

        gradInputHx = gradOutput.new()
        gradInInput = gradOutput.new(*ctx.igate_size)
        gradInHidden = gradOutput.new(*ctx.hgate_size)

        if isinstance(gradInInput, torch.cuda.FloatTensor):
            with torch.cuda.device_of(gradInInput):
                xilinx.torch.quantized_fused_gru_Cudabackward(
                    gradInInput, gradInHidden, gradOutput, gradInputHx, ctx.workspace)

        elif isinstance(gradInInput, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(gradInInput):
                xilinx.torch.quantized_fused_gru_CudaDoublebackward(
                    gradInInput, gradInHidden, gradOutput, gradInputHx, ctx.workspace)

        else:
            raise RuntimeError('Unimplemented Tensor type.')

        gb1 = gb2 = None
        if ctx.has_bias:
            gb1 = gradInInput.sum(0, keepdim=False)
            gb2 = gradInHidden.sum(0, keepdim=False)
        return gradInInput, gradInHidden, gradInputHx, gb1, gb2


class QuantizedFusedLSTM(Function):

    @staticmethod
    def forward(ctx, input_gate, hidden_gate, cx, q_ibias=None, q_hbias=None, internal_activation_bit_width=None):
        hy = input_gate.new()
        cy = input_gate.new()

        ctx.has_bias = False
        if q_ibias is not None:
            ctx.has_bias = True
            if q_ibias.dim() == 1:
                q_ibias = q_ibias.unsqueeze(0)
            if q_hbias.dim() == 1:
                q_hbias = q_hbias.unsqueeze(0)

            # input_gate gets overwritten with some intermediate values to use in backwards
            if isinstance(input_gate, torch.cuda.FloatTensor):
                bit_width_tensor = torch.cuda.FloatTensor([internal_activation_bit_width])
                with torch.cuda.device_of(input_gate):        
                    xilinx.torch.quantized_fused_lstm_Cudaforward(
                        input_gate, hidden_gate, q_ibias, q_hbias, cx, hy, cy, bit_width_tensor)

            elif isinstance(input_gate, torch.cuda.DoubleTensor):
                bit_width_tensor = torch.cuda.DoubleTensor([internal_activation_bit_width])
                with torch.cuda.device_of(input_gate):        
                    xilinx.torch.quantized_fused_lstm_CudaDoubleforward(
                        input_gate, hidden_gate, q_ibias, q_hbias, cx, hy, cy, bit_width_tensor)
            else:
                raise RuntimeError('Unimplemented Tensor type.')
        else:
            # input_gate gets overwritten with some intermediate values to use in backwards
            if isinstance(input_gate, torch.cuda.FloatTensor):
                bit_width_tensor = torch.cuda.FloatTensor([internal_activation_bit_width])
                with torch.cuda.device_of(input_gate):        
                    xilinx.torch.quantized_fused_lstm_nobias_Cudaforward(
                        input_gate, hidden_gate, cx, hy, cy, bit_width_tensor)

            elif isinstance(input_gate, torch.cuda.DoubleTensor):
                bit_width_tensor = torch.cuda.DoubleTensor([internal_activation_bit_width])
                with torch.cuda.device_of(input_gate):        
                    xilinx.torch.quantized_fused_lstm_nobias_CudaDoubleforward(
                        input_gate, hidden_gate, cx, hy, cy, bit_width_tensor)
            else:
                raise RuntimeError('Unimplemented Tensor type.')

        ctx.hgate_size = hidden_gate.size()
        ctx.save_for_backward(input_gate, cx, cy)

        return hy, cy

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradOutput):
        gradInputCx = gradOutput[0].new()
        gradInGates = gradOutput[0].new(*ctx.hgate_size)

        saved_tens, cx, cy = ctx.saved_tensors

        if isinstance(gradInputCx, torch.cuda.FloatTensor):
            with torch.cuda.device_of(gradInputCx):
                xilinx.torch.quantized_fused_lstm_Cudabackward(
                    saved_tens, gradInGates, cx, cy,
                    gradOutput[0], gradOutput[1], gradInputCx)

        elif isinstance(gradInputCx, torch.cuda.DoubleTensor):
            with torch.cuda.device_of(gradInputCx):
                xilinx.torch.quantized_fused_lstm_CudaDoublebackward(
                    saved_tens, gradInGates, cx, cy,
                    gradOutput[0], gradOutput[1], gradInputCx)

        else:
            raise RuntimeError('Unimplemented Tensor type.')

        gb1 = gb2 = None
        if ctx.has_bias:
            gb1 = gradInGates.sum(0, keepdim=False)
            gb2 = gradInGates.sum(0, keepdim=False)

        return gradInGates, gradInGates, gradInputCx, gb1, gb2, None


def QuantizedLSTMCell(input, hidden, w_ih, w_hh, 
                      b_ih=None, b_hh=None,
                      weight_quantization_scheme=None, 
                      activation_quantization_scheme=None,
                      bias_quantization_scheme=None,
                      tanh_quantization_scheme=None,
                      sigmoid_quantization_scheme=None):
    if input.is_cuda:
        igates = quantized_linear(input, w_ih, weight_quantization_scheme=weight_quantization_scheme)
        hgates = quantized_linear(hidden[0], w_hh, weight_quantization_scheme=weight_quantization_scheme)
        state = QuantizedFusedLSTM.apply
        if b_ih is None:
            output = state(igates, hgates, hidden[1], tanh_quantization_scheme.q_params.bit_width)  
        else:
            bias_quantization_scheme.fp_transform_(b_ih)
            bias_quantization_scheme.fp_transform_(b_hh)
            q_b_ih = bias_quantization_scheme.q_forward(b_ih)
            q_b_hh = bias_quantization_scheme.q_forward(b_hh)
            output = state(igates, hgates, hidden[1], q_b_ih, q_b_hh, tanh_quantization_scheme.q_params.bit_width)
        quantized_h = activation_quantization_scheme.q_forward(output[0])
        return (quantized_h, output[1])
    else:
        
        igates = quantized_linear(input, w_ih, weight_quantization_scheme=weight_quantization_scheme)
        hgates = quantized_linear(hidden[0], w_hh, weight_quantization_scheme=weight_quantization_scheme)
        
        def state(igates, hgates, cell_state, i_bias=None, h_bias=None):
            hidden_size = igates.size()[1] / 4 #first dimension is batch size
            if i_bias is not None and h_bias is not None:
                gates = igates + hgates + i_bias + h_bias
            else:
                gates = igates + hgates
            ig = sigmoid_quantization_scheme.q_forward(torch.nn.functional.sigmoid(gates[:, 0:hidden_size]))
            fg = sigmoid_quantization_scheme.q_forward(torch.nn.functional.sigmoid(gates[:, hidden_size:2*hidden_size]))
            cg = tanh_quantization_scheme.q_forward(torch.nn.functional.tanh(gates[:, 2*hidden_size:3*hidden_size]))
            og = sigmoid_quantization_scheme.q_forward(torch.nn.functional.sigmoid(gates[:, 3*hidden_size:4*hidden_size]))
            cg = ig * cg
            fg = cell_state * fg
            cg = fg + cg
            og = tanh_quantization_scheme.q_forward(torch.nn.functional.tanh(cg)) * og
            return (og, cg)

        if b_ih is None:
            output = state(igates, hgates, hidden[1])  
        else:
            bias_quantization_scheme.fp_transform_(b_ih)
            bias_quantization_scheme.fp_transform_(b_hh)
            q_b_ih = bias_quantization_scheme.q_forward(b_ih)
            q_b_hh = bias_quantization_scheme.q_forward(b_hh)
            output = state(igates, hgates, hidden[1], q_b_ih, q_b_hh)
        quantized_h = activation_quantization_scheme.q_forward(output[0])
        return (quantized_h, output[1])
        
        


def QuantizedGRUCell(input, hidden, w_ih, w_hh,
                     b_ih=None, b_hh=None,
                     weight_quantization_scheme=None, 
                     activation_quantization_scheme=None,
                     bias_quantization_scheme=None):

    if input.is_cuda:
        gi = quantized_linear(input, w_ih, weight_quantization_scheme=weight_quantization_scheme)
        gh = quantized_linear(hidden, w_hh, weight_quantization_scheme=weight_quantization_scheme)
        state = QuantizedFusedGRU.apply
        if b_ih is None:
            output = state(gi, gh, hidden)  
        else:
            bias_quantization_scheme.fp_transform_(b_ih)
            bias_quantization_scheme.fp_transform_(b_hh)
            q_b_ih = bias_quantization_scheme.q_forward(b_ih)
            q_b_hh = bias_quantization_scheme.q_forward(b_hh)
            output = state(gi, gh, hidden, q_b_ih, q_b_hh)
        return activation_quantization_scheme.q_forward(output)

    else:
        raise Exception('No QuantizedGRUCell CPU implementation')


def QuantizedAutogradRNN(mode, input_size, hidden_size, weight_quantization_scheme, 
                activation_quantization_scheme, sigmoid_quantization_scheme, 
                tanh_quantization_scheme, num_layers=1, batch_first=False,
                dropout=0, train=True, bidirectional=False, batch_sizes=None,
                dropout_state=None, flat_weight=None, bias_quantization_scheme=None):

    if mode == 'QLSTM':
        cell = partial(QuantizedLSTMCell, 
                       weight_quantization_scheme=weight_quantization_scheme,
                       bias_quantization_scheme=bias_quantization_scheme,
                       activation_quantization_scheme=activation_quantization_scheme,
                       sigmoid_quantization_scheme=sigmoid_quantization_scheme,
                       tanh_quantization_scheme=tanh_quantization_scheme)
    elif mode == 'QGRU':
        cell = partial(QuantizedGRUCell,
                       weight_quantization_scheme=weight_quantization_scheme,
                       bias_quantization_scheme=bias_quantization_scheme,
                       activation_quantization_scheme=activation_quantization_scheme)
    else:
        raise Exception('Unknown mode: {}'.format(mode))

    if batch_sizes is None:
        rec_factory = Recurrent
    else:
        rec_factory = variable_recurrent_factory(batch_sizes)

    if bidirectional:
        layer = (rec_factory(cell), rec_factory(cell, reverse=True))
    else:
        layer = (rec_factory(cell),)

    func = StackedRNN(layer,
                      num_layers,
                      (mode == 'QLSTM'),
                      dropout=dropout,
                      train=train)

    def forward(input, weight, hidden):
        if batch_first and batch_sizes is None:
            input = input.transpose(0, 1)

        nexth, output = func(input, hidden, weight)

        if batch_first and batch_sizes is None:
            output = output.transpose(0, 1)

        return output, nexth

    return forward



def StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):

    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight):
        assert(len(weight) == total_layers)
        next_hidden = []

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j

                hy, output = inner(input, hidden[l], weight[l])
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, input.dim() - 1)

            if dropout != 0 and i < num_layers - 1:
                input = F.dropout(input, p=dropout, training=train, inplace=False)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(
                total_layers, *next_hidden[0].size())

        return next_hidden, input

    return forward


def Recurrent(inner, reverse=False):
    def forward(input, hidden, weight):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward


def variable_recurrent_factory(batch_sizes):
    def fac(inner, reverse=False):
        if reverse:
            return VariableRecurrentReverse(batch_sizes, inner)
        else:
            return VariableRecurrent(batch_sizes, inner)
    return fac


def VariableRecurrent(batch_sizes, inner):
    def forward(input, hidden, weight):
        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        for batch_size in batch_sizes:
            step_input = input[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)

            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()

        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)

        return hidden, output

    return forward


def VariableRecurrentReverse(batch_sizes, inner):
    def forward(input, hidden, weight):
        output = []
        input_offset = input.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[:batch_sizes[-1]] for h in hidden)
        for batch_size in reversed(batch_sizes):
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden = tuple(torch.cat((h, ih[last_batch_size:batch_size]), 0)
                               for h, ih in zip(hidden, initial_hidden))
            last_batch_size = batch_size
            step_input = input[input_offset - batch_size:input_offset]
            input_offset -= batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)
            output.append(hidden[0])

        output.reverse()
        output = torch.cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        return hidden, output

    return forward