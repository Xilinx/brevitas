# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU, 
#    NEC Laboratories America and IDIAP Research Institute nor the names 
#    of its contributors may be used to endorse or promote products derived 
#    from this software without specific prior written permission.

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
import torch
import warnings

from torch.nn import Module
from torch.nn.parameter import Parameter
from quantization.utils.rnn import PackedSequence

from quantization.function.quantization_scheme import WeightQuantizationScheme, ActivationQuantizationScheme, BiasQuantizationScheme
from quantization.function.quantized_fused_rnn import QuantizedAutogradRNN
from quantization.function.quantized_linear import quantized_linear_hls_weight_string, quantized_linear_hls_bias_string

class QuantizedRNNBase(Module):

    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False, 
                 weight_bit_width=32, weight_q_type='FP', 
                 bias_bit_width=32, bias_q_type='FP', 
                 activation_bit_width=32, activation_q_type='FP',
                 internal_activation_bit_width=32):
        super(QuantizedRNNBase, self).__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        self.weight_fixed_scale_factor = 1.0 / math.sqrt(self.input_size + self.hidden_size) #used in scaled binary q_type
        self.weight_quantization_scheme = WeightQuantizationScheme(weight_bit_width, weight_q_type, scale_factor=self.weight_fixed_scale_factor)
        self.bias_quantization_scheme = BiasQuantizationScheme(bias_bit_width, bias_q_type)
        self.activation_quantization_scheme = ActivationQuantizationScheme(activation_bit_width, activation_q_type)

        if internal_activation_bit_width == 32:
            self.tanh_quantization_scheme = ActivationQuantizationScheme(internal_activation_bit_width, 'FP')
            self.sigmoid_quantization_scheme = ActivationQuantizationScheme(internal_activation_bit_width, 'FP') 
        else:
            self.tanh_quantization_scheme = ActivationQuantizationScheme(internal_activation_bit_width, 'SIGNED_FIXED_UNIT')
            self.sigmoid_quantization_scheme = ActivationQuantizationScheme(internal_activation_bit_width, 'UNSIGNED_FIXED_UNIT')

        if mode == 'QLSTM':
            gate_size = 4 * hidden_size
        elif mode == 'QGRU':
            gate_size = 3 * hidden_size
        else:
            gate_size = hidden_size

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions

                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size))
                w_hh = Parameter(torch.Tensor(gate_size, hidden_size))
                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))
                layer_params = (w_ih, w_hh, b_ih, b_hh)

                suffix = '_reverse' if direction == 1 else ''
                param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.flatten_parameters()
        self.reset_parameters()

    @property
    def num_directions(self):
        return 2 if self.bidirectional else 1

    def flatten_parameters(self):
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
        any_param = next(self.parameters()).data
        if not any_param.is_cuda or not torch.backends.cudnn.is_acceptable(any_param):
            self._data_ptrs = []
            return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for l in self.all_weights for p in l)
        if len(unique_data_ptrs) != sum(len(l) for l in self.all_weights):
            self._data_ptrs = []
            return

        with torch.cuda.device_of(any_param):
            # This is quite ugly, but it allows us to reuse the cuDNN code without larger
            # modifications. It's really a low-level API that doesn't belong in here, but
            # let's make this exception.
            from torch.backends.cudnn import rnn
            from torch.backends import cudnn
            from torch.nn._functions.rnn import CudnnRNN
            handle = cudnn.get_handle()
            with warnings.catch_warnings(record=True):
                fn = CudnnRNN(
                    self.mode,
                    self.input_size,
                    self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=self.batch_first,
                    dropout=self.dropout,
                    train=self.training,
                    bidirectional=self.bidirectional,
                    dropout_state=self.dropout_state
                )

            # Initialize descriptors
            fn.datatype = cudnn._typemap[any_param.type()]
            fn.x_descs = cudnn.descriptor(any_param.new(1, self.input_size), 1)
            fn.rnn_desc = rnn.init_rnn_descriptor(fn, handle)

            # Allocate buffer to hold the weights
            self._param_buf_size = rnn.get_num_weights(handle, fn.rnn_desc, fn.x_descs[0], fn.datatype)
            fn.weight_buf = any_param.new(self._param_buf_size).zero_()
            fn.w_desc = rnn.init_weight_descriptor(fn, fn.weight_buf)

            # Slice off views into weight_buf
            all_weights = [[p.data for p in l] for l in self.all_weights]
            params = rnn.get_parameters(fn, handle, fn.weight_buf)

            # Copy weights and update their storage
            rnn._copyParams(all_weights, params)
            for orig_layer_param, new_layer_param in zip(all_weights, params):
                for orig_param, new_param in zip(orig_layer_param, new_layer_param):
                    orig_param.set_(new_param.view_as(orig_param))

            self._data_ptrs = list(p.data.data_ptr() for p in self.parameters())

    def _apply(self, fn):
        ret = super(QuantizedRNNBase, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            hx = torch.autograd.Variable(input.data.new(self.num_layers *
                                                        self.num_directions,
                                                        max_batch_size,
                                                        self.hidden_size).zero_(), requires_grad=False)
            if self.mode == 'QLSTM':
                hx = (hx, hx)

        has_flat_weights = list(p.data.data_ptr() for p in self.parameters()) == self._data_ptrs
        if has_flat_weights:
            first_data = next(self.parameters()).data
            assert first_data.storage().size() == self._param_buf_size
            flat_weight = first_data.new().set_(first_data.storage(), 0, torch.Size([self._param_buf_size]))
        else:
            flat_weight = None
        func = QuantizedAutogradRNN(
            self.mode,
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            batch_sizes=batch_sizes,
            dropout_state=self.dropout_state,
            flat_weight=flat_weight,
            weight_quantization_scheme=self.weight_quantization_scheme,
            bias_quantization_scheme=self.bias_quantization_scheme,
            activation_quantization_scheme=self.activation_quantization_scheme,
            tanh_quantization_scheme=self.tanh_quantization_scheme,
            sigmoid_quantization_scheme=self.sigmoid_quantization_scheme
        )
        output, hidden = func(input, self.all_weights, hx)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def __setstate__(self, d):
        super(QuantizedRNNBase, self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs', [])
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(self.num_directions):
                suffix = '_reverse' if direction == 1 else ''
                weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}', 'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

    '''
    For the time being export methods works only for 1 layer
    '''
    def hls_lstm_weight_ih_string(self, simd, pe):
        if self.bidirectional:
            return self.hls_lstm_weight_string(self.weight_ih_l0.data, self.weight_ih_l0_reverse.data, "ih", simd, pe)
        else:
            return self.hls_lstm_weight_string(self.weight_ih_l0.data, None, "ih", simd, pe)

    '''
    For the time being export methods works only for 1 layer
    '''
    def hls_lstm_weight_hh_string(self, simd, pe):
        if self.bidirectional:
            return self.hls_lstm_weight_string(self.weight_hh_l0.data, self.weight_hh_l0_reverse.data, "hh", simd, pe)
        else:
            return self.hls_lstm_weight_string(self.weight_hh_l0.data, None, "hh", simd, pe)

    '''
    For the time being export methods works only for 1 layer
    '''
    def hls_lstm_weight_string(self, w, w_reverse, postfix, simd, pe):
        if w_reverse is not None:
            wgi = torch.cat([w[0:self.hidden_size,:], w_reverse[0:self.hidden_size,:]], 0).t()
            wgf = torch.cat([w[self.hidden_size:2*self.hidden_size,:], w_reverse[self.hidden_size:2*self.hidden_size,:]], 0).t()
            wci = torch.cat([w[2*self.hidden_size:3*self.hidden_size,:], w_reverse[2*self.hidden_size:3*self.hidden_size,:]], 0).t()
            wgo = torch.cat([w[3*self.hidden_size:4*self.hidden_size,:], w_reverse[3*self.hidden_size:4*self.hidden_size,:]], 0).t()
        else:
            wgi = w[0:self.hidden_size,:].t()
            wgf = w[self.hidden_size:2*self.hidden_size,:].t()
            wci = w[2*self.hidden_size:3*self.hidden_size,:].t()
            wgo = w[3*self.hidden_size:4*self.hidden_size,:].t()

        wgi_string = quantized_linear_hls_weight_string(wgi, self.weight_quantization_scheme, "wgi_{}".format(postfix), simd, pe)
        wgf_string = quantized_linear_hls_weight_string(wgf, self.weight_quantization_scheme, "wgf_{}".format(postfix), simd, pe)
        wci_string = quantized_linear_hls_weight_string(wci, self.weight_quantization_scheme, "wci_{}".format(postfix), simd, pe)
        wgo_string = quantized_linear_hls_weight_string(wgo, self.weight_quantization_scheme, "wgo_{}".format(postfix), simd, pe)

        return (wgi_string, wgf_string, wgo_string, wci_string)
    
    def hls_lstm_bias_strings(self, pe):
        if self.bias:
            b_ih = self.bias_ih_l0.data
            b_hh = self.bias_hh_l0.data

            if self.bidirectional:
                b_ih_reverse = self.bias_ih_l0_reverse.data
                b_hh_reverse = self.bias_hh_l0_reverse.data

            bgi_ih = torch.cat([b_ih[0:self.hidden_size], b_ih_reverse[0:self.hidden_size]], 0).expand(1, -1)
            bgf_ih = torch.cat([b_ih[self.hidden_size:2*self.hidden_size], b_ih_reverse[self.hidden_size:2*self.hidden_size]], 0).expand(1, -1)
            bci_ih = torch.cat([b_ih[2*self.hidden_size:3*self.hidden_size], b_ih_reverse[2*self.hidden_size:3*self.hidden_size]], 0).expand(1, -1)
            bgo_ih = torch.cat([b_ih[3*self.hidden_size:4*self.hidden_size], b_ih_reverse[3*self.hidden_size:4*self.hidden_size]], 0).expand(1, -1)

            bgi_hh = torch.cat([b_hh[0:self.hidden_size], b_hh_reverse[0:self.hidden_size]], 0).expand(1, -1)
            bgf_hh = torch.cat([b_hh[self.hidden_size:2*self.hidden_size], b_hh_reverse[self.hidden_size:2*self.hidden_size]], 0).expand(1, -1)
            bci_hh = torch.cat([b_hh[2*self.hidden_size:3*self.hidden_size], b_hh_reverse[2*self.hidden_size:3*self.hidden_size]], 0).expand(1, -1)
            bgo_hh = torch.cat([b_hh[3*self.hidden_size:4*self.hidden_size], b_hh_reverse[3*self.hidden_size:4*self.hidden_size]], 0).expand(1, -1)
        
            bgi_i_string = quantized_linear_hls_bias_string(bgi_ih, self.bias_quantization_scheme, "bgi_ih", pe)
            bgf_i_string = quantized_linear_hls_bias_string(bgf_ih, self.bias_quantization_scheme, "bgf_ih", pe)
            bci_i_string = quantized_linear_hls_bias_string(bci_ih, self.bias_quantization_scheme, "bci_ih", pe)
            bgo_i_string = quantized_linear_hls_bias_string(bgo_ih, self.bias_quantization_scheme, "bgo_ih", pe)

            bgi_h_string = quantized_linear_hls_bias_string(bgi_hh, self.bias_quantization_scheme, "bgi_hh", pe)
            bgf_h_string = quantized_linear_hls_bias_string(bgf_hh, self.bias_quantization_scheme, "bgf_hh", pe)
            bci_h_string = quantized_linear_hls_bias_string(bci_hh, self.bias_quantization_scheme, "bci_hh", pe)
            bgo_h_string = quantized_linear_hls_bias_string(bgo_hh, self.bias_quantization_scheme, "bgo_hh", pe)
            
            return (bgi_i_string, bgf_i_string, bgo_i_string, bci_i_string, bgi_h_string, bgf_h_string, bgo_h_string, bci_h_string)
        
        else:
            return None


class QuantizedGRU(QuantizedRNNBase):

    def __init__(self, *args, **kwargs):
        super(QuantizedGRU, self).__init__('QGRU', *args, **kwargs)


class QuantizedLSTM(QuantizedRNNBase):

    def __init__(self, *args, **kwargs):
        super(QuantizedLSTM, self).__init__('QLSTM', *args, **kwargs)
