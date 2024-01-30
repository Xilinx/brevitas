# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod

import torch
from torch import Tensor

from brevitas.export.common.handler.base import BaseHandler
from brevitas.export.onnx import onnx_export_opset
from brevitas.export.onnx.debug import DebugMarkerFunction
from brevitas.nn.quant_rnn import _QuantLSTMLayer

from .function import LSTMCellFn

__all__ = [
    'Kernel1dApplHandlerMixin',
    'Kernel2dApplHandlerMixin',
    'ONNXBaseHandler',
    'QuantLSTMLayerHandler']


class Kernel1dApplHandlerMixin(ABC):

    @staticmethod
    def padding(module):
        if isinstance(module.padding, int):
            padding = [module.padding] * 2
        else:
            padding = list(module.padding)
            if len(padding) == 1:
                return padding + padding
        return padding

    @staticmethod
    def stride(module):
        if isinstance(module.stride, int):
            return [module.stride]
        else:
            return list(module.stride)

    @staticmethod
    def dilation(module):
        if isinstance(module.dilation, int):
            return [module.dilation]
        else:
            dilation = list(module.dilation)
            return dilation

    @staticmethod
    def kernel_shape(module):
        if isinstance(module.kernel_size, int):
            return [module.kernel_size]
        else:
            return list(module.kernel_size)


class Kernel2dApplHandlerMixin(ABC):

    @staticmethod
    def padding(module):
        if isinstance(module.padding, int):
            padding = [module.padding] * 4
        else:
            padding = list(module.padding) + list(module.padding)
        return padding

    @staticmethod
    def stride(module):
        if isinstance(module.stride, int):
            return [module.stride] * 2
        else:
            return list(module.stride)

    @staticmethod
    def dilation(module):
        if isinstance(module.dilation, int):
            return [module.dilation] * 2
        else:
            return list(module.dilation)

    @staticmethod
    def kernel_shape(module):
        if isinstance(module.kernel_size, int):
            return [module.kernel_size] * 2
        else:
            return list(module.kernel_size)


class ONNXBaseHandler(BaseHandler, ABC):

    def __init__(self):
        super().__init__()
        self.symbolic_kwargs = {}
        self.export_debug_name = None
        self.debug_input = False
        self.debug_output = False

    @abstractmethod
    def symbolic_execution(self, *args, **kwargs):
        pass

    def attach_debug_info(self, m):
        self.export_debug_name = m.export_debug_name
        self.debug_input = m.export_input_debug
        self.debug_output = m.export_output_debug

    def forward(self, inp: Tensor, *args, **kwargs):
        debug_fn = lambda x, name: DebugMarkerFunction.apply(x, self.export_debug_name + name)
        if self.export_debug_name is not None and self.debug_input:
            inp = debug_fn(inp, '.input')
        out = self.symbolic_execution(inp, *args, **kwargs)
        if self.export_debug_name is not None and self.debug_output:
            if isinstance(out, Tensor):
                out = debug_fn(out, '.output')
            elif isinstance(out, tuple) and isinstance(out[0], Tensor):
                out = list(out)
                out[0] = debug_fn(out[0], '.output')
                out = tuple(out)
        return out


class QuantLSTMLayerHandler(ONNXBaseHandler, ABC):
    handled_layer = _QuantLSTMLayer

    def prepare_for_export(self, module: _QuantLSTMLayer):
        self.symbolic_kwargs = {
            'batch_first': module.cell.batch_first,
            'reverse_input': module.cell.reverse_input,
            'cifg': module.cifg}
        quantizers = [
            'output',
            'cell_state',
            'input_acc',
            'forget_acc',
            'cell_acc',
            'output_acc',
            'input_sigmoid',
            'forget_sigmoid',
            'cell_tanh',
            'output_sigmoid',
            'hidden_state_tanh']
        quantizers = {name: getattr(module.cell, name + '_quant') for name in quantizers}

        if all([q.is_quant_enabled for q in quantizers.values()]):
            self.quantized_cell = True
        elif all([not q.is_quant_enabled for q in quantizers.values()]):
            self.quantized_cell = False
        else:
            raise RuntimeError("Export of a partially quantized LSTM cell not supported.")

        if self.quantized_cell:
            for (name, quant) in quantizers.items():
                # ONNX export doesn't handle optional values well, so in case of cifg
                # we duplicate the forget ones to the input ones
                if module.cifg and name == 'input_acc':
                    quant = quantizers['forget_acc']
                elif module.cifg and name == 'input_sigmoid':
                    quant = quantizers['forget_sigmoid']
                self.symbolic_kwargs[name + '_scale'] = quant.scale()
                self.symbolic_kwargs[name + '_zero_point'] = quant.zero_point()
                self.symbolic_kwargs[name + '_bit_width'] = quant.bit_width()
                self.symbolic_kwargs[name + '_narrow_range'] = quant.is_narrow_range
                self.symbolic_kwargs[name + '_signed'] = quant.is_signed
                self.symbolic_kwargs[name + '_rounding_mode'] = quant.rounding_mode

    @abstractmethod
    def quantized_cell_symbolic_execution(
            self,
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
            quant_bias_output):
        pass

    def symbolic_execution(
            self,
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
            quant_bias_output):
        if self.quantized_cell:
            return self.quantized_cell_symbolic_execution(
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
        else:
            if onnx_export_opset() < 14:
                raise RuntimeError("Export of float LSTM cell requires at least opset_version=14.")
            # The ONNX standard requires parameters to have shape
            # weight_i: [num_directions, 4*hidden_size, input_size]
            # weight_h: [num_directions, 4*hidden_size, hidden_size]
            # bias: [num_directions, 8*hidden_size]
            quant_weight_i = torch.cat(
                [quant_weight_ii, quant_weight_io, quant_weight_if, quant_weight_ic], dim=0)
            quant_weight_h = torch.cat(
                [quant_weight_hi, quant_weight_ho, quant_weight_hf, quant_weight_hc], dim=0)
            quant_bias = torch.cat(
                [quant_bias_input, quant_bias_output, quant_bias_forget, quant_bias_cell], dim=0)
            # The ONNX standards assumes separate hidden biases, we set them to 0
            quant_bias = torch.cat([quant_bias, torch.zeros_like(quant_bias)], dim=0)
            # Add a leading dimension for the direction
            quant_weight_i = quant_weight_i.unsqueeze(0)
            quant_weight_h = quant_weight_h.unsqueeze(0)
            quant_bias = quant_bias.unsqueeze(0)
            # Compute relevant dimensions
            seq_len = quant_input.size(int(self.symbolic_kwargs['batch_first']))
            hidden_size = int(quant_hidden_state.size(1))
            # sSquence_lens is a tensor of dimension batch size with the sequence length
            # of each element in the batch. We don't support variable sequence length yet
            # so they are set all to the same value
            sequence_lens = torch.empty(
                quant_hidden_state.size(0), dtype=torch.int32).fill_(seq_len)
            # Initial hidden and cell state have an extra direction dimension and
            #  different shapes depending on whether batch_first is set or not
            quant_hidden_state = quant_hidden_state.unsqueeze(0)
            quant_cell_state = quant_cell_state.unsqueeze(0)
            if self.symbolic_kwargs['batch_first']:
                quant_hidden_state = quant_hidden_state.permute(1, 0, 2)
                quant_cell_state = quant_cell_state.permute(1, 0, 2)
            # The ONNX standards requires an extra dimension for the direction,
            # which we don't want here, so we squeeze it. We compute the Squeeze op
            # manually within the function since tracing tend to overcomplicate things
            if self.symbolic_kwargs['batch_first']:
                output_dir_axes = torch.tensor([1], dtype=torch.int64)
                state_dir_axes = torch.tensor([1], dtype=torch.int64)
            else:
                output_dir_axes = torch.tensor([2], dtype=torch.int64)
                state_dir_axes = torch.tensor([0], dtype=torch.int64)
            return LSTMCellFn.apply(
                quant_input,
                quant_weight_i,
                quant_weight_h,
                quant_bias,
                sequence_lens,
                quant_hidden_state,
                quant_cell_state,
                'reverse' if self.symbolic_kwargs['reverse_input'] else 'forward',
                hidden_size,
                self.symbolic_kwargs['cifg'],
                self.symbolic_kwargs['batch_first'],
                output_dir_axes,
                state_dir_axes)
