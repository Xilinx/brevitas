from abc import ABC
from copy import copy

import torch
from torch import Tensor

from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.proxy import WeightQuantProxyFromInjector
from brevitas.proxy import DecoupledWeightQuantProxyFromInjector
from brevitas.proxy import BiasQuantProxyFromInjector
from brevitas.proxy import ActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import TruncQuantProxyFromInjector
from brevitas.nn.quant_rnn import _QuantLSTMLayer

from .function import BrevitasQuantFn
from .function import BrevitasBinaryQuantFn
from .function import BrevitasTruncFn
from .function import BrevitasQuantLSTMCellFn
from .function import LSTMCellFn


class BrevitasQuantProxyHandler(ONNXBaseHandler, ABC):

    def validate(self, module):
        if module.bit_width() == 1:
            assert module.zero_point() == 0, "Zero-point not supported for binary quant."

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.validate(module)
            self.symbolic_kwargs = {
                'scale': module.scale(),
                'zero_point': module.zero_point(),
                'bit_width': module.bit_width(),
                'narrow_range': module.is_narrow_range,
                'signed': module.is_signed,
                'rounding_mode': module.rounding_mode}

    def symbolic_execution(self, x: Tensor):
        scale = self.symbolic_kwargs['scale']
        zero_point = self.symbolic_kwargs['zero_point']
        bit_width = self.symbolic_kwargs['bit_width']
        if bit_width == 1:
            x = BrevitasBinaryQuantFn.apply(x, *self.symbolic_kwargs.values())
        else:
            x = BrevitasQuantFn.apply(x, *self.symbolic_kwargs.values())
        return x, scale, zero_point, bit_width


class BrevitasWeightQuantProxyHandler(BrevitasQuantProxyHandler):
    handled_layer = WeightQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.quant_weights = None

    def prepare_for_export(self, module: WeightQuantProxyFromInjector):
        super().prepare_for_export(module)
        quant_weights = {
            tm.weight.data_ptr(): tm.quant_weight().value for tm in module.tracked_module_list}
        self.quant_weights = quant_weights
        # override rounding mode since quantization has been pre-applied
        self.symbolic_kwargs['rounding_mode'] = 'ROUND'

    def symbolic_execution(self, x: Tensor):
        quant_weight = self.quant_weights[x.data_ptr()]
        return super().symbolic_execution(quant_weight)


class BrevitasDecoupledWeightQuantProxyHandler(BrevitasWeightQuantProxyHandler):
    handled_layer = DecoupledWeightQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.extra_kwargs = {}

    def prepare_for_export(self, module: DecoupledWeightQuantProxyFromInjector):
        super().prepare_for_export(module)
        self.extra_kwargs['pre_scale'] = module.pre_scale()
        self.extra_kwargs['pre_zero_point'] = module.pre_zero_point()

    def symbolic_execution(self, x: Tensor):
        out, scale, zero_point, bit_width = super().symbolic_execution(x)
        pre_scale = self.extra_kwargs['pre_scale']
        pre_zero_point = self.extra_kwargs['pre_zero_point']
        return out, pre_scale, pre_zero_point, scale, zero_point, bit_width


class BrevitasActQuantProxyHandler(BrevitasQuantProxyHandler):
    handled_layer = ActQuantProxyFromInjector


class BrevitasBiasQuantProxyHandler(BrevitasQuantProxyHandler):
    handled_layer = BiasQuantProxyFromInjector

    def symbolic_execution(self, x: Tensor, input_scale=None, input_bit_width=None):
        # avoid in place pop in case the proxy is shared
        symbolic_kwargs = copy(self.symbolic_kwargs)
        scale = symbolic_kwargs.pop('scale')
        bit_width = symbolic_kwargs.pop('bit_width')
        zero_point = symbolic_kwargs.pop('zero_point')
        if scale is None:
            assert input_scale is not None, 'Input scale required for bias export'
            scale = input_scale
        if bit_width is None:
            assert input_bit_width is not None, 'Input bit_width required for bias export'
            bit_width = input_bit_width
        y = BrevitasQuantFn.apply(
            x, scale, zero_point, bit_width, *symbolic_kwargs.values())
        return y, scale, zero_point, bit_width


class BrevitasTruncQuantProxyHandler(ONNXBaseHandler):
    handled_layer = TruncQuantProxyFromInjector

    def prepare_for_export(self, module: TruncQuantProxyFromInjector):
        self.symbolic_kwargs = {
                'output_bit_width': module.bit_width(),
                'rounding_mode': module.rounding_mode}

    def symbolic_execution(
            self, x: Tensor, scale: Tensor, zero_point: Tensor, input_bit_width: Tensor, signed: Tensor):
        y = BrevitasTruncFn.apply(
            x, scale, zero_point, input_bit_width, *self.symbolic_kwargs.values())
        return y, scale, zero_point, self.symbolic_kwargs['output_bit_width']


class BrevitasQuantLSTMLayerHandler(ONNXBaseHandler):
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
            return BrevitasQuantLSTMCellFn.apply(
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
                quant_bias_output, 
                *self.symbolic_kwargs.values())
        else:
            from torch.onnx.symbolic_helper import _export_onnx_opset_version
            if _export_onnx_opset_version < 14:
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
            sequence_lens = torch.empty(quant_hidden_state.size(0), dtype=torch.int32).fill_(seq_len)
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
