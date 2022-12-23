from pytest_cases import parametrize, set_case_id
from torch import nn

from brevitas.quant.scaled_int import Int32Bias
from .common import *


class QuantWBIOLCases:

    @parametrize('impl', QUANT_WBIOL_IMPL, ids=[f'{c.__name__}' for c in QUANT_WBIOL_IMPL])
    @parametrize('input_bit_width', BIT_WIDTHS, ids=[f'i{b}' for b in BIT_WIDTHS])
    @parametrize('weight_bit_width', BIT_WIDTHS, ids=[f'w{b}' for b in BIT_WIDTHS])
    @parametrize('output_bit_width', BIT_WIDTHS, ids=[f'o{b}' for b in BIT_WIDTHS])
    @parametrize('per_channel', [True, False])
    @parametrize('quantizers', QUANTIZERS.values(), ids=list(QUANTIZERS.keys()))
    def case_quant_wbiol(
            self, impl, input_bit_width, weight_bit_width, output_bit_width, per_channel, quantizers, request):
        
        # Change the case_id based on current value of Parameters
        set_case_id(request.node.callspec.id, QuantWBIOLCases.case_quant_wbiol) 
        
        weight_quant, io_quant = quantizers
        if impl is QuantLinear:
            layer_kwargs = {
                'in_features': IN_CH,
                'out_features': OUT_CH}
        else:
            layer_kwargs = {
                'in_channels': IN_CH,
                'out_channels': OUT_CH,
                'kernel_size': KERNEL_SIZE}

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv = impl(
                    **layer_kwargs,
                    bias=True,
                    weight_quant=weight_quant,
                    input_quant=io_quant,
                    output_quant=io_quant,
                    weight_bit_width=weight_bit_width,
                    input_bit_width=input_bit_width,
                    output_bit_width=output_bit_width,
                    bias_quant=Int32Bias,
                    weight_scaling_per_output_channel=per_channel,
                    return_quant_tensor=True)
                self.conv.weight.data.uniform_(-0.01, 0.01)

            def forward(self, x):
                return self.conv(x)

        torch.random.manual_seed(SEED)
        module = Model()
        return module
    
    
class QuantRecurrentCases:

    @parametrize('bidirectional', [True, False, 'shared_input_hidden'])
    @parametrize('cifg', [True, False])
    @parametrize('num_layers', [1, 2])
    def case_float_lstm(self, bidirectional, cifg, num_layers, request):
        
        # Change the case_id based on current value of Parameters
        set_case_id(request.node.callspec.id, QuantRecurrentCases.case_float_lstm)

        if bidirectional == 'shared_input_hidden':
            bidirectional = True
            shared_input_hidden = True
        else:
            shared_input_hidden = False
        
        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.lstm = QuantLSTM(
                    input_size=IN_CH,
                    hidden_size=OUT_CH,
                    weight_quant=None,
                    bias_quant=None,
                    io_quant=None,
                    gate_acc_quant=None,
                    sigmoid_quant=None,
                    tanh_quant=None,
                    cell_state_quant=None,
                    batch_first=False, # ort doesn't support batch_first=True (layout = 1)
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    shared_input_hidden_weights=shared_input_hidden,
                    coupled_input_forget_gates=cifg)

            def forward(self, x):
                return self.lstm(x)

        torch.random.manual_seed(SEED)
        module = Model()
        return module
    
    @parametrize('bidirectional', [True, False, 'shared_input_hidden'])
    @parametrize('cifg', [True, False])
    @parametrize('num_layers', [1, 2])
    @parametrize('weight_bit_width', BIT_WIDTHS, ids=[f'w{b}' for b in BIT_WIDTHS])
    @parametrize('per_channel', [True, False])
    @parametrize('quantizers', QUANTIZERS.values(), ids=list(QUANTIZERS.keys()))
    def case_quant_lstm(self, bidirectional, cifg, num_layers, weight_bit_width, per_channel, quantizers, request):
        
        # Change the case_id based on current value of Parameters
        set_case_id(request.node.callspec.id, QuantRecurrentCases.case_quant_lstm)
        
        weight_quant, _ = quantizers
        if bidirectional == 'shared_input_hidden':
            bidirectional = True
            shared_input_hidden = True
        else:
            shared_input_hidden = False
        
        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.lstm = QuantLSTM(
                    input_size=IN_CH,
                    hidden_size=OUT_CH,
                    weight_quant=weight_quant,
                    weight_bit_width=weight_bit_width,
                    weight_scaling_per_output_channel=per_channel,
                    bias_quant=None,
                    io_quant=None,
                    gate_acc_quant=None,
                    sigmoid_quant=None,
                    tanh_quant=None,
                    cell_state_quant=None,
                    batch_first=False, # ort doesn't support batch_first=True (layout = 1)
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    shared_input_hidden_weights=shared_input_hidden,
                    coupled_input_forget_gates=cifg)

            def forward(self, x):
                return self.lstm(x)

        torch.random.manual_seed(SEED)
        module = Model()
        return module


