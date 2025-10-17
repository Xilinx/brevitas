# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from pytest_cases import parametrize
from pytest_cases import set_case_id
from torch import nn

from brevitas.nn.quant_activation import QuantIdentity
from brevitas.nn.quant_avg_pool import TruncAvgPool2d
from brevitas.nn.quant_rnn import QuantLSTM
from brevitas.quant.scaled_int import Int32Bias

from .common import *


class QuantWBIOLCases:

    @parametrize(
        'rounding_type', ['round', 'floor'], ids=[f'rtype_{r}' for r in ['round', 'floor']])
    @parametrize('impl', QUANT_WBIOL_IMPL, ids=[f'{c.__name__}' for c in QUANT_WBIOL_IMPL])
    @parametrize('input_bit_width', BIT_WIDTHS, ids=[f'i{b}' for b in BIT_WIDTHS])
    @parametrize('weight_bit_width', BIT_WIDTHS, ids=[f'w{b}' for b in BIT_WIDTHS])
    @parametrize('output_bit_width', BIT_WIDTHS, ids=[f'o{b}' for b in BIT_WIDTHS])
    @parametrize('quantizers', WBIOL_QUANTIZERS.values(), ids=list(WBIOL_QUANTIZERS.keys()))
    def case_quant_wbiol(
            self,
            rounding_type,
            impl,
            input_bit_width,
            weight_bit_width,
            output_bit_width,
            quantizers,
            request):

        # Change the case_id based on current value of Parameters
        set_case_id(request.node.callspec.id, QuantWBIOLCases.case_quant_wbiol)

        weight_quant, io_quant = quantizers
        is_fp8 = weight_quant == Fp8e4m3OCPWeightPerTensorFloat
        is_dynamic = io_quant == ShiftedUint8DynamicActPerTensorFloat
        if is_fp8 or rounding_type == 'floor':
            if weight_bit_width < 8 or input_bit_width < 8 or output_bit_width < 8:
                pytest.skip('FP8 export and FLOOR rounding require all bitwidths equal to 8')
            torch.use_deterministic_algorithms(False)
        else:
            torch.use_deterministic_algorithms(True)

        if impl is QuantLinear:
            layer_kwargs = {'in_features': IN_CH, 'out_features': OUT_CH}
        else:
            layer_kwargs = {
                'in_channels': IN_CH, 'out_channels': OUT_CH, 'kernel_size': KERNEL_SIZE}

        bias_quantizer = None if (is_fp8 or is_dynamic) else Int32Bias
        # Required because of numpy error with FP8 data type. Export iself works fine.
        return_quant_tensor = False if is_fp8 else True

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
                    bias_quant=bias_quantizer,
                    weight_float_to_int_impl_type=rounding_type,
                    return_quant_tensor=return_quant_tensor)
                self.conv.weight.data.uniform_(-0.01, 0.01)

            def forward(self, x):
                return self.conv(x)

        torch.random.manual_seed(SEED)
        module = Model()
        return module


class QuantAvgPoolCases:

    @parametrize('output_bit_width', BIT_WIDTHS, ids=[f'o{b}' for b in BIT_WIDTHS])
    @parametrize('input_signed', [True, False])
    def case_quant_avgpool(self, input_signed, output_bit_width):

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.in_quant = QuantIdentity(signed=input_signed, return_quant_tensor=True)
                self.quant_avg_pool = TruncAvgPool2d(
                    kernel_size=3,
                    stride=2,
                    bit_width=output_bit_width,
                    float_to_int_impl_type='round')

            def forward(self, x):
                return self.quant_avg_pool(self.in_quant(x))

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
                    batch_first=False,  # ort doesn't support batch_first=True (layout = 1)
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
    @parametrize('quantizers', LSTM_QUANTIZERS.values(), ids=list(LSTM_QUANTIZERS.keys()))
    def case_quant_lstm(
            self, bidirectional, cifg, num_layers, weight_bit_width, quantizers, request):

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
                    bias_quant=None,
                    io_quant=None,
                    gate_acc_quant=None,
                    sigmoid_quant=None,
                    tanh_quant=None,
                    cell_state_quant=None,
                    batch_first=False,  # ort doesn't support batch_first=True (layout = 1)
                    num_layers=num_layers,
                    bidirectional=bidirectional,
                    shared_input_hidden_weights=shared_input_hidden,
                    coupled_input_forget_gates=cifg)

            def forward(self, x):
                return self.lstm(x)

        torch.random.manual_seed(SEED)
        module = Model()
        return module
