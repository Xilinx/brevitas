# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Optional

from hypothesis import strategies as st
from pytest_cases import parametrize
from pytest_cases import set_case_id
from torch import nn

from brevitas.nn.quant_activation import QuantIdentity
from brevitas.nn.quant_avg_pool import TruncAvgPool2d
from brevitas.nn.quant_rnn import QuantLSTM
from brevitas.quant.scaled_int import Int32Bias

from .common import *

# Bit-width examples Hypothesis draws per enumerated flag combination (each a full export + ORT run).
WBIOL_BITWIDTH_EXAMPLES = 10


@dataclass(frozen=True)
class WBIOLFlags:
    """A valid WBIOL configuration minus the bit-widths (the enumerated axes)."""
    quantizer_name: str
    weight_quant: type
    io_quant: type
    impl: type
    rounding_type: str
    export_type: str
    bias_quant: Optional[type]
    export_q_weight: bool

    @property
    def is_fp8(self):
        return self.weight_quant == Fp8e4m3OCPWeightPerTensorFloat

    @property
    def is_dynamic(self):
        return self.io_quant == ShiftedUint8DynamicActPerTensorFloat

    @property
    def id(self):
        bias = self.bias_quant.__name__ if self.bias_quant is not None else 'none'
        return (
            f'{self.quantizer_name}-{self.impl.__name__}-rtype_{self.rounding_type}'
            f'-{self.export_type}-bias_{bias}-qw_{int(self.export_q_weight)}')


def enumerate_wbiol_flags():
    """Enumerate every valid WBIOL flag combination (bit-widths are sampled per node).

    Validity mirrors the exporter's limits: QuantLinear+asymmetric is excluded; dynamo exports
    need torch>=2.8 and dynamic act quant is QCDQ-only; fp8/dynamic force bias_quant=None;
    export_q_weight is forced True for fp8/qcdq_dynamo and False for a2q/floor.
    """
    combos = []
    names = list(WBIOL_QUANTIZERS)
    if torch_version < parse('2.1'):
        names = [n for n in names if 'fp8' not in n]  # fp8 requires PyTorch >= 2.1
    for quantizer_name in names:
        weight_quant, io_quant = WBIOL_QUANTIZERS[quantizer_name]
        is_fp8 = weight_quant == Fp8e4m3OCPWeightPerTensorFloat
        is_dynamic = io_quant == ShiftedUint8DynamicActPerTensorFloat
        for impl in QUANT_WBIOL_IMPL:
            if impl is QuantLinear and 'asymmetric' in quantizer_name:
                continue
            for rounding_type in ['round', 'floor']:
                exports = ['qcdq', 'qonnx']
                if torch_version >= parse('2.8'):
                    exports.append('qonnx_dynamo')
                    if rounding_type == 'round' and (is_fp8 or is_dynamic):
                        exports.append('qcdq_dynamo')
                if is_dynamic:
                    exports = [e for e in exports if e in ('qcdq', 'qcdq_dynamo')]
                for export_type in exports:
                    biases = [None] if (is_fp8 or is_dynamic) else [None, Int32Bias]
                    for bias_quant in biases:
                        if export_type not in ('qcdq', 'qcdq_dynamo'):
                            qws = [False]  # ignored on the qonnx export path
                        elif is_fp8 or export_type == 'qcdq_dynamo':
                            qws = [True]
                        elif rounding_type == 'floor' or 'a2q' in quantizer_name:
                            qws = [False]
                        else:
                            qws = [True, False]
                        for export_q_weight in qws:
                            combos.append(
                                WBIOLFlags(
                                    quantizer_name,
                                    weight_quant,
                                    io_quant,
                                    impl,
                                    rounding_type,
                                    export_type,
                                    bias_quant,
                                    export_q_weight))
    return combos


WBIOL_FLAG_COMBOS = enumerate_wbiol_flags()


@st.composite
def wbiol_bitwidths_st(draw, flags):
    """Sample (output, weight, input) bit-widths valid for these flags (fp8: all-8; dynamic: 8-bit i/o)."""
    if flags.is_fp8:
        return 8, 8, 8
    if flags.is_dynamic:
        return 8, draw(st.sampled_from(list(BIT_WIDTHS))), 8
    return tuple(draw(st.sampled_from(list(BIT_WIDTHS))) for _ in range(3))


def build_wbiol_model(flags, bit_widths):
    output_bit_width, weight_bit_width, input_bit_width = bit_widths
    weight_quant, io_quant = flags.weight_quant, flags.io_quant
    if flags.is_fp8 or flags.rounding_type == 'floor':
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)

    impl = flags.impl
    if impl is QuantLinear:
        layer_kwargs = {'in_features': IN_CH, 'out_features': OUT_CH}
    else:
        layer_kwargs = {'in_channels': IN_CH, 'out_channels': OUT_CH, 'kernel_size': KERNEL_SIZE}

    bias_quantizer = flags.bias_quant
    # Required because of numpy error with FP8 data type. Export iself works fine.
    return_quant_tensor = False if flags.is_fp8 else True

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
                weight_float_to_int_impl_type=flags.rounding_type,
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
