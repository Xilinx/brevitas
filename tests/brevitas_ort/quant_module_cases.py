# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from hypothesis import strategies as st
from pytest_cases import parametrize
from pytest_cases import set_case_id
from torch import nn

from brevitas.nn.quant_activation import QuantIdentity
from brevitas.nn.quant_avg_pool import TruncAvgPool2d
from brevitas.nn.quant_rnn import QuantLSTM
from brevitas.quant.scaled_int import Int32Bias

from .common import *

# Number of Hypothesis examples drawn for the (very large) WBIOL configuration space. Drawing
# per-axis keeps rare quantizers/export types well represented without any explicit handling.
# Each example is a full ONNX export + ORT inference, so this constant directly drives the
# runtime of test_ort_wbiol (a single, serially-run @given node).
WBIOL_MAX_EXAMPLES = 1000


@dataclass(frozen=True)
class WBIOLConfig:
    quantizer_name: str
    weight_quant: type
    io_quant: type
    output_bit_width: int
    weight_bit_width: int
    input_bit_width: int
    impl: type
    rounding_type: str
    export_type: str

    @property
    def id(self):
        return (
            f'wbiol-{self.quantizer_name}-o{self.output_bit_width}-w{self.weight_bit_width}'
            f'-i{self.input_bit_width}-{self.impl.__name__}-rtype_{self.rounding_type}'
            f'-{self.export_type}')


@st.composite
def wbiol_config_st(draw):
    """Draw a WBIOL configuration.

    floor bit-widths are drawn unconstrained on purpose (see the NOTE below), so a drawn config
    is not guaranteed exportable/valid: that constraint is being left to CI to confirm as
    genuinely required vs merely search-space-shrinking. The fp8 and dynamic-act 8-bit
    constraints have already been confirmed required by CI and are enforced here.
    """
    names = list(WBIOL_QUANTIZERS)
    if torch_version < parse('2.1'):
        names = [n for n in names if 'fp8' not in n]  # fp8 requires PyTorch >= 2.1
    quantizer_name = draw(st.sampled_from(names))
    weight_quant, io_quant = WBIOL_QUANTIZERS[quantizer_name]
    is_fp8 = weight_quant == Fp8e4m3OCPWeightPerTensorFloat
    is_dynamic = io_quant == ShiftedUint8DynamicActPerTensorFloat

    impls = QUANT_WBIOL_IMPL
    if 'asymmetric' in quantizer_name:
        # QuantLinear + asymmetric fails unreliably in ORT execution.
        impls = [i for i in impls if i is not QuantLinear]
    impl = draw(st.sampled_from(impls))

    rounding_type = draw(st.sampled_from(['round', 'floor']))

    # fp8 requires all-8 bit-widths: OCP e4m3 is a fixed 1+4+3 split, so overriding bit_width
    # breaks is_ocp_e4m3 (mantissa==3 and exponent==4, src/brevitas/proxy/float_parameter_quant.py)
    # and the exporter rejects it with 'Only OCP/FNUZ Standard are supported for FP8 export'
    # (src/brevitas/export/onnx/standard/qcdq/handler.py). Confirmed matrix-wide by CI run
    # 33414112132.
    #
    # Dynamic act quant requires 8-bit input/output: the QCDQ exporter validates it via
    # validate_8b_bit_width (src/brevitas/export/onnx/standard/qcdq/handler.py), which raised
    # 'Bit width 2 is not supported, should be 8b.' on every matrix cell of CI run 33403653736
    # once this was left unconstrained. Weight bit-width is not validated, so it stays free.
    #
    # NOTE: floor bit-widths are drawn unconstrained on purpose. floor was previously lumped into
    # the fp8 all-8 branch (commit 004479ef) with no independent justification; we are letting CI
    # determine whether it is genuinely required. Re-add if it proves necessary.
    if is_fp8:
        o = w = i = 8
    elif is_dynamic:
        o, i = 8, 8
        w = draw(st.sampled_from(list(BIT_WIDTHS)))
    else:
        o = draw(st.sampled_from(list(BIT_WIDTHS)))
        w = draw(st.sampled_from(list(BIT_WIDTHS)))
        i = draw(st.sampled_from(list(BIT_WIDTHS)))

    exports = ['qcdq', 'qonnx']
    if torch_version >= parse('2.8'):
        exports.append('qonnx_dynamo')
        # Dynamo QCDQ exports weights as a round-only Q-node and cannot export quantized bias,
        # so it is limited to round + fp8/dynamic quantizers (which don't quantize bias).
        if rounding_type == 'round' and (is_fp8 or is_dynamic):
            exports.append('qcdq_dynamo')
    if is_dynamic:  # dynamic act quant is only supported on the QCDQ export paths
        exports = [e for e in exports if e in ('qcdq', 'qcdq_dynamo')]
    export_type = draw(st.sampled_from(exports))

    return WBIOLConfig(
        quantizer_name, weight_quant, io_quant, o, w, i, impl, rounding_type, export_type)


def build_wbiol_model(config):
    weight_quant, io_quant = config.weight_quant, config.io_quant
    is_fp8 = weight_quant == Fp8e4m3OCPWeightPerTensorFloat
    is_dynamic = io_quant == ShiftedUint8DynamicActPerTensorFloat
    if is_fp8 or config.rounding_type == 'floor':
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)

    impl = config.impl
    if impl is QuantLinear:
        layer_kwargs = {'in_features': IN_CH, 'out_features': OUT_CH}
    else:
        layer_kwargs = {'in_channels': IN_CH, 'out_channels': OUT_CH, 'kernel_size': KERNEL_SIZE}

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
                weight_bit_width=config.weight_bit_width,
                input_bit_width=config.input_bit_width,
                output_bit_width=config.output_bit_width,
                bias_quant=bias_quantizer,
                weight_float_to_int_impl_type=config.rounding_type,
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
