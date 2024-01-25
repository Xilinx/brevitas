# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging import version
import pytest
import pytest_cases
from pytest_cases import get_case_id
from pytest_cases import set_case_id
import torch
import torch.nn as nn

from brevitas import torch_version
from brevitas.nn import QuantConv1d
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantConvTranspose1d
from brevitas.nn import QuantConvTranspose2d
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
from brevitas.nn.quant_mha import QuantMultiheadAttention
from brevitas.nn.quant_rnn import QuantLSTM
from brevitas.nn.quant_rnn import QuantRNN
from brevitas.quant.scaled_int import Int8AccumulatorAwareWeightQuant
from brevitas.quant.scaled_int import Int8AccumulatorAwareZeroCenterWeightQuant
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloatBatchQuant1d
from brevitas.quant.scaled_int import Int8ActPerTensorFloatBatchQuant2d
from brevitas.quant.scaled_int import Int8BiasPerTensorFloatInternalScaling
from brevitas.quant.scaled_int import Int8WeightNormL2PerChannelFixedPoint
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int16Bias
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas.quant_tensor import QuantTensor

SEED = 123456
OUT_CH = 16
IN_CH = 8
FEATURES = 5
KERNEL_SIZE = 3
EMBED_DIM = 9
NUM_HEADS = 3

LSTM_WEIGHT_QUANTIZER = {
    'None': None,
    'quant_sym': Int8WeightPerTensorFloat,
    'quant_asym': ShiftedUint8WeightPerTensorFloat}

A2Q_WBIOL_WEIGHT_QUANTIZER = {
    'quant_a2q': Int8AccumulatorAwareWeightQuant,
    'quant_a2q_plus': Int8AccumulatorAwareZeroCenterWeightQuant}

WBIOL_WEIGHT_QUANTIZER = {
    'None': None,
    'quant_sym': Int8WeightPerTensorFloat,
    'quant_asym': ShiftedUint8WeightPerTensorFloat,
    'quant_decoupled': Int8WeightNormL2PerChannelFixedPoint,
    **A2Q_WBIOL_WEIGHT_QUANTIZER}

WBIOL_IO_QUANTIZER = {
    'None': None,
    'batch_quant': (Int8ActPerTensorFloatBatchQuant1d, Int8ActPerTensorFloatBatchQuant2d),
    'quant_sym': Int8ActPerTensorFloat,
    'quant_asym': ShiftedUint8ActPerTensorFloat}

LSTM_IO_QUANTIZER = {
    'None': None,
    'quant_sym': Int8ActPerTensorFloat,}

MHA_IO_QUANTIZER = {
    'None': None,
    'batch_quant': (Int8ActPerTensorFloatBatchQuant1d, Int8ActPerTensorFloat),
    'quant_sym': Int8ActPerTensorFloat,
    'quant_asym': ShiftedUint8ActPerTensorFloat}

SIGNED_ACT_QUANTIZER = {
    'None': None,
    'quant_sym': Int8ActPerTensorFloat,
    'quant_asym': ShiftedUint8ActPerTensorFloat,}

UNSIGNED_ACT_QUANTIZER = {
    'None': None,
    'quant_sym': Uint8ActPerTensorFloat,}

BIAS_QUANTIZER = {
    'None': None,
    'quant_external': Int16Bias,
    'quant_internal': Int8BiasPerTensorFloatInternalScaling,}

QUANT_WBIOL_IMPL = [
    QuantLinear,
    QuantConv1d,
    QuantConv2d,
    QuantConvTranspose1d,
    QuantConvTranspose2d,]

ACC_BIT_WIDTHS = [8, 9, 10, 12, 16, 24, 32]


def build_case_model(
        weight_quantizer,
        bias_quantizer,
        io_quantizer,
        return_quant_tensor,
        module,
        case_id,
        input_quantized,
        is_training,
        accumulator_bit_width=32):

    k, weight_quantizer = weight_quantizer
    _, bias_quantizer = bias_quantizer
    _, io_quantizer = io_quantizer

    if io_quantizer is None and not input_quantized and k in A2Q_WBIOL_WEIGHT_QUANTIZER:
        pytest.skip(
            "A2Q uses an input-aware decoupled weight proxy that requires a quantized input tensor."
        )

    impl = module.__name__
    # BatchQuant has dimension specific quantizers
    if isinstance(io_quantizer, tuple):
        if '1d' in impl:
            io_quantizer = io_quantizer[0]  # select 1d quantizer
        elif '2d' in impl:
            io_quantizer = io_quantizer[1]  # select 2d quantizer
        else:
            pytest.skip("Combination of layer and quantizer not supported.")
    if impl == 'QuantLinear':
        layer_kwargs = {'in_features': IN_CH, 'out_features': OUT_CH}
    else:
        layer_kwargs = {'in_channels': IN_CH, 'out_channels': OUT_CH, 'kernel_size': KERNEL_SIZE}

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = module(
                **layer_kwargs,
                bias=True,
                weight_quant=weight_quantizer,
                input_quant=io_quantizer,
                output_quant=io_quantizer,
                bias_quant=bias_quantizer,
                return_quant_tensor=return_quant_tensor,
                weight_accumulator_bit_width=accumulator_bit_width)
            self.conv.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.conv(x)

    torch.random.manual_seed(SEED)
    module = Model()
    module.train(is_training)

    if impl in ('QuantLinear',):
        in_size = (1, IN_CH)
    elif impl in ('QuantConv1d', 'QuantConvTranspose1d'):
        in_size = (1, IN_CH, FEATURES)
    else:
        in_size = (1, IN_CH, FEATURES, FEATURES)

    if input_quantized:
        quant_inp = QuantTensor(
            torch.randint(-128, 127, in_size) * 0.128, 0.128, 0., 8., True, is_training)
    else:
        quant_inp = torch.randn(in_size)
    return module, quant_inp


@pytest_cases.parametrize(
    'input_quantized', [True, False], ids=[f'input_quantized${c}' for c in [True, False]])
@pytest_cases.parametrize(
    'bias_quantizer',
    BIAS_QUANTIZER.items(),
    ids=[f'bias_quant${c}' for c, _ in BIAS_QUANTIZER.items()])
@pytest_cases.parametrize(
    'io_quantizer',
    WBIOL_IO_QUANTIZER.items(),
    ids=[f'io_quant${c}' for c, _ in WBIOL_IO_QUANTIZER.items()])
@pytest_cases.parametrize(
    'weight_quantizer',
    WBIOL_WEIGHT_QUANTIZER.items(),
    ids=[f'weight_quant${c}' for c, _ in WBIOL_WEIGHT_QUANTIZER.items()])
@pytest_cases.parametrize(
    'return_quant_tensor', [True, False], ids=[f'return_quant_tensor${f}' for f in [True, False]])
@pytest_cases.parametrize(
    'module', QUANT_WBIOL_IMPL, ids=[f'model_type${c.__name__}' for c in QUANT_WBIOL_IMPL])
@pytest_cases.parametrize(
    'is_training', [True, False], ids=[f'is_training${f}' for f in [True, False]])
def case_model(
        weight_quantizer,
        bias_quantizer,
        io_quantizer,
        return_quant_tensor,
        module,
        request,
        input_quantized,
        is_training):
    set_case_id(request.node.callspec.id, case_model)
    case_id = get_case_id(case_model)
    return build_case_model(
        weight_quantizer,
        bias_quantizer,
        io_quantizer,
        return_quant_tensor,
        module,
        case_id,
        input_quantized,
        is_training)


@pytest_cases.parametrize(
    'io_quantizer',
    WBIOL_IO_QUANTIZER.items(),
    ids=[f'io_quant${c}' for c, _ in WBIOL_IO_QUANTIZER.items()])
@pytest_cases.parametrize(
    'module', QUANT_WBIOL_IMPL, ids=[f'model_type${c.__name__}' for c in QUANT_WBIOL_IMPL])
@pytest_cases.parametrize(
    'accumulator_bit_width',
    ACC_BIT_WIDTHS,
    ids=[f'accumulator_bit_width${bw}' for bw in ACC_BIT_WIDTHS])
@pytest_cases.parametrize(
    'weight_quantizer',
    A2Q_WBIOL_WEIGHT_QUANTIZER.items(),
    ids=[f'weight_quant${c}' for c, _ in A2Q_WBIOL_WEIGHT_QUANTIZER.items()])
def case_model_a2q(io_quantizer, module, request, accumulator_bit_width, weight_quantizer):
    set_case_id(request.node.callspec.id, case_model_a2q)
    case_id = get_case_id(case_model_a2q)
    # reducing coverage by fixing some case parameters
    return build_case_model(
        weight_quantizer,
        ("None", None),  # force bias_quantizer = None (irrelevant)
        io_quantizer,
        True,  # force return_quant_tensor = True (irrelevant)
        module,
        case_id,
        True,  # force input_quantized = True (required)
        True,  # force is_training = True (irrelevant)
        accumulator_bit_width=accumulator_bit_width)


@pytest_cases.parametrize(
    'io_quantizer',
    LSTM_IO_QUANTIZER.items(),
    ids=[f'io_quant${c}' for c, _ in LSTM_IO_QUANTIZER.items()])
@pytest_cases.parametrize(
    'input_quantized', [True, False], ids=[f'input_quantized${c}' for c in [True, False]])
@pytest_cases.parametrize(
    'bias_quantizer',
    BIAS_QUANTIZER.items(),
    ids=[f'bias_quant${c}' for c, _ in BIAS_QUANTIZER.items()])
@pytest_cases.parametrize(
    'weight_quantizer',
    LSTM_WEIGHT_QUANTIZER.items(),
    ids=[f'weight_quant${c}' for c, _ in LSTM_WEIGHT_QUANTIZER.items()])
@pytest_cases.parametrize(
    'return_quant_tensor', [True, False], ids=[f'return_quant_tensor${f}' for f in [True, False]])
@pytest_cases.parametrize(
    'bidirectional', [True, False, 'shared_input_hidden'],
    ids=[f'bidirectional${f}' for f in [True, False, 'shared_input_hidden']])
@pytest_cases.parametrize('cifg', [True, False])
@pytest_cases.parametrize('num_layers', [1, 2], ids=[f'num_layers${f}' for f in [1, 2]])
def case_quant_lstm(
        weight_quantizer,
        bias_quantizer,
        return_quant_tensor,
        input_quantized,
        request,
        bidirectional,
        cifg,
        num_layers,
        io_quantizer):

    # Change the case_id based on current value of Parameters
    set_case_id(request.node.callspec.id, case_quant_lstm)
    _, weight_quantizer = weight_quantizer
    _, bias_quantizer = bias_quantizer
    _, io_quantizer = io_quantizer

    if bidirectional == 'shared_input_hidden':
        bidirectional = True
        shared_input_hidden = True
    else:
        shared_input_hidden = False

    if return_quant_tensor and io_quantizer is None:
        pytest.skip("return_quant_tensor cannot be True if no io_quantizer is specified")

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.lstm = QuantLSTM(
                input_size=IN_CH,
                hidden_size=OUT_CH,
                weight_quant=weight_quantizer,
                bias_quant=bias_quantizer,
                io_quant=io_quantizer,
                gate_acc_quant=io_quantizer,
                sigmoid_quant=io_quantizer,
                tanh_quant=io_quantizer,
                cell_state_quant=io_quantizer,
                batch_first=False,  # ort doesn't support batch_first=True (layout = 1)
                num_layers=num_layers,
                bidirectional=bidirectional,
                shared_input_hidden_weights=shared_input_hidden,
                return_quant_tensor=return_quant_tensor,
                coupled_input_forget_gates=cifg)

        def forward(self, x):
            return self.lstm(x)

    torch.random.manual_seed(SEED)
    module = Model()

    in_size = (FEATURES, 1, IN_CH)
    inp = torch.randn(in_size)

    if input_quantized:
        act = QuantIdentity(return_quant_tensor=True)
        quant_inp = act(inp)
    else:
        quant_inp = inp

    return module, quant_inp


@pytest_cases.parametrize(
    'io_quantizer',
    LSTM_IO_QUANTIZER.items(),
    ids=[f'io_quant${c}' for c, _ in LSTM_IO_QUANTIZER.items()])
@pytest_cases.parametrize(
    'input_quantized', [True, False], ids=[f'input_quantized${c}' for c in [True, False]])
@pytest_cases.parametrize(
    'bias_quantizer',
    BIAS_QUANTIZER.items(),
    ids=[f'bias_quant${c}' for c, _ in BIAS_QUANTIZER.items()])
@pytest_cases.parametrize(
    'signed_act_quantizer',
    SIGNED_ACT_QUANTIZER.items(),
    ids=[f'signed_act${c}' for c, _ in SIGNED_ACT_QUANTIZER.items()])
@pytest_cases.parametrize(
    'unsigned_act_quantizer',
    UNSIGNED_ACT_QUANTIZER.items(),
    ids=[f'unsigned_act${c}' for c, _ in UNSIGNED_ACT_QUANTIZER.items()])
@pytest_cases.parametrize(
    'weight_quantizer',
    LSTM_WEIGHT_QUANTIZER.items(),
    ids=[f'weight_quant${c}' for c, _ in LSTM_WEIGHT_QUANTIZER.items()])
@pytest_cases.parametrize(
    'return_quant_tensor', [True, False], ids=[f'return_quant_tensor${f}' for f in [True, False]])
@pytest_cases.parametrize(
    'bidirectional', [True, False, 'shared_input_hidden'],
    ids=[f'bidirectional${f}' for f in [True, False, 'shared_input_hidden']])
@pytest_cases.parametrize('cifg', [True, False])
@pytest_cases.parametrize('num_layers', [1, 2], ids=[f'num_layers${f}' for f in [1, 2]])
def case_quant_lstm_full(
        weight_quantizer,
        bias_quantizer,
        return_quant_tensor,
        input_quantized,
        request,
        bidirectional,
        cifg,
        num_layers,
        io_quantizer,
        unsigned_act_quantizer,
        signed_act_quantizer):

    # Change the case_id based on current value of Parameters
    set_case_id(request.node.callspec.id, case_quant_lstm_full)
    _, weight_quantizer = weight_quantizer
    _, bias_quantizer = bias_quantizer
    _, io_quantizer = io_quantizer
    _, signed_act_quantizer = signed_act_quantizer
    _, unsigned_act_quantizer = unsigned_act_quantizer

    if bidirectional == 'shared_input_hidden':
        bidirectional = True
        shared_input_hidden = True
    else:
        shared_input_hidden = False

    if return_quant_tensor and io_quantizer is None:
        pytest.skip("return_quant_tensor cannot be True if no io_quantizer is specified")

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.lstm = QuantLSTM(
                input_size=IN_CH,
                hidden_size=OUT_CH,
                weight_quant=weight_quantizer,
                bias_quant=bias_quantizer,
                io_quant=io_quantizer,
                gate_acc_quant=signed_act_quantizer,
                sigmoid_quant=unsigned_act_quantizer,
                tanh_quant=signed_act_quantizer,
                cell_state_quant=signed_act_quantizer,
                batch_first=False,  # ort doesn't support batch_first=True (layout = 1)
                num_layers=num_layers,
                bidirectional=bidirectional,
                shared_input_hidden_weights=shared_input_hidden,
                return_quant_tensor=return_quant_tensor,
                coupled_input_forget_gates=cifg)

        def forward(self, x):
            return self.lstm(x)

    torch.random.manual_seed(SEED)
    module = Model()

    in_size = (FEATURES, 1, IN_CH)
    inp = torch.randn(in_size)

    if input_quantized:
        act = QuantIdentity(return_quant_tensor=True)
        quant_inp = act(inp)
    else:
        quant_inp = inp

    return module, quant_inp


@pytest_cases.parametrize(
    'io_quantizer',
    LSTM_IO_QUANTIZER.items(),
    ids=[f'io_quant${c}' for c, _ in LSTM_IO_QUANTIZER.items()])
@pytest_cases.parametrize(
    'input_quantized', [True, False], ids=[f'input_quantized${c}' for c in [True, False]])
@pytest_cases.parametrize(
    'bias_quantizer',
    BIAS_QUANTIZER.items(),
    ids=[f'bias_quant${c}' for c, _ in BIAS_QUANTIZER.items()])
@pytest_cases.parametrize(
    'weight_quantizer',
    LSTM_WEIGHT_QUANTIZER.items(),
    ids=[f'weight_quant${c}' for c, _ in LSTM_WEIGHT_QUANTIZER.items()])
@pytest_cases.parametrize(
    'return_quant_tensor', [True, False], ids=[f'return_quant_tensor${f}' for f in [True, False]])
@pytest_cases.parametrize(
    'bidirectional', [True, False], ids=[f'bidirectional${f}' for f in [True, False]])
@pytest_cases.parametrize('num_layers', [1, 2], ids=[f'num_layers${f}' for f in [1, 2]])
def case_quant_rnn(
        weight_quantizer,
        bias_quantizer,
        return_quant_tensor,
        input_quantized,
        request,
        bidirectional,
        num_layers,
        io_quantizer):

    # Change the case_id based on current value of Parameters
    set_case_id(request.node.callspec.id, case_quant_rnn)
    _, weight_quantizer = weight_quantizer
    _, bias_quantizer = bias_quantizer
    _, io_quantizer = io_quantizer

    if return_quant_tensor and io_quantizer is None:
        pytest.skip("return_quant_tensor cannot be True if no io_quantizer is specified")

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.lstm = QuantRNN(
                input_size=IN_CH,
                hidden_size=OUT_CH,
                weight_quant=weight_quantizer,
                bias_quant=bias_quantizer,
                io_quant=io_quantizer,
                gate_acc_quant=io_quantizer,
                batch_first=False,
                bidirectional=bidirectional,
                num_layers=num_layers,
                return_quant_tensor=return_quant_tensor)

        def forward(self, x):
            return self.lstm(x)

    torch.random.manual_seed(SEED)
    module = Model()

    in_size = (FEATURES, 1, IN_CH)
    inp = torch.randn(in_size)

    if input_quantized:
        act = QuantIdentity(return_quant_tensor=True)
        quant_inp = act(inp)
    else:
        quant_inp = inp

    return module, quant_inp


@pytest_cases.parametrize(
    'io_quantizer',
    LSTM_IO_QUANTIZER.items(),
    ids=[f'io_quant${c}' for c, _ in LSTM_IO_QUANTIZER.items()])
@pytest_cases.parametrize(
    'input_quantized', [True, False], ids=[f'input_quantized${c}' for c in [True, False]])
@pytest_cases.parametrize(
    'bias_quantizer',
    BIAS_QUANTIZER.items(),
    ids=[f'bias_quant${c}' for c, _ in BIAS_QUANTIZER.items()])
@pytest_cases.parametrize(
    'signed_act_quantizer',
    SIGNED_ACT_QUANTIZER.items(),
    ids=[f'signed_act${c}' for c, _ in SIGNED_ACT_QUANTIZER.items()])
@pytest_cases.parametrize(
    'weight_quantizer',
    LSTM_WEIGHT_QUANTIZER.items(),
    ids=[f'weight_quant${c}' for c, _ in LSTM_WEIGHT_QUANTIZER.items()])
@pytest_cases.parametrize(
    'return_quant_tensor', [True, False], ids=[f'return_quant_tensor${f}' for f in [True, False]])
@pytest_cases.parametrize(
    'bidirectional', [True, False], ids=[f'bidirectional${f}' for f in [True, False]])
@pytest_cases.parametrize('num_layers', [1, 2], ids=[f'num_layers${f}' for f in [1, 2]])
def case_quant_rnn_full(
        weight_quantizer,
        bias_quantizer,
        return_quant_tensor,
        input_quantized,
        request,
        bidirectional,
        num_layers,
        io_quantizer,
        signed_act_quantizer):

    # Change the case_id based on current value of Parameters
    set_case_id(request.node.callspec.id, case_quant_rnn_full)
    _, weight_quantizer = weight_quantizer
    _, bias_quantizer = bias_quantizer
    _, io_quantizer = io_quantizer
    _, signed_act_quantizer = signed_act_quantizer

    if return_quant_tensor and io_quantizer is None:
        pytest.skip("return_quant_tensor cannot be True if no io_quantizer is specified")

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.lstm = QuantRNN(
                input_size=IN_CH,
                hidden_size=OUT_CH,
                weight_quant=weight_quantizer,
                bias_quant=bias_quantizer,
                io_quant=io_quantizer,
                gate_acc_quant=signed_act_quantizer,
                batch_first=False,
                bidirectional=bidirectional,
                num_layers=num_layers,
                return_quant_tensor=return_quant_tensor)

        def forward(self, x):
            return self.lstm(x)

    torch.random.manual_seed(SEED)
    module = Model()

    in_size = (FEATURES, 1, IN_CH)
    inp = torch.randn(in_size)

    if input_quantized:
        act = QuantIdentity(return_quant_tensor=True)
        quant_inp = act(inp)
    else:
        quant_inp = inp

    return module, quant_inp


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("packed_in_proj", [True, False])
@pytest_cases.parametrize(
    'io_quantizer',
    MHA_IO_QUANTIZER.items(),
    ids=[f'io_quant${c}' for c, _ in MHA_IO_QUANTIZER.items()])
@pytest_cases.parametrize(
    'input_quantized', [True, False], ids=[f'input_quantized${c}' for c in [True, False]])
@pytest_cases.parametrize(
    'bias_quantizer',
    BIAS_QUANTIZER.items(),
    ids=[f'bias_quant${c}' for c, _ in BIAS_QUANTIZER.items()])
@pytest_cases.parametrize(
    'weight_quantizer',
    WBIOL_WEIGHT_QUANTIZER.items(),
    ids=[f'weight_quant${c}' for c, _ in WBIOL_WEIGHT_QUANTIZER.items()])
@pytest_cases.parametrize(
    'return_quant_tensor', [True, False], ids=[f'return_quant_tensor${f}' for f in [True, False]])
def case_mha(
        batch_first,
        packed_in_proj,
        weight_quantizer,
        bias_quantizer,
        return_quant_tensor,
        input_quantized,
        request,
        io_quantizer):
    extra_kwargs = {}
    if torch_version >= version.parse('1.9.1'):
        extra_kwargs['batch_first'] = batch_first

    # Change the case_id based on current value of Parameters
    set_case_id(request.node.callspec.id, case_mha)
    k, weight_quantizer = weight_quantizer
    _, bias_quantizer = bias_quantizer
    _, io_quantizer = io_quantizer

    if io_quantizer is None and k in A2Q_WBIOL_WEIGHT_QUANTIZER:
        # Can't rely on a QuantTensor input for quant_mha at this point
        pytest.skip(
            "A2Q uses an input-aware decoupled weight proxy that requires a quantized input tensor."
        )

    # BatchQuant1d works over 3d input but not 2d, so we have a separate quantizer for out_proj
    if isinstance(io_quantizer, tuple):
        io_quantizer, out_proj_io_quantizer = io_quantizer
        if not batch_first:
            pytest.skip("BatchQuant requires batch_first=True.")
    else:
        out_proj_io_quantizer = io_quantizer

    module = QuantMultiheadAttention(
        EMBED_DIM,
        NUM_HEADS,
        packed_in_proj=packed_in_proj,
        in_proj_input_quant=io_quantizer,
        in_proj_weight_quant=weight_quantizer,
        in_proj_bias_quant=bias_quantizer,
        softmax_input_quant=io_quantizer,
        attn_output_weights_quant=io_quantizer,
        q_scaled_quant=io_quantizer,
        k_transposed_quant=io_quantizer,
        v_quant=io_quantizer,
        out_proj_input_quant=out_proj_io_quantizer,
        out_proj_weight_quant=weight_quantizer,
        out_proj_bias_quant=bias_quantizer,
        out_proj_output_quant=out_proj_io_quantizer,
        bias=True,
        return_quant_tensor=return_quant_tensor,
        **extra_kwargs)

    torch.random.manual_seed(SEED)

    in_size = (2, 5, EMBED_DIM)
    inp = torch.randn(in_size)

    if input_quantized:
        act = QuantIdentity(return_quant_tensor=True)
        quant_inp = act(inp)
    else:
        quant_inp = inp

    return module, quant_inp
