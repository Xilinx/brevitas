# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import pytest_cases
from pytest_cases import get_case_id
import torch

from brevitas.quant_tensor import QuantTensor

from .nn_quantizers_fixture import case_mha
from .nn_quantizers_fixture import case_model
from .nn_quantizers_fixture import case_quant_lstm
from .nn_quantizers_fixture import case_quant_lstm_full
from .nn_quantizers_fixture import case_quant_rnn
from .nn_quantizers_fixture import case_quant_rnn_full


def parse_args(args):
    kwargs = {}
    for arg in args:
        if '$' not in arg:
            continue
        k, v = arg.split('$')
        try:
            v = eval(v)
        except:
            pass
        kwargs[k] = v
    return kwargs


@pytest_cases.parametrize_with_cases('model_input', cases=case_model)
def test_quant_wbiol(model_input, current_cases):
    model, input = model_input

    cases_generator_func = current_cases['model_input'][1]
    case_id = get_case_id(cases_generator_func)
    args = case_id.split('-')[1:]  # Exclude first argument
    kwargs = parse_args(args)

    is_input_quanttensor = kwargs['io_quant'] is not None or kwargs['input_quantized']

    if (not (is_input_quanttensor and kwargs['weight_quant'] is not None) and
            kwargs['io_quant'] is None) and kwargs['return_quant_tensor']:
        with pytest.raises(RuntimeError, match='QuantLayer is not correctly configured'):
            output = model(input)
        return
    elif (not is_input_quanttensor or
          kwargs['weight_quant'] is None) and kwargs['bias_quant'] == 'quant_external':
        with pytest.raises(RuntimeError, match='Input scale required'):
            output = model(input)
        return
    elif kwargs['weight_quant'] == 'quant_asym' and kwargs['return_quant_tensor'] and kwargs['io_quant'] is None \
        and kwargs['input_quantized']:
        with pytest.raises(
                AssertionError,
                match='QuantLayer is not correctly configured, check if warnings were raised'):
            output = model(input)
        return
    else:
        output = model(input)

    if kwargs['return_quant_tensor']:
        assert isinstance(output, QuantTensor)
    else:
        assert isinstance(output, torch.Tensor)


@pytest_cases.parametrize_with_cases(
    'model_input', cases=[case_quant_lstm_full, case_quant_rnn_full])
def test_quant_lstm_rnn_full(model_input, current_cases):

    cases_generator_func = current_cases['model_input'][1]
    case_id = get_case_id(cases_generator_func)
    args = case_id.split('-')
    kwargs = parse_args(args)

    is_input_quanttensor = kwargs['io_quant'] is not None or kwargs['input_quantized']
    return_quant_tensor = kwargs['return_quant_tensor']

    model, input = model_input
    if (kwargs['bias_quant'] == 'quant_external') and ( \
        (not is_input_quanttensor or kwargs['weight_quant'] is None) or \
        (kwargs['num_layers']> 1 and (kwargs['weight_quant'] is None or kwargs['io_quant'] is None))):
        with pytest.raises(RuntimeError, match='Input scale required'):
            output = model(input)
        return
    else:
        output = model(input)
    if len(output) == 1:
        output = output[0]
        h, c = None, None
    elif len(output) == 2:
        if 'quant_lstm' in args[0]:
            output, (h, c) = output
        else:
            output, h = output
            c = None

    if return_quant_tensor:
        assert isinstance(output, QuantTensor)
    else:
        assert isinstance(output, torch.Tensor)

    if h is not None:
        if return_quant_tensor:
            assert isinstance(h, QuantTensor)
        else:
            assert isinstance(h, torch.Tensor)

    if c is not None:
        if return_quant_tensor:
            assert isinstance(c, QuantTensor)
        else:
            assert isinstance(c, torch.Tensor)


@pytest_cases.parametrize_with_cases('model_input', cases=[case_quant_lstm, case_quant_rnn])
def test_quant_lstm_rnn(model_input, current_cases):
    model, input = model_input

    cases_generator_func = current_cases['model_input'][1]
    case_id = get_case_id(cases_generator_func)
    args = case_id.split('-')
    kwargs = parse_args(args)

    is_input_quanttensor = kwargs['io_quant'] is not None or kwargs['input_quantized']

    if (kwargs['bias_quant'] == 'quant_external') and ( \
        (not is_input_quanttensor or kwargs['weight_quant'] is None) or \
        (kwargs['num_layers']> 1 and (kwargs['weight_quant'] is None or kwargs['io_quant'] is None))):
        with pytest.raises(RuntimeError, match='Input scale required'):
            output = model(input)
        return
    else:
        output = model(input)
    if len(output) == 1:
        output = output[0]
        h, c = None, None
    elif len(output) == 2:
        if args[0] == 'quant_lstm':
            output, (h, c) = output
        else:
            output, h = output
            c = None
    return_quant_tensor = kwargs['return_quant_tensor']

    if return_quant_tensor:
        assert isinstance(output, QuantTensor)
    else:
        assert isinstance(output, torch.Tensor)

    if h is not None:
        if return_quant_tensor:
            assert isinstance(h, QuantTensor)
        else:
            assert isinstance(h, torch.Tensor)

    if c is not None:
        if return_quant_tensor:
            assert isinstance(c, QuantTensor)
        else:
            assert isinstance(c, torch.Tensor)


@pytest_cases.parametrize_with_cases('model_input', cases=case_mha)
def test_quant_mha(model_input, current_cases):
    model, inp = model_input

    cases_generator_func = current_cases['model_input'][1]
    case_id = get_case_id(cases_generator_func)
    args = case_id.split('-')[1:]  # Exclude first argument
    kwargs = parse_args(args)

    is_input_quanttensor = kwargs['io_quant'] is not None or kwargs['input_quantized']
    if (not is_input_quanttensor or
            kwargs['weight_quant'] is None) and kwargs['bias_quant'] == 'quant_external':
        with pytest.raises(RuntimeError, match='Input scale required'):
            output, _ = model(inp, inp, inp)
        return
    elif kwargs['io_quant'] is None and kwargs['return_quant_tensor']:
        with pytest.raises(RuntimeError, match='QuantLayer is not correctly configured'):
            output, _ = model(inp, inp, inp)
        return
    elif kwargs['io_quant'] is None and kwargs['bias_quant'] == 'quant_external':
        with pytest.raises(RuntimeError, match='Input scale required'):
            output, _ = model(inp, inp, inp)
        return
    elif kwargs['weight_quant'] is not None and kwargs['io_quant'] is None:
        if kwargs['weight_quant'] == 'quant_asym' and kwargs['return_quant_tensor']:
            with pytest.raises(
                    AssertionError,
                    match='QuantLayer is not correctly configured, check if warnings were raised'):
                output, _ = model(inp, inp, inp)
            return
    output, _ = model(inp, inp, inp)

    if kwargs['return_quant_tensor']:
        assert isinstance(output, QuantTensor)
    else:
        assert isinstance(output, torch.Tensor)
