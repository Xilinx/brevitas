# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import pytest_cases
from pytest_cases import get_case_id
import torch
from torch import Tensor

from brevitas.quant_tensor import QuantTensor

from .nn_quantizers_fixture import case_model_a2q


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


def calc_a2q_acc_bit_width(
        weight_max_l1_norm: Tensor,
        input_bit_width: Tensor,
        input_is_signed: bool,
        min_val: Optional[float] = 1e-10):
    """Using the closed-form bounds on accumulator bit-width as derived in `A2Q: Accumulator-Aware Quantization with
    Guaranteed Overflow Avoidance`. This function returns the minimum accumulator bit-width that can be used without
    risk of overflow."""
    assert weight_max_l1_norm.numel() == 1
    input_is_signed = float(input_is_signed)
    weight_max_l1_norm = torch.clamp_min(weight_max_l1_norm, min_val)
    alpha = torch.log2(weight_max_l1_norm) + input_bit_width - input_is_signed
    phi = lambda x: torch.log2(1. + pow(2., -x))
    min_bit_width = alpha + phi(alpha) + 1.
    min_bit_width = torch.ceil(min_bit_width)
    return min_bit_width


def calc_a2q_plus_acc_bit_width(
        weight_max_l1_norm: Tensor,
        input_bit_width: Tensor,
        input_is_signed: bool,
        min_val: Optional[float] = 1e-10):
    """Using the closed-form bounds on accumulator bit-width as derived in `A2Q+:
    Improving Accumulator-Aware Weight Quantization`. This function returns the
    minimum accumulator bit-width that can be used without risk of overflow,
    assuming that the floating-point weights are zero-centered."""
    input_is_signed = float(input_is_signed)
    assert weight_max_l1_norm.numel() == 1
    weight_max_l1_norm = torch.clamp_min(weight_max_l1_norm, min_val)
    input_range = pow(2., input_bit_width) - 1.  # 2^N - 1.
    min_bit_width = torch.log2(weight_max_l1_norm * input_range + 2.)
    min_bit_width = torch.ceil(min_bit_width)
    return min_bit_width


calc_fnc = {"quant_a2q": calc_a2q_acc_bit_width, "quant_a2q_plus": calc_a2q_plus_acc_bit_width}


@pytest_cases.parametrize_with_cases('model_input', cases=case_model_a2q)
def test_quant_wbiol_a2q(model_input, current_cases):
    """This test only verifies that the accumulator-aware weight quantization constraints the l1-norm of
    the weights enough use the user-specified accumulator bit-width. Baseline functionality is in the
    test_nn_quantizers."""
    model, input = model_input

    cases_generator_func = current_cases['model_input'][1]
    case_id = get_case_id(cases_generator_func)
    args = case_id.split('-')[1:]  # Exclude first argument
    kwargs = parse_args(args)
    fnc = calc_fnc[kwargs['weight_quant']]

    # A2Q needs to have a quantized input, which can be done by input quantizer or returning
    # a quantized tensor from the preceding layer
    is_input_quant_tensor = kwargs['io_quant'] is not None or isinstance(input, QuantTensor)
    assert is_input_quant_tensor, "All A2Q models require quantized inputs."

    # testing the forward pass
    output = model(input)

    # bit-width and sign need to come from the quant tensor of the preceding layer if no io_quant
    quant_input = model.conv.input_quant(input)
    input_bit_width = quant_input.bit_width
    input_is_signed = quant_input.signed

    # the tensor quantizer requires a QuantTensor with specified bit-width and sign
    quant_weight = model.conv.quant_weight(quant_input)
    quant_weight = quant_weight.int().float()
    if kwargs['model_type'] == 'QuantLinear':  # shape = (out_features, in_features)
        quant_weight_per_channel_l1_norm = quant_weight.norm(p=1, dim=1)
    elif kwargs['model_type'] == 'QuantConv1d':  # shape = (out_channels, in_channels, kernel_size)
        quant_weight_per_channel_l1_norm = quant_weight.norm(p=1, dim=(1, 2))
    elif kwargs[
            'model_type'] == 'QuantConv2d':  # shape = (out_channels, in_channels, kernel_size, kernel_size)
        quant_weight_per_channel_l1_norm = quant_weight.norm(p=1, dim=(1, 2, 3))
    elif kwargs[
            'model_type'] == 'QuantConvTranspose1d':  # shape = (in_channels, out_channels, kernel_size)
        quant_weight_per_channel_l1_norm = quant_weight.norm(p=1, dim=(0, 2))
    elif kwargs[
            'model_type'] == 'QuantConvTranspose2d':  # shape = (in_channels, out_channels, kernel_size)
        quant_weight_per_channel_l1_norm = quant_weight.norm(p=1, dim=(0, 2, 3))
    else:
        raise NotImplementedError(f"Check for {kwargs['model_type']} is not yet implemented.")

    # using the closed-form bounds on accumulator bit-width
    cur_acc_bit_width = fnc(
        quant_weight_per_channel_l1_norm.max(), input_bit_width, input_is_signed)
    exp_acc_bit_width = kwargs['accumulator_bit_width']
    assert cur_acc_bit_width <= exp_acc_bit_width, \
        f"Model does not satisfy accumulator bit-width bounds. Expected {exp_acc_bit_width}, got {cur_acc_bit_width}"
