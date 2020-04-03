# Copyright (c) 2019-     Xilinx, Inc              (Giuseppe Franco)
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

import brevitas.nn as quant_nn
import torch
from brevitas.quant_tensor import pack_quant_tensor
import brevitas.config
from brevitas.core.scaling import ScalingImplType
import pytest
from hypothesis import given, example, note
import hypothesis.strategies as st
from common import float_st, float_st_nz, two_lists_equal_size, list_float_st, float_st_p, generate_quant_input
from common import check_dynamic_quant_jit_skip, check_expected_pyt_120_fail, combine_conditions

# Constants
MIN_BITWIDTH = 3
MAX_BITWIDTH = 8

BATCH_SIZE = 2
INPUT_CHANNEL = 2
WIDTH = 3
LENGTH = 3
OUTPUT_CHANNEL = 3
KERNEL_SIZE = 3

ATOL_DYN = 1e-04
RTOL_DYN = 1e-04
weight_scaling_impl_type_options = [('STATS'), ('CONST'), ('PARAMETER_FROM_STATS')]
activation_scaling_impl_type_options = [('STATS'), ('CONST'), ('PARAMETER')]


def perform_dynamic_quant_test_weight(input_fp, input_quant, layer_type, layer_config, quant_config):
    layer = layer_type(*layer_config, **quant_config)
    layer.eval()

    brevitas.config.USE_DYNAMIC_QUANTIZATION = True
    output = layer(input_quant)[0]

    # Expected
    brevitas.config.USE_DYNAMIC_QUANTIZATION = False
    expected_output = layer(input_fp)

    assert torch.allclose(expected_output, output, RTOL_DYN, ATOL_DYN)


def perform_dynamic_quant_test_activation(input_fp, layer_type, quant_config):
    layer = layer_type(**quant_config)
    layer.eval()

    brevitas.config.USE_DYNAMIC_QUANTIZATION = True
    output = layer(input_fp)[0]

    # Expected
    brevitas.config.USE_DYNAMIC_QUANTIZATION = False
    expected_output = layer(input_fp)

    assert torch.allclose(expected_output, output, RTOL_DYN, ATOL_DYN)

@given(input_quant_scale_bit=generate_quant_input(MIN_BITWIDTH, MAX_BITWIDTH),
       bit_width_layer=st.integers(min_value=MIN_BITWIDTH, max_value=MAX_BITWIDTH))
@pytest.mark.parametrize('weight_scaling_impl_type', weight_scaling_impl_type_options)
@combine_conditions(check_expected_pyt_120_fail, check_dynamic_quant_jit_skip)
def test_conv2d(input_quant_scale_bit, bit_width_layer, weight_scaling_impl_type):
    input_quant = input_quant_scale_bit[0]
    scale = input_quant_scale_bit[1]
    shape_4d = (BATCH_SIZE, INPUT_CHANNEL, WIDTH, LENGTH)
    input_quant = input_quant.view(shape_4d)

    bit_width = input_quant_scale_bit[2]
    input_fp = input_quant * scale

    input_q = pack_quant_tensor(input_fp, scale, bit_width)

    weight_quant_type = 'INT'

    config = {
        'bias': True,
        'weight_bit_width': bit_width_layer,
        'weight_quant_type': weight_quant_type,
        'weight_scaling_impl_type': weight_scaling_impl_type,
        'weight_scaling_const': 0.0001
    }
    layer_config = [INPUT_CHANNEL, OUTPUT_CHANNEL, KERNEL_SIZE]

    perform_dynamic_quant_test_weight(input_fp, input_q, quant_nn.QuantConv2d, layer_config, config)



@given(input_quant_scale_bit=generate_quant_input(MIN_BITWIDTH, MAX_BITWIDTH),
       bit_width_layer=st.integers(min_value=MIN_BITWIDTH, max_value=MAX_BITWIDTH))
@pytest.mark.parametrize('weight_scaling_impl_type', weight_scaling_impl_type_options)
@combine_conditions(check_expected_pyt_120_fail, check_dynamic_quant_jit_skip)
def test_linear(input_quant_scale_bit, bit_width_layer, weight_scaling_impl_type):
    input_quant = input_quant_scale_bit[0]
    scale = input_quant_scale_bit[1]
    bit_width = input_quant_scale_bit[2]
    shape_3d = (BATCH_SIZE, WIDTH*LENGTH, INPUT_CHANNEL)
    input_quant = input_quant.view(shape_3d)
    input_fp = input_quant * scale

    input_q = pack_quant_tensor(input_fp, scale, bit_width)

    weight_quant_type = 'INT'

    config = {
        'bias': True,
        'weight_bit_width': bit_width_layer,
        'weight_quant_type': weight_quant_type,
        'weight_scaling_impl_type': weight_scaling_impl_type,
        'weight_scaling_const': 0.0001
    }
    layer_config = [INPUT_CHANNEL, OUTPUT_CHANNEL]
    perform_dynamic_quant_test_weight(input_fp, input_q, quant_nn.QuantLinear, layer_config, config)


@given(input_quant_scale_bit=generate_quant_input(MIN_BITWIDTH, MAX_BITWIDTH),
       bit_width_layer=st.integers(min_value=MIN_BITWIDTH, max_value=MAX_BITWIDTH))
@pytest.mark.parametrize('weight_scaling_impl_type', weight_scaling_impl_type_options)
@combine_conditions(check_expected_pyt_120_fail, check_dynamic_quant_jit_skip)
def test_conv1d(input_quant_scale_bit, bit_width_layer, weight_scaling_impl_type):
    input_quant = input_quant_scale_bit[0]
    scale = input_quant_scale_bit[1]
    bit_width = input_quant_scale_bit[2]
    shape_3d = (BATCH_SIZE, INPUT_CHANNEL, WIDTH*LENGTH)
    input_quant = input_quant.view(shape_3d)
    input_fp = input_quant * scale

    input_q = pack_quant_tensor(input_fp, scale, bit_width)

    weight_quant_type = 'INT'

    config = {
        'bias': True,
        'weight_bit_width': bit_width_layer,
        'weight_quant_type': weight_quant_type,
        'weight_scaling_impl_type': weight_scaling_impl_type,
        'weight_scaling_const': 0.0001
    }
    layer_config = [INPUT_CHANNEL, OUTPUT_CHANNEL, KERNEL_SIZE]
    perform_dynamic_quant_test_weight(input_fp, input_q, quant_nn.QuantConv1d, layer_config, config)


@given(input_quant_scale_bit=generate_quant_input(MIN_BITWIDTH, MAX_BITWIDTH),
       bit_width_layer=st.integers(min_value=MIN_BITWIDTH, max_value=MAX_BITWIDTH))
@pytest.mark.parametrize('weight_scaling_impl_type', weight_scaling_impl_type_options)
@combine_conditions(check_expected_pyt_120_fail, check_dynamic_quant_jit_skip)
def test_transposedconv1d(input_quant_scale_bit, bit_width_layer, weight_scaling_impl_type):
    input_quant = input_quant_scale_bit[0]
    scale = input_quant_scale_bit[1]
    bit_width = input_quant_scale_bit[2]
    shape_3d = (BATCH_SIZE, INPUT_CHANNEL, WIDTH*LENGTH)
    input_quant = input_quant.view(shape_3d)
    input_fp = input_quant * scale

    input_q = pack_quant_tensor(input_fp, scale, bit_width)

    weight_quant_type = 'INT'

    config = {
        'bias': True,
        'weight_bit_width': bit_width_layer,
        'weight_quant_type': weight_quant_type,
        'weight_scaling_impl_type': weight_scaling_impl_type,
        'weight_scaling_const': 0.0001
    }
    layer_config = [INPUT_CHANNEL, OUTPUT_CHANNEL, KERNEL_SIZE]
    perform_dynamic_quant_test_weight(input_fp, input_q, quant_nn.QuantConvTranspose1d, layer_config, config)


@given(input_quant_scale_bit=generate_quant_input(MIN_BITWIDTH, MAX_BITWIDTH),
       bit_width_layer=st.integers(min_value=MIN_BITWIDTH, max_value=MAX_BITWIDTH))
@pytest.mark.parametrize('weight_scaling_impl_type', weight_scaling_impl_type_options)
@combine_conditions(check_expected_pyt_120_fail, check_dynamic_quant_jit_skip)
def test_quantscalebias(input_quant_scale_bit, bit_width_layer, weight_scaling_impl_type):
    input_quant = input_quant_scale_bit[0]
    scale = input_quant_scale_bit[1]
    bit_width = input_quant_scale_bit[2]
    shape_2d = (BATCH_SIZE, INPUT_CHANNEL*WIDTH*LENGTH)
    input_quant = input_quant.view(shape_2d)
    input_fp = input_quant * scale

    input_q = pack_quant_tensor(input_fp, scale, bit_width)

    weight_quant_type = 'INT'

    config = {
        'weight_bit_width': bit_width_layer,
        'weight_quant_type': weight_quant_type,
        'weight_scaling_impl_type': weight_scaling_impl_type,
        'weight_scaling_const': 0.0001
    }
    layer_config = [INPUT_CHANNEL*WIDTH*LENGTH]
    perform_dynamic_quant_test_weight(input_fp, input_q, quant_nn.QuantScaleBias, layer_config, config)


@given(input_quant_scale_bit=generate_quant_input(MIN_BITWIDTH, MAX_BITWIDTH),
       bit_width_layer=st.integers(min_value=MIN_BITWIDTH, max_value=MAX_BITWIDTH))
@pytest.mark.parametrize('activation_scaling_impl_type', activation_scaling_impl_type_options)
@combine_conditions(check_expected_pyt_120_fail, check_dynamic_quant_jit_skip)
def test_activationrelu(input_quant_scale_bit, bit_width_layer, activation_scaling_impl_type):
    input_quant = input_quant_scale_bit[0]
    scale = input_quant_scale_bit[1]

    shape_4d = (BATCH_SIZE, INPUT_CHANNEL, WIDTH, LENGTH)
    input_quant = input_quant.view(shape_4d)
    input_fp = input_quant * scale

    quant_type = 'INT'

    config = {
        'bit_width': bit_width_layer,
        'quant_type': quant_type,
        'scaling_impl_type': activation_scaling_impl_type,
        'max_val': 6
    }
    perform_dynamic_quant_test_activation(input_fp, quant_nn.QuantReLU, config)


@given(input_quant_scale_bit=generate_quant_input(MIN_BITWIDTH, MAX_BITWIDTH),
       bit_width_layer=st.integers(min_value=MIN_BITWIDTH, max_value=MAX_BITWIDTH))
@combine_conditions(check_expected_pyt_120_fail, check_dynamic_quant_jit_skip)
def test_activationsigmoid(input_quant_scale_bit, bit_width_layer):
    input_quant = input_quant_scale_bit[0]
    scale = input_quant_scale_bit[1]

    shape_4d = (BATCH_SIZE, INPUT_CHANNEL, WIDTH, LENGTH)
    input_quant = input_quant.view(shape_4d)
    input_fp = input_quant * scale

    quant_type = 'INT'

    config = {
        'bit_width': bit_width_layer,
        'quant_type': quant_type,
    }
    perform_dynamic_quant_test_activation(input_fp, quant_nn.QuantSigmoid, config)


@given(input_quant_scale_bit=generate_quant_input(MIN_BITWIDTH, MAX_BITWIDTH),
       bit_width_layer=st.integers(min_value=MIN_BITWIDTH, max_value=MAX_BITWIDTH))
@combine_conditions(check_expected_pyt_120_fail, check_dynamic_quant_jit_skip)
def test_activationtanh(input_quant_scale_bit, bit_width_layer):
    input_quant = input_quant_scale_bit[0]
    scale = input_quant_scale_bit[1]

    shape_4d = (BATCH_SIZE, INPUT_CHANNEL, WIDTH, LENGTH)
    input_quant = input_quant.view(shape_4d)
    input_fp = input_quant * scale

    quant_type = 'INT'

    config = {
        'bit_width': bit_width_layer,
        'quant_type': quant_type,
    }
    perform_dynamic_quant_test_activation(input_fp, quant_nn.QuantTanh, config)


@given(input_quant_scale_bit=generate_quant_input(MIN_BITWIDTH, MAX_BITWIDTH),
       bit_width_layer=st.integers(min_value=MIN_BITWIDTH, max_value=MAX_BITWIDTH))
@pytest.mark.parametrize('activation_scaling_impl_type', activation_scaling_impl_type_options)
@combine_conditions(check_expected_pyt_120_fail, check_dynamic_quant_jit_skip)
def test_activationhardtanh(input_quant_scale_bit, bit_width_layer, activation_scaling_impl_type):
    input_quant = input_quant_scale_bit[0]
    scale = input_quant_scale_bit[1]

    shape_4d = (BATCH_SIZE, INPUT_CHANNEL, WIDTH, LENGTH)
    input_quant = input_quant.view(shape_4d)
    input_fp = input_quant * scale

    quant_type = 'INT'

    config = {
        'bit_width': bit_width_layer,
        'quant_type': quant_type,
        'scaling_impl_type': activation_scaling_impl_type,
        'min_val': -10,
        'max_val': 10
    }
    perform_dynamic_quant_test_activation(input_fp, quant_nn.QuantHardTanh, config)