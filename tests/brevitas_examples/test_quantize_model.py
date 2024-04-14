from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from brevitas.nn import QuantConv2d
from brevitas.nn import QuantLinear
from brevitas.nn import QuantReLU
from brevitas.quant_tensor import QuantTensor
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model

# TODO:
# - finish minifloat testing
# - Possibility to use statistics or MSE for scale factor computations for weights and activations.
# - Percentiles used for the activations' statistics computation during calibration.

# CONSTANTS
IMAGE_DIM = 16

# Random seed (set because in affine quant we test zero-points aren't zero.
# There may be a random seed that sets one of them to 0.)
torch.manual_seed(0)


##################
# EXAMPLE MODELS #
##################
@pytest.fixture
def minimal_model():
    """
    Inputs:
    Implictly takes in a torch.Tensor, size: (batch_size, 3, IMAGE_DIM, IMAGE_DIM).

    Outputs:
    Implictly returns a torch.Tensor, size: (batch_size, 16, IMAGE_DIM, IMAGE_DIM)

    This model has 3 input channels. I.e., for `layerwise` quantization it will use `layerwise_first_last_bit_width`
    bits for the weight and activation quantization.
    """
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
    )


@pytest.fixture
def simple_model():
    """
    Inputs:
    Implictly takes in a torch.Tensor, size: (batch_size, 10, IMAGE_DIM, IMAGE_DIM).

    Outputs:
    Implictly returns a torch.Tensor, size: (batch_size, 1000).

    """
    assert IMAGE_DIM % 2 == 0, "`IMAGE_DIM` should be a multiple of 2"
    return nn.Sequential(
        nn.Conv2d(10, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),  # downsample from IMAGE_DIM to half that
        nn.Flatten(),
        nn.Linear(32 * int(IMAGE_DIM / 2) ** 2, 1000))


##############
# Unit tests #
##############


###################
# FX MODE TESTING #
###################
@pytest.mark.parametrize("weight_bit_width", [2, 8, 16])
@pytest.mark.parametrize("act_bit_width", [2, 5, 8])
@pytest.mark.parametrize("bias_bit_width", [16, 32, 0])
def test_fx_model(simple_model, weight_bit_width, bias_bit_width, act_bit_width):
    """
    We test:
    - The FX-graph, quantized model is a GraphModule.
    - We can feed data through the model.
    - That the weight, bias and input/output quantization is toggled as expected.
    - That setting `None` for the `bias_bit_width` returns a dequantized bias.
    - That the bit widths are as desired.
    """
    fx_model = torch.fx.symbolic_trace(simple_model)
    quant_model = quantize_model(
        model=fx_model,
        backend='fx',
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        bias_bit_width=bias_bit_width if bias_bit_width > 0 else None,
        weight_quant_granularity='per_tensor',
        act_quant_percentile=99.9,
        act_quant_type='sym',
        scale_factor_type='float_scale',
        quant_format='int',
        layerwise_first_last_bit_width=5,
    )
    # Assert it is a GraphModule
    assert isinstance(quant_model, torch.fx.graph_module.GraphModule)

    # Assert we can feed data of the correct size through the model
    quant_model(torch.rand(1, 10, IMAGE_DIM, IMAGE_DIM))

    # Get first/last layer for testing its quantization.
    first_conv_layer = quant_model.get_submodule('0')
    first_relu_layer = quant_model.get_submodule('1')
    last_layer = quant_model.get_submodule('6')
    last_layer_output = quant_model.get_submodule('_6_output_quant')

    # Assert the module types are as desired
    assert isinstance(first_conv_layer, QuantConv2d)
    assert isinstance(last_layer, QuantLinear)

    # Check quantizaton is toggled as expected
    if bias_bit_width == 0:
        # If bias_bit_width is set as `None` (local variable value in scope of this function is 0),
        # the bias should be dequantized.
        assert not first_conv_layer.bias_quant.is_quant_enabled
    else:
        assert first_conv_layer.bias_quant.is_quant_enabled
    assert first_conv_layer.weight_quant.is_quant_enabled
    assert not first_conv_layer.input_quant.is_quant_enabled  # unlike with the layerwise backend, the input quantization is disabled.
    assert not first_conv_layer.output_quant.is_quant_enabled

    assert not first_relu_layer.input_quant.is_quant_enabled
    assert first_relu_layer.act_quant.is_quant_enabled  # the output of the "fused" ConvReLU is quantized

    # Assert types are as expected
    assert isinstance(quant_model.get_submodule('3'), QuantReLU)

    # Assert quantization bit widths are as desired
    # Biases
    if bias_bit_width > 0:
        assert first_conv_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers[
            'value'].item() == bias_bit_width
        assert last_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers[
            'value'].item() == bias_bit_width
    else:
        # If bias_bit_width is `None`, the quantized bias should return a fully floating point parameter.
        assert not isinstance(first_conv_layer.quant_bias(), QuantTensor)

    # Weights
    assert first_conv_layer.weight_quant.bit_width().item() == weight_bit_width
    assert last_layer.weight_quant.bit_width().item() == weight_bit_width
    # Activations
    assert first_relu_layer.act_quant.bit_width().item() == act_bit_width
    assert last_layer_output.act_quant.bit_width().item() == act_bit_width


def test_fx_sym_quant(simple_model):
    """
    We test fx quantization, with symmetric quantization for weights and activations.

    We test:
    - We can feed data through the model.
    - That the weight, bias and input/output quantization is toggled as expected.
    - That the quantization is symmetric.
    - That the bit widths are as desired.
    """
    weight_bit_width = 8
    act_bit_width = 8
    bias_bit_width = 32

    fx_model = torch.fx.symbolic_trace(simple_model)
    quant_model = quantize_model(
        model=fx_model,
        backend='fx',
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        bias_bit_width=bias_bit_width,
        weight_quant_granularity='per_tensor',
        act_quant_percentile=99.9,
        act_quant_type='sym',
        weight_quant_type='sym',
        scale_factor_type='float_scale',
        quant_format='int',
        layerwise_first_last_bit_width=5,
    )
    # Assert we can feed data of the correct size through the model
    quant_model(torch.rand(1, 10, IMAGE_DIM, IMAGE_DIM))

    # Get first/last layer for testing its quantization.
    first_conv_layer = quant_model.get_submodule('0')
    first_relu_layer = quant_model.get_submodule('1')
    last_layer = quant_model.get_submodule('6')
    last_layer_output = quant_model.get_submodule('_6_output_quant')

    # Check quantizaton is toggled as expected
    assert first_conv_layer.bias_quant.is_quant_enabled
    assert first_conv_layer.weight_quant.is_quant_enabled
    assert not first_conv_layer.input_quant.is_quant_enabled  # unlike with the layerwise backend, the input quantization is disabled.
    assert not first_conv_layer.output_quant.is_quant_enabled

    assert not first_relu_layer.input_quant.is_quant_enabled
    assert first_relu_layer.act_quant.is_quant_enabled  # the output of the "fused" ConvReLU is quantized

    # Assert the tensors are signed as expected for symmetric quantization, with zero-points at 0.
    # Weights
    assert first_conv_layer.quant_weight().signed_t
    assert torch.isclose(first_conv_layer.quant_weight().zero_point, torch.Tensor([0.0]))
    assert last_layer.quant_weight().signed_t
    assert torch.isclose(last_layer.quant_weight().zero_point, torch.Tensor([0.0]))

    # Activations
    assert last_layer_output.act_quant.is_signed
    assert torch.isclose(last_layer_output.act_quant.zero_point(), torch.tensor([0.0]))

    # Assert quantization bit widths are as desired
    # Biases
    assert first_conv_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers[
        'value'].item() == bias_bit_width
    assert last_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers[
        'value'].item() == bias_bit_width
    # Weights
    assert first_conv_layer.weight_quant.bit_width().item() == weight_bit_width
    assert last_layer.weight_quant.bit_width().item() == weight_bit_width
    # Activations
    assert first_relu_layer.act_quant.bit_width().item() == act_bit_width
    assert last_layer_output.act_quant.bit_width().item() == act_bit_width


def test_fx_affine_quantization(simple_model):
    """
    We test asymmetric quantization of the weights and activations.

    We test:
    - We can feed data through the model.
    - That the weights and activations are quantized on a affine basis.
    - That the bit widths are as desired.
    """
    weight_bit_width = 8
    act_bit_width = 8
    bias_bit_width = 32

    fx_model = torch.fx.symbolic_trace(simple_model)
    quant_model = quantize_model(
        model=fx_model,
        backend='fx',
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        bias_bit_width=bias_bit_width,
        weight_quant_granularity='per_tensor',
        act_quant_percentile=99.9,
        act_quant_type='asym',
        weight_quant_type='asym',
        scale_factor_type='float_scale',
        quant_format='int',
    )

    # Assert we can feed data of the correct size through the model
    quant_model(torch.rand(1, 10, IMAGE_DIM, IMAGE_DIM))

    # Get first/last layer for testing its quantization.
    first_conv_layer = quant_model.get_submodule('0')
    last_layer = quant_model.get_submodule('6')
    last_layer_output = quant_model.get_submodule('_6_output_quant')

    # Assert the tensors are unsigned as expected for asymmetric quantization, with zero-points not at 0.
    # Weights
    assert not first_conv_layer.weight_quant.is_signed
    assert not torch.isclose(first_conv_layer.quant_weight().zero_point, torch.tensor([0.0]))
    assert not last_layer.weight_quant.is_signed
    assert not torch.isclose(last_layer.quant_weight().zero_point, torch.Tensor([0.0]))

    # Activations
    assert not last_layer_output.act_quant.is_signed
    assert not torch.isclose(last_layer_output.act_quant.zero_point(), torch.tensor([0.0]))

    # Assert quantization bit widths are as desired
    # Biases
    assert first_conv_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers[
        'value'].item() == bias_bit_width
    assert last_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers[
        'value'].item() == bias_bit_width
    # Weights
    assert first_conv_layer.weight_quant.bit_width().item() == weight_bit_width
    assert last_layer.weight_quant.bit_width().item() == weight_bit_width
    # Activation
    assert last_layer_output.act_quant.bit_width().item() == act_bit_width


def test_fx_per_chan_weight_quantization(simple_model):
    """
    We test per-channel weight quantization.

    We test:
    - We can feed data through the model.
    - That the weights are quantized on a per-channel basis.
    - That the bit widths are as desired.
    """
    weight_bit_width = 8
    act_bit_width = 8
    bias_bit_width = 32

    fx_model = torch.fx.symbolic_trace(simple_model)
    quant_model = quantize_model(
        model=fx_model,
        backend='fx',
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        bias_bit_width=bias_bit_width,
        weight_quant_granularity='per_channel',
        act_quant_percentile=99.9,
        act_quant_type='sym',
        scale_factor_type='float_scale',
        quant_format='float',
    )

    # Assert we can feed data of the correct size through the model
    quant_model(torch.rand(1, 10, IMAGE_DIM, IMAGE_DIM))

    # Get first/last layer for testing its quantization.
    first_conv_layer = quant_model.get_submodule('0')
    last_layer = quant_model.get_submodule('6')
    last_layer_output = quant_model.get_submodule('_6_output_quant')

    # Assert per-channel quantization of weights
    # 16 is the nb of output channels of first layer of `simple_model`
    assert len(first_conv_layer.weight_quant.scale()) == 16
    # 1000 is the nb of output channels of last layer of `simple_model`
    assert len(last_layer.weight_quant.scale()) == 1000

    # Assert quantization bit widths are as desired
    # Biases
    assert first_conv_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers[
        'value'].item() == bias_bit_width
    assert last_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers[
        'value'].item() == bias_bit_width
    # Weights
    assert first_conv_layer.weight_quant.bit_width().item() == weight_bit_width
    assert last_layer.weight_quant.bit_width().item() == weight_bit_width
    # Activation
    assert last_layer_output.act_quant.bit_width().item() == act_bit_width


def test_invalid_input(simple_model):
    """
    We test various invalid inputs, e.g. invalid strings and zero/negative bit widths.
    """
    fx_model = torch.fx.symbolic_trace(simple_model)
    with pytest.raises(KeyError):
        quantize_model(
            model=fx_model,
            backend='invalid_backend',  # invalid input
            weight_bit_width=8,
            act_bit_width=8,
            bias_bit_width=32,
            weight_quant_granularity='per_tensor',
            act_quant_percentile=99.9,
            act_quant_type='sym',
            scale_factor_type='float_scale',
            quant_format='int',
        )
    with pytest.raises(KeyError):
        quantize_model(
            model=fx_model,
            backend='fx',
            weight_bit_width=8,
            act_bit_width=8,
            bias_bit_width=32,
            weight_quant_granularity='per_tensor',
            act_quant_percentile=99.9,
            act_quant_type='sym',
            scale_factor_type='invalid_scale',  # invalid input
            quant_format='int',
        )
    # Test that zero values are invalid for bit widths
    with pytest.raises(AssertionError):
        quantize_model(
            model=fx_model,
            backend='fx',
            weight_bit_width=0.0,  # NOTE: invalid input
            act_bit_width=0.0,  # NOTE: invalid input
            bias_bit_width=32,
            weight_quant_granularity='per_tensor',
            act_quant_percentile=99.9,
            act_quant_type='sym',
            scale_factor_type='float_scale',
            quant_format='int',
        )
    # Test that negative values are invalid for bit widths
    with pytest.raises(AssertionError):
        quantize_model(
            model=fx_model,
            backend='fx',
            weight_bit_width=-1,  # NOTE: invalid input
            act_bit_width=-1,  # NOTE: invalid input
            bias_bit_width=32,
            weight_quant_granularity='per_tensor',
            act_quant_percentile=99.9,
            act_quant_type='sym',
            scale_factor_type='float_scale',
            quant_format='int',
        )
    # Test that invalid bias values are caught
    with pytest.raises(KeyError):
        quantize_model(
            model=fx_model,
            backend='fx',
            weight_bit_width=8,
            act_bit_width=8,
            bias_bit_width=25,  # Valid values (16, 32)
            weight_quant_granularity='per_tensor',
            act_quant_percentile=99.9,
            act_quant_type='sym',
            scale_factor_type='float_scale',
            quant_format='int',
        )


##########################
# LAYERWISE MODE TESTING #
##########################
def test_layerwise_minifloat_invalid_bitwidths(minimal_model):
    """
    We test invalid mantissa and exponent bit widths. The mantissa + exponent + signed
    should equal the total bit width for weights and activations, if doing
    minifloat quantization (scale_factor_type='float_scale', quant_format='float').

    We use a model with 3 input channels, and test `layerwise` quantization.
    Because of the 3 input channels, this will set the quantization bit width
    of the weights and activations of the first layer to be equal to
    `layerwise_first_last_bit_width`.

    We test:
    - That the quantization will throw an error because the bit widths do not sum together correctly.
    """

    weight_bit_width = 8
    act_bit_width = 8
    bias_bit_width = 32

    with pytest.raises(RuntimeError):
        quantize_model(
            model=deepcopy(minimal_model),
            backend='layerwise',
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            bias_bit_width=bias_bit_width,
            weight_quant_granularity='per_tensor',
            act_quant_percentile=99.9,
            act_quant_type='sym',  # signed bit = 1
            scale_factor_type='float_scale',
            quant_format='float',
            layerwise_first_last_bit_width=13,  # invalid value, should be 11
            layerwise_first_last_mantissa_bit_width=7,
            layerwise_first_last_exponent_bit_width=3,
            weight_mantissa_bit_width=6,
            weight_exponent_bit_width=4,
            act_mantissa_bit_width=5,
            act_exponent_bit_width=5,
        )


def test_layerwise_valid_minifloat_bit_widths(minimal_model):
    """
    We test valid mantissa and exponent bit widths. The mantissa + exponent + signed
    should equal the total bit width for weights and activations, if doing
    minifloat quantization (scale_factor_type='float_scale', quant_format='float').

    See paper for details: https://arxiv.org/abs/2311.12359

    We use a model with 3 input channels, and test `layerwise` quantization.
    Because of the 3 input channels, this will set the quantization bit width
    of the weights and activations of the first layer to be equal to
    `layerwise_first_last_bit_width`.

    We test:
    - That the bit widths are as desired.
    """

    weight_bit_width = 8
    act_bit_width = 8
    bias_bit_width = 32
    layerwise_first_last_bit_width = 11
    layerwise_first_last_mantissa_bit_width = 4
    layerwise_first_last_exponent_bit_width = 6
    weight_mantissa_bit_width = 3
    weight_exponent_bit_width = 7
    act_mantissa_bit_width = 5
    act_exponent_bit_width = 5

    quant_model = quantize_model(
        model=deepcopy(minimal_model),
        backend='layerwise',
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        bias_bit_width=bias_bit_width,
        weight_quant_granularity='per_tensor',
        act_quant_percentile=99.9,
        act_quant_type='sym',
        scale_factor_type='float_scale',
        quant_format='float',
        layerwise_first_last_bit_width=layerwise_first_last_bit_width,
        layerwise_first_last_mantissa_bit_width=layerwise_first_last_mantissa_bit_width,
        layerwise_first_last_exponent_bit_width=layerwise_first_last_exponent_bit_width,
        weight_mantissa_bit_width=weight_mantissa_bit_width,
        weight_exponent_bit_width=weight_exponent_bit_width,
        act_mantissa_bit_width=act_mantissa_bit_width,
        act_exponent_bit_width=act_exponent_bit_width,
    )
    assert isinstance(quant_model, nn.Sequential)

    # Make sure we can feed data through the model
    _ = quant_model(torch.rand(1, 3, IMAGE_DIM, IMAGE_DIM))

    # Get first layer for testing its quantization.
    # We also test we can feed data through the first layer in isolation
    first_layer = quant_model.get_submodule('0')
    first_layer_output = first_layer(torch.rand(1, 3, IMAGE_DIM, IMAGE_DIM))

    # Assert quantization bit widths are as desired
    # Biases
    assert first_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers[
        'value'].item() == bias_bit_width
    # Weights
    assert first_layer.weight_quant.bit_width().item() == layerwise_first_last_bit_width
    # Activations
    assert first_layer.input_quant.bit_width().item() == layerwise_first_last_bit_width

    # Verify outputs of a layer
    torch.manual_seed(0)
    x = torch.rand(1, 3, 10, 10)
    # layerwise_first_last_mantissa_bit_width=4
    # layerwise_first_last_exponent_bit_width=6
    # weight_mantissa_bit_width=3
    # weight_exponent_bit_width=7

    # Refer to paper for mini-float details: https://arxiv.org/pdf/2311.12359.pdf
    signed = first_layer.input_quant.is_signed
    if signed:
        qmin = -2 ** (layerwise_first_last_bit_width - 1)
        qmax = 2 ** (layerwise_first_last_bit_width - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** (layerwise_first_last_bit_width) - 1

    # scale = first_layer.input_quant.scale()
    scale = torch.max(x) / qmax
    m = act_mantissa_bit_width
    e = act_exponent_bit_width
    b = 2 ** (e - 1) - 1
    scaled_x = x / scale
    scaling_factor = 2 ** (
        torch.clamp(torch.floor(torch.log2(torch.abs(scaled_x))) - m, min=(1 - b - m)))

    x_q = torch.clamp(scaling_factor * (torch.round(scaled_x / scaling_factor)), min=qmin, max=qmax)
    x_q_test = x_q * scale
    x_test = first_layer.input_quant(x)
    # WIP, need to confirm these are equal and my implementation is correct


@pytest.mark.parametrize("weight_bit_width", [2, 5, 8, 16])
@pytest.mark.parametrize("act_bit_width", [2, 5, 8])
@pytest.mark.parametrize("bias_bit_width", [16, 32])
@pytest.mark.parametrize("layerwise_first_last_bit_width", [2, 8])
def test_layerwise_10_in_channels_quantize_model(
        simple_model, weight_bit_width, bias_bit_width, act_bit_width,
        layerwise_first_last_bit_width):
    """
    We use a model with 10 input channels, and test `layerwise` quantization.
    Because of the 10 input channels, this will ignore the `layerwise_first_last_bit_width`
    value, and will quantize everything according to:
    - weight_bit_width
    - bias_bit_width
    - act_bit_width

    We test:
    - We can feed data through the model.
    - The modules are of the Quant type.
    - That the weight, bias and input/output quantization is toggled as expected (only test the first layer).
    - That the bit widths are as desired.
    """
    quant_model = quantize_model(
        model=deepcopy(simple_model),
        backend='layerwise',
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        bias_bit_width=bias_bit_width,
        weight_quant_granularity='per_tensor',
        act_quant_percentile=99.9,
        act_quant_type='sym',
        scale_factor_type='float_scale',
        quant_format='int',
        layerwise_first_last_bit_width=layerwise_first_last_bit_width,
    )
    assert isinstance(quant_model, nn.Sequential)

    # Make sure we can feed data through the model
    _ = quant_model(torch.rand(1, 10, IMAGE_DIM, IMAGE_DIM))

    # Get first layer for testing its quantization.
    # We also test we can feed data through the first layer in isolation
    first_layer = quant_model.get_submodule('0')
    first_layer_output = first_layer(torch.rand(1, 10, IMAGE_DIM, IMAGE_DIM))

    # Assert the module types are as desired
    assert isinstance(first_layer, QuantConv2d)

    # Assert only weight is quantized by default
    # However, here input and bias are also quantized
    assert first_layer.weight_quant.is_quant_enabled
    assert first_layer.bias_quant.is_quant_enabled
    assert first_layer.input_quant.is_quant_enabled  # unlike with the fx backend, the input quantization is enabled.
    assert not first_layer.output_quant.is_quant_enabled
    # NOTE: The `layerwise` backend also differs from the `fx` backend in that: the input quantization is enabled
    # for the first Conv layer by default in the `layerwise`, whereas it is disabled in the `fx` backend. However,
    # in practice this is because the `fx` backend introduces an extra quantization module (i.e. QuantIdentity) before
    # the first layer that quantizes the input to the first layer, so it comes to the same: in both cases, the Conv
    # receives a quantized input tensor.

    # Assert quantization bit widths are as desired
    # Biases
    assert first_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers[
        'value'].item() == bias_bit_width
    # Weights
    assert first_layer.weight_quant.bit_width().item() == weight_bit_width
    # Activations
    assert first_layer.input_quant.bit_width().item() == act_bit_width


@pytest.mark.parametrize("weight_bit_width", [9, 16])
@pytest.mark.parametrize("act_bit_width", [5, 9])
@pytest.mark.parametrize("bias_bit_width", [16, 32])
@pytest.mark.parametrize("layerwise_first_last_bit_width", [2, 8])
def test_layerwise_3_in_channels_quantize_model(
        minimal_model, weight_bit_width, bias_bit_width, act_bit_width,
        layerwise_first_last_bit_width):
    """
    We use a model with 3 input channels, and test `layerwise` quantization.
    Because of the 3 input channels, this will set the quantization bit width
    of the weights and activations of the first layer to be equal to
    `layerwise_first_last_bit_width`.

    We test:
    - That the bit widths are as desired.
    """
    quant_model = quantize_model(
        model=deepcopy(minimal_model),
        backend='layerwise',
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        bias_bit_width=bias_bit_width,
        weight_quant_granularity='per_tensor',
        act_quant_percentile=99.9,
        act_quant_type='sym',
        scale_factor_type='float_scale',
        quant_format='int',
        layerwise_first_last_bit_width=layerwise_first_last_bit_width,
    )

    # Get first layer for testing its quantization.
    first_layer = quant_model.get_submodule('0')

    # Assert quantization bit widths are as desired
    # Biases
    assert first_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers[
        'value'].item() == bias_bit_width
    # Weights
    assert first_layer.weight_quant.bit_width().item() == layerwise_first_last_bit_width
    # Activations
    assert first_layer.input_quant.bit_width().item() == layerwise_first_last_bit_width


def test_po2_layerwise_quantization(simple_model):
    """
    We test:
    - We can feed data through the model.
    - That the quantization scales are powers of 2 as expected.
    - That the bit widths are as desired.
    """
    weight_bit_width = 8
    act_bit_width = 8
    bias_bit_width = 32

    quant_model = quantize_model(
        model=simple_model,
        backend='layerwise',
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        bias_bit_width=bias_bit_width,
        weight_quant_granularity='per_tensor',
        act_quant_percentile=99.9,
        act_quant_type='sym',
        scale_factor_type='po2_scale',  # float_scale
        quant_format='int',  # float
    )

    # Assert we can feed data of the correct size through the model
    quant_model(torch.rand(1, 10, IMAGE_DIM, IMAGE_DIM))

    # Get first/last layer for testing its quantization.
    first_conv_layer = quant_model.get_submodule('0')
    last_layer = quant_model.get_submodule('6')

    # Assert scales are powers of 2 as expected
    assert torch.isclose(torch.log2(first_conv_layer.input_quant.scale()) % 1, torch.Tensor([0.0]))
    assert torch.isclose(torch.log2(first_conv_layer.weight_quant.scale()) % 1, torch.Tensor([0.0]))
    assert torch.isclose(torch.log2(last_layer.input_quant.scale()) % 1, torch.Tensor([0.0]))
    assert torch.isclose(torch.log2(last_layer.weight_quant.scale()) % 1, torch.Tensor([0.0]))

    # Assert quantization bit widths are as desired
    # Biases
    assert first_conv_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers[
        'value'].item() == bias_bit_width
    assert last_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers[
        'value'].item() == bias_bit_width
    # Weights
    assert first_conv_layer.weight_quant.bit_width().item() == weight_bit_width
    assert last_layer.weight_quant.bit_width().item() == weight_bit_width
    # Activation
    assert first_conv_layer.input_quant.bit_width().item() == act_bit_width
