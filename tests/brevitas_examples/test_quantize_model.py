import pytest
from copy import deepcopy
import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model
from brevitas.graph.quantize import preprocess_for_quantize

import ipdb


# CONSTANTS
IMAGE_DIM = 16

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
        nn.MaxPool2d(2), # downsample from IMAGE_DIM to half that
        nn.Flatten(),
        nn.Linear(32 * int(IMAGE_DIM/2) ** 2, 1000)
    )

##############
# Unit tests #
##############
@pytest.mark.parametrize("weight_bit_width", [2, 5, 8, 16])
@pytest.mark.parametrize("act_bit_width", [2, 5, 8])
@pytest.mark.parametrize("bias_bit_width", [16, 32])
@pytest.mark.parametrize("layerwise_first_last_bit_width", [2, 8])
def test_layerwise_10_in_channels_quantize_model(simple_model, weight_bit_width, bias_bit_width, act_bit_width, layerwise_first_last_bit_width):
    """
    We use a model with  input channels, and test `layerwise` quantization.
    Because of the 10 input channels, this will ignore the `layerwise_first_last_bit_width`
    value, and will quantize evrything according to:
    - weight_bit_width
    - bias_bit_width
    - act_bit_width
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
        #layerwise_first_last_mantissa_bit_width=10,
        #layerwise_first_last_exponent_bit_width=10,
        #weight_mantissa_bit_width=10,
        #weight_exponent_bit_width=10,
        #act_mantissa_bit_width=layerwise_first_last_bit_width,
        #act_exponent_bit_width=layerwise_first_last_bit_width,
    )
    assert isinstance(quant_model, nn.Sequential)

    # Make sure we can feed data through the model
    _ = quant_model(torch.rand(1,10,IMAGE_DIM, IMAGE_DIM))
    
    # Get first layer for testing its quantization.
    # We also test we can feed data through the first layer in isolation
    first_layer = quant_model.get_submodule('0')
    first_layer_output = first_layer(torch.rand(1,10,IMAGE_DIM, IMAGE_DIM))

    # Assert only weight is quantized by default
    # However, here input and bias are also quantized
    assert first_layer.is_weight_quant_enabled
    assert first_layer.is_bias_quant_enabled
    assert first_layer.is_input_quant_enabled # unlike with the fx backend, the input quantization is enabled. 
    assert not first_layer.is_output_quant_enabled
    # NOTE: The `layerwise` backend also differs from the `fx` backend in that: the input quantization is enabled
    # for the first Conv layer by default in the `layerwise`, whereas it is disabled in the `fx` backend. However, 
    # in practice this is because the `fx` backend introduces an extra quantization module (i.e. QuantStub) before
    # the first layer that quantizes the input to the first layer, so it comes to the same: in both cases, the Conv
    # receives a quantized input tensor.

    # Assert quantization bit widths are as desired
    # Biases
    assert first_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers['value'].item() == bias_bit_width
    # Weights
    assert first_layer.weight_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers['value'].item() == weight_bit_width
    # Activations
    assert first_layer.input_quant.fused_activation_quant_proxy.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers['value'].item() == act_bit_width


@pytest.mark.parametrize("weight_bit_width", [2, 5, 8, 16])
@pytest.mark.parametrize("act_bit_width", [2, 5, 8])
@pytest.mark.parametrize("bias_bit_width", [16, 32])
@pytest.mark.parametrize("layerwise_first_last_bit_width", [2, 8])
def test_layerwise_3_in_channels_quantize_model(minimal_model, weight_bit_width, bias_bit_width, act_bit_width, layerwise_first_last_bit_width):

    """
    We use a model with 3 input channels, and test `layerwise` quantization.
    Because of the 3 input channels, this will set the quantization bit width
    of the weights and activations of the first layer to be equal to 
    `layerwise_first_last_bit_width`.
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
    assert first_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers['value'].item() == bias_bit_width
    # Weights
    assert first_layer.weight_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers['value'].item() == layerwise_first_last_bit_width
    # Activations
    assert first_layer.input_quant.fused_activation_quant_proxy.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers['value'].item() == layerwise_first_last_bit_width


@pytest.mark.parametrize("weight_bit_width", [2, 5, 8, 16])
@pytest.mark.parametrize("act_bit_width", [2, 5, 8])
@pytest.mark.parametrize("bias_bit_width", [16, 32])
def test_fx_model(simple_model, weight_bit_width, bias_bit_width, act_bit_width):
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


    # Check quantization is toggled as expected
    assert first_conv_layer.is_weight_quant_enabled
    assert first_conv_layer.is_bias_quant_enabled
    assert not first_conv_layer.is_input_quant_enabled # unlike with the layerwise backend, the input quantization is disabled.
    assert not first_conv_layer.is_output_quant_enabled

    assert not first_relu_layer.is_input_quant_enabled 
    assert first_relu_layer.is_output_quant_enabled # the output of the "fused" ConvReLU is quantized

    # Assert types are as expected
    assert isinstance(quant_model.get_submodule('3'), QuantReLU)

    # Assert quantization bit widths are as desired
    # Biases
    assert first_conv_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers['value'].item() == bias_bit_width
    assert last_layer.bias_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers['value'].item() == bias_bit_width
    # Weights
    assert first_conv_layer.weight_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers['value'].item() == weight_bit_width
    assert last_layer.weight_quant.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers['value'].item() == weight_bit_width
    # Activations
    assert first_relu_layer.act_quant.fused_activation_quant_proxy.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers['value'].item() == act_bit_width
    assert last_layer_output.act_quant.fused_activation_quant_proxy.tensor_quant.msb_clamp_bit_width_impl.bit_width._buffers['value'].item() == act_bit_width

def test_float_quantization(simple_model):
    quant_model = quantize_model(
        model=simple_model,
        backend='layerwise',
        weight_bit_width=8,
        act_bit_width=8,
        bias_bit_width=32,
        weight_quant_granularity='per_tensor',
        act_quant_percentile=99.9,
        act_quant_type='sym',
        scale_factor_type='float_scale',
        quant_format='float',
    )
    assert isinstance(quant_model, nn.Sequential)
    #assert all(isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) for m in quant_model.children())
    #TODO

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
    with pytest.raises(KeyError):
        quantize_model(
            model=fx_model,
            backend='fx',
            weight_bit_width=0, # NOTE: this is considered valid, which may be an issue 
            act_bit_width=0,
            bias_bit_width=32,
            weight_quant_granularity='per_tensor',
            act_quant_percentile=99.9,
            act_quant_type='sym',
            scale_factor_type='float_scale',  
            quant_format='int',
        )
    # Test that negative values are invalid for bit widths
    with pytest.raises(KeyError):
        quantize_model(
            model=fx_model,
            backend='fx',
            weight_bit_width=-1, # NOTE: this is considered valid, which may be an issue 
            act_bit_width=-1,
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

