from copy import deepcopy

import pytest
import torch
import torch.nn as nn

import brevitas
from brevitas.core.function_wrapper.shape import OverOutputChannelView
from brevitas.core.function_wrapper.shape import OverTensorView
from brevitas.core.stats.stats_op import MSE
from brevitas.graph.calibrate import calibration_mode
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantLinear
from brevitas.nn import QuantReLU
from brevitas.quant_tensor import QuantTensor
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model
from tests.marker import jit_disabled_for_local_loss
from tests.marker import jit_disabled_for_mock

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

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(10, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # downsample from IMAGE_DIM to half that
                nn.Flatten(),
                nn.Linear(32 * int(IMAGE_DIM / 2) ** 2, 1000))

        def forward(self, x):
            return self.layers(x)

    return Model()


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
    fx_model = brevitas.fx.symbolic_trace(simple_model)
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

    # Assert we can feed data of the correct size through the model
    quant_model(torch.rand(1, 10, IMAGE_DIM, IMAGE_DIM))

    # Get first/last layer for testing its quantization.
    first_conv_layer = quant_model.layers.get_submodule('0')
    first_relu_layer = quant_model.layers.get_submodule('1')
    last_layer = quant_model.layers.get_submodule('6')
    last_layer_output = quant_model.get_submodule('layers_6_output_quant')

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
    assert isinstance(quant_model.layers.get_submodule('3'), QuantReLU)

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

    fx_model = brevitas.fx.symbolic_trace(simple_model)
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
    first_conv_layer = quant_model.layers.get_submodule('0')
    first_relu_layer = quant_model.layers.get_submodule('1')
    last_layer = quant_model.layers.get_submodule('6')
    last_layer_output = quant_model.get_submodule('layers_6_output_quant')

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

    fx_model = brevitas.fx.symbolic_trace(simple_model)
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
    first_conv_layer = quant_model.layers.get_submodule('0')
    last_layer = quant_model.layers.get_submodule('6')
    last_layer_output = quant_model.get_submodule('layers_6_output_quant')

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


@pytest.mark.parametrize("weight_bit_width", [2, 8, 16])
@pytest.mark.parametrize("act_bit_width", [2, 5, 8])
@pytest.mark.parametrize("bias_bit_width", [16, 32, 0])
def test_fx_param_method_stats(simple_model, weight_bit_width, bias_bit_width, act_bit_width):
    """
    We test fx quantization, with the weight and activation quantization `stats` parameter methods.
    `stats` is the default setting, but we also test it explicitly in case it ever changes from the
    default.

    We test:
    - The FX-graph, quantized model is a GraphModule.
    - We can feed data through the model.
    - That the weight, bias and input/output quantization is toggled as expected.
    - That setting `None` for the `bias_bit_width` returns a dequantized bias.
    - That the bit widths are as desired.
    """
    fx_model = brevitas.fx.symbolic_trace(simple_model)
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
        weight_param_method='stats',
        act_param_method='stats',
    )

    # Assert we can feed data of the correct size through the model
    quant_model(torch.rand(1, 10, IMAGE_DIM, IMAGE_DIM))

    # Get first/last layer for testing its quantization.
    first_conv_layer = quant_model.layers.get_submodule('0')
    first_relu_layer = quant_model.layers.get_submodule('1')
    last_layer = quant_model.layers.get_submodule('6')
    last_layer_output = quant_model.get_submodule('layers_6_output_quant')

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
    assert isinstance(quant_model.layers.get_submodule('3'), QuantReLU)

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

    fx_model = brevitas.fx.symbolic_trace(simple_model)
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
        quant_format='int',
    )

    # Assert we can feed data of the correct size through the model
    quant_model(torch.rand(1, 10, IMAGE_DIM, IMAGE_DIM))

    # Get first/last layer for testing its quantization.
    first_conv_layer = quant_model.layers.get_submodule('0')
    last_layer = quant_model.layers.get_submodule('6')
    last_layer_output = quant_model.get_submodule('layers_6_output_quant')

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
    fx_model = brevitas.fx.symbolic_trace(simple_model)
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
@pytest.mark.parametrize("act_quant_percentile", [1, 50, 99.9])
def test_layerwise_percentile_for_calibration(simple_model, act_quant_percentile):
    """
    We test different values for the percentile used for the activations' statistics computation
    during calibration.
    We test if the percentile correcrly produces the desired qparams for a `QuantIdentity`
    when fed a tensor with linearly scaled values between 0 and 1.

    We test:
    - We can feed data through the model.
    - The desired qparams manifest under controlled conditions as a result of calibration
    percentiles.
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
        act_quant_percentile=act_quant_percentile,
        act_quant_type='asym',
        scale_factor_type='float_scale',
        quant_format='int',
    )

    # Assert we can feed data of the correct size through the model
    # We are also performing calibration
    quant_model.train()
    # We create an input with values linearly scaled between 0 and 1.
    input = torch.arange(0, 1, step=1 / (10 * IMAGE_DIM ** 2))
    input = input.view(1, 10, IMAGE_DIM, IMAGE_DIM).float()
    with torch.no_grad():
        with calibration_mode(quant_model):
            for _ in range(1000):
                quant_model(input)
    quant_model.eval()

    # Get first/last layer for testing its quantization.
    first_conv_layer = quant_model.layers.get_submodule('0')

    # We check the calibration. We do so by ensuring that the quantization range is within a few quantization
    # bin tolerance of the `act_quant_percentile`. This should be the case, given that we are doing
    # affine quantization with a strictly positive tensor (and so zero_point should be 0), and because
    # we are doing the calibration with a tensor with all values linearly increasinging from 0 to 1.
    assert torch.isclose(first_conv_layer.input_quant.zero_point(), torch.Tensor([0.]))
    tolerance = 8  # quantization bins of tolerance, on the "plus side".
    ideal_range = act_quant_percentile / 100
    scale = first_conv_layer.input_quant.scale()

    # The quantization range is always smaller than the data covered up to the percentile, because
    # of how the percnetile->qrange calculation happens.
    assert ideal_range > scale * 255
    # We make sure the quantization range is still reasonably close to covering the entire data, up
    # to the provided percentile.
    assert ideal_range < scale * (255 + tolerance)


@pytest.mark.parametrize("quant_granularity", ["per_tensor", "per_channel"])
@jit_disabled_for_local_loss()
def test_layerwise_param_method_mse(simple_model, quant_granularity):
    """
    We test layerwise quantization, with the weight and activation quantization `mse` parameter
    methods.

    We test:
    - We can feed data through the model.
    - That the stat observer is explictly MSE.
    - That the view on the quantization granularity is as desired.
    - That during calibration, the qparams are derived by finding values that minimize the MSE
    between the floating point and quantized tensor.
    """
    weight_bit_width = 8
    act_bit_width = 8
    bias_bit_width = 32
    quant_model = quantize_model(
        model=simple_model,
        backend='layerwise',
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        bias_bit_width=bias_bit_width if bias_bit_width > 0 else None,
        weight_quant_granularity=quant_granularity,
        act_quant_type='asym',
        act_quant_percentile=99.9,  # Unused
        scale_factor_type='float_scale',
        quant_format='int',
        weight_param_method='mse',
        act_param_method='mse',
    )

    # Assert we can feed data of the correct size through the model
    # We are also performing calibration
    quant_model.train()
    # We create an input with values linearly scaled between 0 and 1.
    input = torch.arange(0, 1, step=1 / (10 * IMAGE_DIM ** 2))
    input = input.view(1, 10, IMAGE_DIM, IMAGE_DIM).float()
    with torch.no_grad():
        with calibration_mode(quant_model):
            quant_model(input)
    quant_model.eval()

    # Get first/last layer for testing its quantization.
    first_conv_layer = quant_model.layers.get_submodule('0')
    last_layer = quant_model.layers.get_submodule('6')

    # Check that the quant param method module is MSE as it should be
    # Weights
    first_weight_param_mod = first_conv_layer.weight_quant.tensor_quant.scaling_impl.parameter_list_stats.stats.stats_impl
    last_weight_param_mod = last_layer.weight_quant.tensor_quant.scaling_impl.parameter_list_stats.stats.stats_impl
    assert isinstance(first_weight_param_mod, MSE)
    assert isinstance(last_weight_param_mod, MSE)

    # Check observation is over tensor or channel as desired
    def check_dim_of_observation(module: torch.nn.Module, quant_granularity: str):
        if quant_granularity == 'per_tensor':
            assert isinstance(module.input_view_shape_impl, OverTensorView)
        elif quant_granularity == 'per_channel':
            assert isinstance(module.input_view_shape_impl, OverOutputChannelView)

    # Weight
    check_dim_of_observation(first_weight_param_mod, quant_granularity)
    check_dim_of_observation(last_weight_param_mod, quant_granularity)

    # We test the calibrated qparams. We fed in a tensor with linearly scaled values between 0 and 1.
    # We check that varying the qparams gives worse or equal MSE than the calibrated qparams.
    # We assume a convex problem.
    scale = first_conv_layer.input_quant.scale()
    zero_point = first_conv_layer.input_quant.zero_point()

    def get_qmse(
            scale: torch.Tensor, zero_point: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        quant_tensor = scale * (
            torch.clamp(torch.round(input / scale + zero_point), 0, 255) - zero_point)
        mse = torch.mean((quant_tensor - input) ** 2)
        return mse

    orig_mse = get_qmse(scale, zero_point, input)
    for scale_diff in [0.1 * scale, 0, -0.1 * scale]:
        for zero_diff in [1, 0, -1]:
            diff_mse = get_qmse(scale + scale_diff, zero_point + zero_diff, input)
            assert torch.isclose(diff_mse, orig_mse) or (diff_mse > orig_mse)


@pytest.mark.parametrize("quant_granularity", ["per_tensor", "per_channel"])
@jit_disabled_for_local_loss()
def test_layerwise_stats_vs_mse(simple_model, quant_granularity):
    """
    We test layerwise quantization, with the weight and activation quantization `mse` parameter
    methods.

    We test:
    - Recostruction error of MSE should be smaller or equal to stats
    """
    weight_bit_width = 8
    act_bit_width = 8
    bias_bit_width = 32
    quant_model_mse = quantize_model(
        model=deepcopy(simple_model),
        backend='layerwise',
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        bias_bit_width=bias_bit_width if bias_bit_width > 0 else None,
        weight_quant_granularity=quant_granularity,
        act_quant_type='asym',
        act_quant_percentile=99.9,  # Unused
        scale_factor_type='float_scale',
        quant_format='int',
        weight_param_method='mse',
        act_param_method='mse')

    quant_model_stats = quantize_model(
        model=deepcopy(simple_model),
        backend='layerwise',
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        bias_bit_width=bias_bit_width if bias_bit_width > 0 else None,
        weight_quant_granularity=quant_granularity,
        act_quant_type='asym',
        act_quant_percentile=99.9,  # Unused
        scale_factor_type='float_scale',
        quant_format='int',
        weight_param_method='stats',
        act_param_method='mse')

    # We create an input with values linearly scaled between 0 and 1.
    input = torch.arange(0, 1, step=1 / (10 * IMAGE_DIM ** 2))
    input = input.view(1, 10, IMAGE_DIM, IMAGE_DIM).float()
    with torch.no_grad():
        with calibration_mode(quant_model_mse):
            quant_model_mse(input)
    quant_model_mse.eval()
    with torch.no_grad():
        with calibration_mode(quant_model_stats):
            quant_model_stats(input)
    quant_model_stats.eval()
    weight = simple_model.layers.get_submodule('0').weight
    first_conv_layer_mse = quant_model_mse.layers.get_submodule('0')
    first_conv_layer_stats = quant_model_stats.layers.get_submodule('0')

    l2_stats = ((weight - first_conv_layer_stats.quant_weight().value) ** 2).sum()
    l2_mse = ((weight - first_conv_layer_mse.quant_weight().value) ** 2).sum()

    # Recostruction error of MSE should be smaller or equal to stats
    assert l2_mse - l2_stats <= torch.tensor(1e-5)


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
    assert isinstance(quant_model.layers, nn.Sequential)

    # Make sure we can feed data through the model
    _ = quant_model(torch.rand(1, 10, IMAGE_DIM, IMAGE_DIM))

    # Get first layer for testing its quantization.
    # We also test we can feed data through the first layer in isolation
    first_layer = quant_model.layers.get_submodule('0')
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
    first_conv_layer = quant_model.layers.get_submodule('0')
    last_layer = quant_model.layers.get_submodule('6')

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
