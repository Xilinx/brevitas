# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging import version
import pytest
import pytest_cases
from pytest_cases import fixture_union
import torch
import torch.nn as nn
from torchvision import models

from brevitas import torch_version
from brevitas.graph.equalize import _cross_layer_equalization
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3Weight
from brevitas.quant.mx_quant_ocp import MXInt8Act
from brevitas.quant.mx_quant_ocp import MXInt8Weight
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat

SEED = 123456
ATOL_DICT = {
    torch.float32: 1e-3,
    torch.float16: 5e-2,
    torch.bfloat16: 3e-1,}
ATOL = 1e-3
IN_FEATURES = 24

MODELS = {
    'vit_b_32': [0.396, 0.657],
    'shufflenet_v2_x0_5': [0.318, 0.649],
    'mobilenet_v2': [0.161, 0.320],
    'resnet18': [0.487, 0.952],
    'googlenet': [0.495, 0.982],
    'inception_v3': [0.497, 0.989],
    'alexnet': [0.75, 0.75],}

IN_SIZE_CONV = (1, 3, 224, 224)
IN_SIZE_LINEAR = (1, 224, 3)
IN_SIZE_CONV_SMALL = (1, 3, 32, 32)


def equalize_test(model, regions, merge_bias, bias_shrinkage, scale_computation_type):
    scale_factors_regions = []
    for i in range(3):
        for region in regions:
            scale_factors_region, _ = _cross_layer_equalization(
                model,
                region,
                merge_bias=merge_bias,
                bias_shrinkage=bias_shrinkage,
                scale_computation_type=scale_computation_type)
            if i == 0:
                scale_factors_regions.append(scale_factors_region)
    return scale_factors_regions


@pytest_cases.fixture
@pytest_cases.parametrize(
    "model_dict", [(model_name, coverage) for model_name, coverage in MODELS.items()],
    ids=[model_name for model_name, _ in MODELS.items()])
def model_coverage(model_dict: dict):
    model_name, coverage = model_dict

    kwargs = dict()
    if model_name in ('inception_v3', 'googlenet'):
        kwargs['transform_input'] = False
    model = getattr(models, model_name)(pretrained=True, **kwargs)

    return model, coverage


@pytest_cases.fixture
def bnconv_model():

    class BNConvModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.bn = nn.BatchNorm2d(3)
            # Simulate statistics gathering
            self.bn.running_mean.data = torch.randn_like(self.bn.running_mean)
            self.bn.running_var.data = torch.abs(torch.randn_like(self.bn.running_var))
            # Simulate learned parameters
            self.bn.weight.data = torch.randn_like(self.bn.weight)
            self.bn.bias.data = torch.randn_like(self.bn.bias)
            self.conv = nn.Conv2d(3, 16, kernel_size=3)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv(x)
            return x

    return BNConvModel


@pytest_cases.fixture
@pytest_cases.parametrize('bias', [True, False])
@pytest_cases.parametrize('add_bias_kv', [True, False])
@pytest_cases.parametrize('batch_first', [True, False])
def linearmha_model(bias, add_bias_kv, batch_first):
    # Skip due to following issue https://github.com/pytorch/pytorch/issues/97128
    if torch_version == version.parse('2.0.1') and not bias and batch_first and not add_bias_kv:
        pytest.skip(f"Skip due to a regression in pytorch 2.0.1")

    class LinearMhaModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 24)
            self.mha = nn.MultiheadAttention(
                24, 3, 0.1, bias=bias, add_bias_kv=add_bias_kv, batch_first=batch_first)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            x, _ = self.mha(x, x, x)
            return x

    return LinearMhaModel


@pytest_cases.fixture
@pytest_cases.parametrize('bias', [True, False])
@pytest_cases.parametrize('add_bias_kv', [True, False])
@pytest_cases.parametrize('batch_first', [True, False])
def layernormmha_model(bias, add_bias_kv, batch_first):
    # Skip due to following issue https://github.com/pytorch/pytorch/issues/97128
    if torch_version == version.parse('2.0.1') and not bias and batch_first and not add_bias_kv:
        pytest.skip(f"Skip due to a regression in pytorch 2.0.1")

    class LayerNormMhaModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.layernorm = nn.LayerNorm(3)
            # Simulate learned parameters
            self.layernorm.weight.data = torch.randn_like(self.layernorm.weight)
            self.layernorm.bias.data = torch.randn_like(self.layernorm.bias)
            self.mha = nn.MultiheadAttention(
                3, 3, 0.1, bias=bias, add_bias_kv=add_bias_kv, batch_first=batch_first)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.layernorm(x)
            x = self.relu(x)
            x, _ = self.mha(x, x, x)
            return x

    return LayerNormMhaModel


@pytest_cases.fixture
@pytest_cases.parametrize('bias', [True, False])
@pytest_cases.parametrize('add_bias_kv', [True, False])
@pytest_cases.parametrize('batch_first', [True, False])
def mhalinear_model(bias, add_bias_kv, batch_first):
    # Skip due to following issue https://github.com/pytorch/pytorch/issues/97128
    if torch_version == version.parse('2.0.1') and not bias and batch_first and not add_bias_kv:
        pytest.skip(f"Skip due to a regression in pytorch 2.0.1")

    class MhaLinearModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.mha = nn.MultiheadAttention(
                3, 1, 0.1, bias=bias, add_bias_kv=add_bias_kv, batch_first=batch_first)
            self.linear = nn.Linear(3, 6)
            self.relu = nn.ReLU()

        def forward(self, x):
            x, _ = self.mha(x, x, x)
            x = self.relu(x)
            x = self.linear(x)
            return x

    return MhaLinearModel


@pytest_cases.fixture
def convdepthconv_model():

    class ConvDepthConvModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3)
            self.conv_0 = nn.Conv2d(16, 16, kernel_size=1, groups=16)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.conv_0(x)
            return x

    return ConvDepthConvModel


@pytest_cases.fixture
def convbn_model():

    class ConvBNModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 128, kernel_size=3)
            self.bn = nn.BatchNorm2d(128)
            # Simulate statistics gathering
            self.bn.running_mean.data = torch.randn_like(self.bn.running_mean)
            self.bn.running_var.data = torch.abs(torch.randn_like(self.bn.running_var))
            # Simulate learned parameters
            self.bn.weight.data = torch.randn_like(self.bn.weight)
            self.bn.bias.data = torch.randn_like(self.bn.bias)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.bn(x)
            return x

    return ConvBNModel


@pytest_cases.fixture
def residual_model():

    class ResidualModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=1)
            self.conv_0 = nn.Conv2d(16, 3, kernel_size=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            start = x
            x = self.conv(x)
            x = self.relu(x)
            x = self.conv_0(x)
            x = start + x
            return x

    return ResidualModel


@pytest_cases.fixture
def srcsinkconflict_model():
    """
    In this example, conv_0 is both a src and sink.
    """

    class ResidualSrcsAndSinkModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv_start = nn.Conv2d(3, 3, kernel_size=1)
            self.conv = nn.Conv2d(3, 3, kernel_size=1)
            self.conv_0 = nn.Conv2d(3, 3, kernel_size=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            start = self.conv_start(x)
            x = self.conv_0(start)
            x = start + x
            x = self.relu(x)
            x = self.conv(x)
            return x

    return ResidualSrcsAndSinkModel


@pytest_cases.fixture
def mul_model():

    class ResidualSrcsAndSinkModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv_1 = nn.Conv2d(3, 3, kernel_size=1)
            self.conv_0 = nn.Conv2d(3, 3, kernel_size=1)
            self.conv_end = nn.Conv2d(3, 3, kernel_size=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x_0 = self.conv_0(x)
            x_1 = self.conv_1(x)
            x = x_0 * x_1
            x = self.relu(x)
            x = self.conv_end(x)
            return x

    return ResidualSrcsAndSinkModel


@pytest_cases.fixture
def convgroupconv_model():

    class ConvGroupConvModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 32, kernel_size=3)
            self.conv_0 = nn.Conv2d(32, 32, kernel_size=1, groups=2)
            self.conv_1 = nn.Conv2d(32, 32, kernel_size=1, groups=1)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.conv_0(x)
            x = self.relu(x)
            x = self.conv_1(x)
            return x

    return ConvGroupConvModel


@pytest_cases.fixture
def convtranspose_model():

    class ConvTransposeModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.relu = nn.ReLU()
            self.conv_0 = nn.ConvTranspose2d(in_channels=3, out_channels=8, kernel_size=3)
            self.conv_1 = nn.ConvTranspose2d(in_channels=8, out_channels=32, kernel_size=3)

        def forward(self, x):
            x = self.conv_0(x)
            x = self.relu(x)
            x = self.conv_1(x)
            return x

    return ConvTransposeModel


list_of_fixtures = [
    'residual_model',
    'srcsinkconflict_model',
    'mul_model',
    'bnconv_model',
    'convdepthconv_model',
    'linearmha_model',
    'mhalinear_model',
    'layernormmha_model',
    'convgroupconv_model',
    'convtranspose_model']

toy_model = fixture_union('toy_model', list_of_fixtures, ids=list_of_fixtures)

RESNET_18_REGIONS = [
    [('layer3.0.bn1',), ('layer3.0.conv2',)],
    [('layer4.1.bn1',), ('layer4.1.conv2',)],
    [('layer2.1.bn1',), ('layer2.1.conv2',)],
    [('layer3.1.bn1',), ('layer3.1.conv2',)],
    [('layer1.0.bn1',), ('layer1.0.conv2',)],
    [('layer3.0.bn2', 'layer3.0.downsample.1', 'layer3.1.bn2'),
     ('layer3.1.conv1', 'layer4.0.conv1', 'layer4.0.downsample.0')],
    [('layer4.0.bn1',), ('layer4.0.conv2',)],
    [('layer2.0.bn2', 'layer2.0.downsample.1', 'layer2.1.bn2'),
     ('layer2.1.conv1', 'layer3.0.conv1', 'layer3.0.downsample.0')],
    [('layer1.1.bn1',), ('layer1.1.conv2',)],
    [('bn1', 'layer1.0.bn2', 'layer1.1.bn2'),
     ('layer1.0.conv1', 'layer1.1.conv1', 'layer2.0.conv1', 'layer2.0.downsample.0')],
    [('layer2.0.bn1',), ('layer2.0.conv2',)],
    [('layer4.0.bn2', 'layer4.0.downsample.1', 'layer4.1.bn2'), ('fc', 'layer4.1.conv1')],]


def _set_weight_quant_to_param(weight_quant):
    # Some quantizers default to scaling_impl_type=stats, which recomputes the scale
    # from the current weights on each forward pass. GPxQ updates weights greedily, so
    # the scale would shift with every update. Forcing parameter_from_stats fixes the
    # scale as a stored parameter initialized once from the initial weights.
    weight_quant = weight_quant.let(scaling_impl_type='parameter_from_stats')
    return weight_quant


list_of_input_weight_quant_tuples = [
    (None, _set_weight_quant_to_param(Int8WeightPerTensorFloat)),
    (Int8ActPerTensorFloat, _set_weight_quant_to_param(Int8WeightPerTensorFloat)),
    (MXInt8Act, _set_weight_quant_to_param(MXInt8Weight)), (MXFloat8e4m3Act, MXFloat8e4m3Weight)]


input_quant, weight_quant = pytest_cases.param_fixtures("input_quant, weight_quant", list_of_input_weight_quant_tuples)


@pytest_cases.fixture
def quant_conv_with_input_quant_model(input_quant, weight_quant):

    class QuantConvModel(nn.Module):
        input_size = IN_SIZE_CONV_SMALL[1:]

        def __init__(self) -> None:
            super().__init__()
            self.conv_0 = qnn.QuantConv2d(
                self.input_size[0],
                16,
                kernel_size=3,
                input_quant=input_quant,
                weight_quant=weight_quant)
            self.conv_1 = qnn.QuantConv2d(
                16, 32, kernel_size=3, input_quant=input_quant, weight_quant=weight_quant)

        def forward(self, x):
            x = self.conv_0(x)
            x = torch.relu(x)
            x = self.conv_1(x)
            return x

    return QuantConvModel


@pytest_cases.fixture
def quant_convdepthconv_model(input_quant, weight_quant):

    class QuantConvDepthConvModel(nn.Module):
        input_size = IN_SIZE_CONV_SMALL[1:]

        def __init__(self) -> None:
            super().__init__()
            self.conv = qnn.QuantConv2d(
                self.input_size[0],
                16,
                kernel_size=3,
                input_quant=input_quant,
                weight_quant=weight_quant)
            self.conv_0 = qnn.QuantConv2d(
                16,
                16,
                kernel_size=1,
                groups=16,
                input_quant=input_quant,
                weight_quant=weight_quant)
            self.relu = qnn.QuantReLU(return_quant_tensor=input_quant != None)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = self.conv_0(x)
            return x

    return QuantConvDepthConvModel


@pytest_cases.fixture
def quant_residual_model(input_quant, weight_quant):

    class QuantResidualModel(nn.Module):
        input_size = IN_SIZE_CONV_SMALL[1:]

        def __init__(self) -> None:
            super().__init__()
            in_channels = self.input_size[0]
            self.conv = qnn.QuantConv2d(
                in_channels, 16, kernel_size=1, input_quant=input_quant, weight_quant=weight_quant)
            self.conv_0 = qnn.QuantConv2d(
                16, in_channels, kernel_size=1, input_quant=input_quant, weight_quant=weight_quant)
            self.relu = qnn.QuantReLU(return_quant_tensor=input_quant != None)

        def forward(self, x):
            start = x
            x = self.conv(x)
            x = self.relu(x)
            x = self.conv_0(x)
            x = start + x

            return x

    return QuantResidualModel


@pytest_cases.fixture
def quant_convtranspose_model(input_quant, weight_quant):

    class QuantConvTransposeModel(nn.Module):
        input_size = IN_SIZE_CONV_SMALL[1:]

        def __init__(self) -> None:
            super().__init__()
            self.relu = qnn.QuantReLU(return_quant_tensor=input_quant != None)
            self.conv_0 = qnn.QuantConvTranspose2d(
                in_channels=self.input_size[0],
                out_channels=8,
                kernel_size=3,
                input_quant=input_quant,
                weight_quant=weight_quant)
            self.conv_1 = qnn.QuantConvTranspose2d(
                in_channels=8,
                out_channels=32,
                kernel_size=3,
                input_quant=input_quant,
                weight_quant=weight_quant)

        def forward(self, x):
            x = self.conv_0(x)
            x = self.relu(x)
            x = self.conv_1(x)
            return x

    return QuantConvTransposeModel


@pytest_cases.fixture
def quant_linear_model(input_quant, weight_quant):

    class QuantLinearModel(nn.Module):
        input_size = IN_SIZE_LINEAR[1:]

        def __init__(self) -> None:
            super().__init__()
            self.linear_0 = qnn.QuantLinear(
                in_features=self.input_size[-1],
                out_features=16,
                input_quant=input_quant,
                weight_quant=weight_quant)
            self.relu = qnn.QuantReLU(return_quant_tensor=input_quant != None)
            self.linear_1 = qnn.QuantLinear(
                in_features=16, out_features=32, input_quant=input_quant, weight_quant=weight_quant)

        def forward(self, x):
            x = self.linear_0(x)
            x = self.relu(x)
            x = self.linear_1(x)
            return x

    return QuantLinearModel


list_of_quant_fixtures = [
    'quant_conv_with_input_quant_model',
    'quant_convdepthconv_model',
    'quant_residual_model',
    'quant_convtranspose_model',
    'quant_linear_model']

toy_quant_model = fixture_union(
    'toy_quant_model', list_of_quant_fixtures, ids=list_of_quant_fixtures)

# Multihead-attention fixtures for exercising batch-dim detection during equalization/GPxQ.
MHA_EMBED_DIM = 24
MHA_NUM_HEADS = 4
# Sequence length and batch size are deliberately different so that a wrong batch_dim
# would lead to a detectable difference (shape or values).
MHA_SEQ_LEN = 8
MHA_BATCH_SIZE = 5


def mha_input(batch_first, batch_size=MHA_BATCH_SIZE):
    # Returns a self-attention input laid out according to batch_first:
    #   batch_first=True  -> (N, L, E)
    #   batch_first=False -> (L, N, E)
    if batch_first:
        shape = (batch_size, MHA_SEQ_LEN, MHA_EMBED_DIM)
    else:
        shape = (MHA_SEQ_LEN, batch_size, MHA_EMBED_DIM)
    return torch.randn(shape)


@pytest_cases.fixture
@pytest_cases.parametrize('batch_first', [True, False])
def vanilla_mha_model(batch_first):
    # A bare torch.nn.MultiheadAttention. This is the realistic target for activation
    # equalization, which runs on the float model before quantization operators are inserted.

    class VanillaMHAModel(nn.Module):
        batch_first = False

        def __init__(self) -> None:
            super().__init__()
            self.mha = nn.MultiheadAttention(
                MHA_EMBED_DIM, MHA_NUM_HEADS, batch_first=self.batch_first)

        def forward(self, x):
            out, _ = self.mha(x, x, x)
            return out

    VanillaMHAModel.batch_first = batch_first
    return VanillaMHAModel


@pytest_cases.fixture
@pytest_cases.parametrize('batch_first', [True, False])
def vanilla_linear_mha_model(batch_first):
    # Linear -> ReLU -> MHA. Provides a source (the Linear) so that graph-mode activation
    # equalization forms a region around the MHA sink.

    class VanillaLinearMHAModel(nn.Module):
        batch_first = False

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(MHA_EMBED_DIM, MHA_EMBED_DIM)
            self.relu = nn.ReLU()
            self.mha = nn.MultiheadAttention(
                MHA_EMBED_DIM, MHA_NUM_HEADS, batch_first=self.batch_first)

        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            out, _ = self.mha(x, x, x)
            return out

    VanillaLinearMHAModel.batch_first = batch_first
    return VanillaLinearMHAModel


@pytest_cases.fixture
@pytest_cases.parametrize('batch_first', [True, False])
@pytest_cases.parametrize('packed_in_proj', [True, False])
def quant_mha_gpxq_model(batch_first, packed_in_proj):
    # QuantMultiheadAttention with weight quantization enabled on the internal projections, so
    # that GPxQ (which runs after quantization) collects and optimizes them. GPxQ hooks the
    # internal projection QuantLinear layers, which always operate on (L, N, E) tensors.
    weight_quant = _set_weight_quant_to_param(Int8WeightPerTensorFloat)

    class QuantMHAGPxQModel(nn.Module):
        batch_first = False
        packed_in_proj = True

        def __init__(self) -> None:
            super().__init__()
            self.mha = qnn.QuantMultiheadAttention(
                embed_dim=MHA_EMBED_DIM,
                num_heads=MHA_NUM_HEADS,
                batch_first=self.batch_first,
                packed_in_proj=self.packed_in_proj,
                in_proj_input_quant=None,
                in_proj_weight_quant=weight_quant,
                in_proj_bias_quant=None,
                out_proj_input_quant=None,
                out_proj_weight_quant=weight_quant,
                out_proj_bias_quant=None,
                out_proj_output_quant=None)

        def forward(self, x):
            out, _ = self.mha(x, x, x)
            return out

    QuantMHAGPxQModel.batch_first = batch_first
    QuantMHAGPxQModel.packed_in_proj = packed_in_proj
    return QuantMHAGPxQModel
