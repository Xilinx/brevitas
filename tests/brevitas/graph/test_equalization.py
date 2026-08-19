# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import platform

from packaging.version import parse
import pytest
import pytest_cases
import torch
from torch import nn
from torchvision import models

from brevitas import torch_version
from brevitas.fx import symbolic_trace
from brevitas.graph.equalize import _batch_norm
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.equalize import _is_supported_module
from brevitas.graph.equalize import _supported_layers
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.equalize import GraphActivationEqualization
from brevitas.graph.equalize import LayerwiseActivationEqualization
from brevitas.graph.standardize import DuplicateSharedStatelessModule
from brevitas.graph.standardize import TorchFunctionalToModule
from brevitas.graph.utils import get_module

from .equalization_fixtures import *


def test_resnet18_equalization():
    model = models.resnet18(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)
    model.eval()
    model = symbolic_trace(model)
    expected_out = model(inp)

    model_orig = copy.deepcopy(model)
    supported_sinks = list(_supported_layers)
    supported_sinks = tuple([
        x for x in _supported_layers if x not in (torch.nn.LayerNorm, *_batch_norm)])
    regions = _extract_regions(model, state_impl_kwargs={'supported_sinks': supported_sinks})
    _ = equalize_test(
        model, regions, merge_bias=True, bias_shrinkage='vaiq', scale_computation_type='maxabs')
    out = model(inp)

    regions = sorted(regions, key=lambda region: sorted([r for r in region.srcs_names]))
    resnet_18_regions = sorted(RESNET_18_REGIONS, key=lambda region: region[0][0])
    equalized_layers = set()
    for r in resnet_18_regions:
        equalized_layers.update(r[0])
        equalized_layers.update(r[1])

    # Check that we found all the expected regions
    for region, expected_region in zip(regions, resnet_18_regions):
        srcs = region.srcs_names
        sources_check = set(srcs) == set(expected_region[0])
        sinks = region.sinks_names
        sinks_check = set(sinks) == set(expected_region[1])
        assert sources_check
        assert sinks_check

    # Check that all layers were equalized and weights changed
    for layer in equalized_layers:
        eq_module = get_module(model, layer)
        orig_module = get_module(model_orig, layer)
        assert not torch.allclose(eq_module.weight, orig_module.weight)

    # Check that equalization is not introducing FP variations
    assert torch.allclose(expected_out, out, atol=ATOL)


@pytest_cases.parametrize("merge_bias", [True, False])
def test_equalization_torchvision_models(model_coverage: tuple, merge_bias: bool):
    model, coverage = model_coverage

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)
    model.eval()
    model = symbolic_trace(model)
    model = TorchFunctionalToModule().apply(model)

    expected_out = model(inp)

    supported_sinks = list(_supported_layers)
    supported_sinks = tuple([
        x for x in _supported_layers if x not in (torch.nn.LayerNorm, *_batch_norm)])
    regions = _extract_regions(model, state_impl_kwargs={'supported_sinks': supported_sinks})
    scale_factor_regions = equalize_test(
        model,
        regions,
        merge_bias=merge_bias,
        bias_shrinkage='vaiq',
        scale_computation_type='maxabs')
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    out = model(inp)
    srcs = set()
    sinks = set()
    for r in regions:
        srcs.update([x for x in list(r.srcs_names)])
        sinks.update([x for x in list(r.sinks_names)])

    count_region_srcs = 0
    count_region_sinks = 0
    for n in model.graph.nodes:
        if _is_supported_module(model, n):
            count_region_srcs += 1
            if not isinstance(get_module(model, n.target), (nn.LayerNorm,) + _batch_norm):
                count_region_sinks += 1

    src_coverage = len(srcs) / count_region_srcs
    sink_coverage = len(sinks) / count_region_sinks
    assert src_coverage >= coverage[0]
    assert sink_coverage >= coverage[1]
    assert torch.allclose(expected_out, out, atol=ATOL)
    # Graph equalization can exit in case of shape mismatches or other error without performing any
    # equalization and returning a scalar value. We check that the equalized regions are as many as
    # expected
    assert all([shape != () for shape in shape_scale_regions])


@pytest_cases.parametrize("merge_bias", [True, False])
def test_models(toy_model, merge_bias, request):
    test_id = request.node.callspec.id

    if 'mha' in test_id:
        in_shape = IN_SIZE_LINEAR
    else:
        in_shape = IN_SIZE_CONV

    model_class = toy_model
    model = model_class()
    inp = torch.randn(in_shape)

    model.eval()
    with torch.no_grad():
        expected_out = model(inp)

    model = symbolic_trace(model)
    supported_sinks = list(_supported_layers)
    supported_sinks = tuple([
        x for x in _supported_layers if x not in (torch.nn.LayerNorm, *_batch_norm)])
    regions = _extract_regions(model, state_impl_kwargs={'supported_sinks': supported_sinks})
    scale_factor_regions = equalize_test(
        model,
        regions,
        merge_bias=merge_bias,
        bias_shrinkage='vaiq',
        scale_computation_type='maxabs')
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    with torch.no_grad():
        out = model(inp)
    assert len(regions) > 0
    assert torch.allclose(expected_out, out, atol=ATOL)
    # Check that at least one region performs "true" equalization
    # If all shapes are scalar, no equalization has been performed
    assert all([shape != () for shape in shape_scale_regions])


@pytest_cases.parametrize("layerwise", [False])
@pytest_cases.parametrize("fuse_scaling", [True, False])
@pytest_cases.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16],
    ids=lambda dtype: str(dtype).split(".")[-1])
@pytest_cases.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
    ids=lambda dtype: str(dtype).split(".")[-1])
def test_act_equalization_models(toy_model, layerwise, fuse_scaling, dtype, device, request):
    if dtype in [torch.float16, torch.bfloat16] and parse('2.3.0') > torch_version:
        pytest.skip(
            "Some operations are not implemented for float16/bfloat16 in PyTorch versions below 2.3.0"
        )
    if dtype in [torch.float16, torch.bfloat16
                ] and device == 'cpu' and platform.system() == 'Windows':
        pytest.skip("Windows CPU oneDNN backend cannot build bf16/fp16 matmul primitives")
    test_id = request.node.callspec.id

    if 'mha' in test_id:
        in_shape = IN_SIZE_LINEAR
    else:
        in_shape = IN_SIZE_CONV

    model_class = toy_model
    model = model_class()
    model.to(device=device, dtype=dtype)
    inp = torch.randn(in_shape, device=device, dtype=dtype)

    model.eval()
    expected_out = model(inp)
    model = symbolic_trace(model)
    with torch.no_grad():
        with activation_equalization_mode(model,
                                          0.5,
                                          True,
                                          layerwise=layerwise,
                                          fuse_scaling=fuse_scaling) as aem:
            regions = aem.graph_act_eq.regions
            model(inp)
    scale_factor_regions = aem.scale_factors
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    out = model(inp)
    assert torch.allclose(expected_out, out, atol=ATOL_DICT[dtype])

    assert len(regions) > 0
    # Check that at least one region performs "true" equalization
    # If all shapes are scalar, no equalization has been performed
    # Layerwise act eq for Groupwise conv is not supported
    if not ('convgroupconv' in test_id and layerwise):
        assert all([shape != () for shape in shape_scale_regions])


@pytest_cases.parametrize(
    "model_dict", [(model_name, coverage) for model_name, coverage in MODELS.items()],
    ids=[model_name for model_name, _ in MODELS.items()])
@pytest_cases.parametrize("layerwise", [True, False])
@pytest_cases.parametrize("fuse_scaling", [True, False])
def test_act_equalization_torchvision_models(model_dict: dict, layerwise: bool, fuse_scaling: bool):
    model, coverage = model_dict

    try:
        model = getattr(models, model)(pretrained=True, transform_input=False)
    except TypeError:
        model = getattr(models, model)(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)
    model.eval()

    model = symbolic_trace(model)
    model = TorchFunctionalToModule().apply(model)
    model = DuplicateSharedStatelessModule().apply(model)
    expected_out = model(inp)

    with torch.no_grad():
        with activation_equalization_mode(model,
                                          0.5,
                                          True,
                                          layerwise=layerwise,
                                          fuse_scaling=fuse_scaling) as aem:
            model(inp)
    scale_factor_regions = aem.scale_factors
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    out = model(inp)

    assert torch.allclose(expected_out, out, atol=ATOL)
    # Check that at least one region performs "true" equalization
    # If all shapes are scalar, no equalization has been performed
    assert any([shape != () for shape in shape_scale_regions])


def test_act_equalization_vanilla_mha_layerwise(vanilla_mha_model):
    # Activation equalization runs on the float model, before quantization operators are
    # inserted, so the realistic target is a plain nn.MultiheadAttention. In layerwise mode the
    # whole MHA module is hooked (nothing internal such as out_proj), and batch_first only
    # permutes the module's I/O. The batch dimension is therefore 0 when batch_first=True (N, L, E)
    # and 1 when batch_first=False (L, N, E).
    model_class = vanilla_mha_model
    batch_first = model_class.batch_first

    torch.manual_seed(SEED)
    inp = mha_input(batch_first)

    model = model_class()
    model.eval()
    with torch.no_grad():
        expected_out = model(inp)
        with activation_equalization_mode(model, 0.5, True, layerwise=True) as aem:
            model(inp)
        out = model(inp)

    batch_dim_map = aem.graph_act_eq.batch_dim_act_map
    # Exactly one module is hooked and it is the MHA itself (its internals are not hooked).
    assert len(batch_dim_map) == 1
    (hooked_module, batch_dim), = batch_dim_map.items()
    assert isinstance(hooked_module, torch.nn.MultiheadAttention)
    expected_batch_dim = 0 if batch_first else 1
    assert batch_dim == expected_batch_dim, \
        f"Expected batch_dim == {expected_batch_dim} for batch_first={batch_first}, got {batch_dim}"

    # Correctness: activation equalization must preserve the output.
    assert torch.allclose(expected_out, out, atol=ATOL)


def test_act_equalization_vanilla_mha_graph(vanilla_linear_mha_model):
    # Graph-mode counterpart of the layerwise test. A source (the Linear) is required so that a
    # region forms around the MHA sink. Graph mode already derives batch_dim from
    # module.batch_first, so this acts as a control that stays green.
    model_class = vanilla_linear_mha_model
    batch_first = model_class.batch_first

    torch.manual_seed(SEED)
    inp = mha_input(batch_first)

    model = model_class()
    model.eval()
    with torch.no_grad():
        expected_out = model(inp)

    model = symbolic_trace(model)
    with torch.no_grad():
        with activation_equalization_mode(model, 0.5, True, layerwise=False) as aem:
            model(inp)
        out = model(inp)

    batch_dims = list(aem.graph_act_eq.batch_dim_act_map.values())
    expected_batch_dim = 0 if batch_first else 1
    assert len(batch_dims) > 0
    assert all(batch_dim == expected_batch_dim for batch_dim in batch_dims), \
        f"Expected batch_dim == {expected_batch_dim} for batch_first={batch_first}, got {batch_dims}"

    assert torch.allclose(expected_out, out, atol=ATOL)


@pytest_cases.parametrize('layerwise', [True, False])
def test_act_equalization_vanilla_mha_layout_equivalence(layerwise):
    # With the correct batch dimension, activation equalization must compute identical scaling
    # factors regardless of whether the same data is presented as (N, L, E) (batch_first=True) or
    # (L, N, E) (batch_first=False). We use scale_computation_type='range' which, unlike the
    # default 'maxabs', is sensitive to reducing over the wrong dimension.
    torch.manual_seed(SEED)
    x_nle = torch.randn(MHA_BATCH_SIZE, MHA_SEQ_LEN, MHA_EMBED_DIM)
    x_lne = x_nle.transpose(0, 1).contiguous()

    def make_model(batch_first):

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                # Graph mode needs a source feeding the MHA sink. Layerwise mode hooks each layer
                # independently, so a lone MHA suffices; a standalone Linear's batch dimension
                # cannot be inferred from batch_first, so we omit it in layerwise mode.
                self.linear = None if layerwise else nn.Linear(MHA_EMBED_DIM, MHA_EMBED_DIM)
                self.relu = nn.ReLU()
                self.mha = nn.MultiheadAttention(
                    MHA_EMBED_DIM, MHA_NUM_HEADS, batch_first=batch_first)

            def forward(self, x):
                if self.linear is not None:
                    x = self.relu(self.linear(x))
                out, _ = self.mha(x, x, x)
                return out

        return Model()

    model_bf = make_model(batch_first=True)
    model_sf = make_model(batch_first=False)
    # Identical weights so any difference in scaling factors comes from the layout handling.
    model_sf.load_state_dict(model_bf.state_dict())

    def run(model, x):
        model.eval()
        if layerwise:
            eq = LayerwiseActivationEqualization(model, scale_computation_type='range')
        else:
            model = symbolic_trace(model)
            eq = GraphActivationEqualization(
                model, add_mul_node=True, scale_computation_type='range')
        eq.setup()
        with torch.no_grad():
            model(x)
        scale_factors, _ = eq.apply(0.5)
        return scale_factors

    scales_bf = run(model_bf, x_nle)
    scales_sf = run(model_sf, x_lne)

    assert len(scales_bf) == len(scales_sf) and len(scales_bf) > 0
    saw_non_scalar = False
    for scale_bf, scale_sf in zip(scales_bf, scales_sf):
        scale_bf = torch.as_tensor(scale_bf).float()
        scale_sf = torch.as_tensor(scale_sf).float()
        assert scale_bf.shape == scale_sf.shape
        assert torch.allclose(scale_bf, scale_sf, atol=ATOL), \
            f"Scaling factors differ between layouts: {scale_bf} vs {scale_sf}"
        if scale_bf.numel() > 1:
            saw_non_scalar = True
    # Ensure the test actually exercised non-trivial equalization.
    assert saw_non_scalar
