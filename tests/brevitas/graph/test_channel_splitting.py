# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.fx import symbolic_trace
from brevitas.graph.channel_splitting import _clean_regions
from brevitas.graph.channel_splitting import _split
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.fixed_point import MergeBatchNorm

from .equalization_fixtures import *

no_split_models = (
    'mul_model',
    'bnconv_model',
    'convdepthconv_model',
    'linearmha_model',
    'layernormmha_model',
    'convgroupconv_model',
    'vit_b_32',
    'shufflenet_v2_x0_5',
    'googlenet',
    'inception_v3')

SPLIT_RATIO = 0.1


@pytest.mark.parametrize('split_input', [False, True])
def test_toymodels(toy_model, split_input, request):
    test_id = request.node.callspec.id

    torch.manual_seed(SEED)

    model_class = toy_model
    model = model_class()
    if 'mha' in test_id:
        inp = torch.randn(IN_SIZE_LINEAR)
    else:
        inp = torch.randn(IN_SIZE_CONV)

    model.eval()
    expected_out = model(inp)

    model = symbolic_trace(model)
    # merge BN before applying channel splitting
    model = MergeBatchNorm().apply(model)

    # save model's state dict to check if channel splitting was done or not
    old_state_dict = model.state_dict()

    regions = _extract_regions(model)
    regions = _clean_regions(regions)
    if model_class in no_split_models:
        assert len(regions) == 0
    else:
        model = _split(model, regions, split_ratio=SPLIT_RATIO, split_input=split_input)

        out = model(inp)
        assert torch.allclose(expected_out, out, atol=ATOL)

        modified_sources = {source for region in regions for source in region.srcs_names}
        # avoiding checking the same module multiple times
        modified_sinks = {
            sink for region in regions for sink in region.sinks_names} - modified_sources
        for module in modified_sources:
            if 'mha' in module:
                module += '.out_proj'
            weight_name = module + '.weight'
            assert not torch.equal(old_state_dict[weight_name], model.state_dict()[weight_name])
            bias_name = module + '.bias'
            # not all modules have bias and they only differ when splitting output channels
            if bias_name in old_state_dict.keys() and not split_input:
                assert not torch.equal(old_state_dict[bias_name], model.state_dict()[bias_name])
        for module in modified_sinks:
            weight_name = module + '.weight'
            assert not torch.equal(old_state_dict[weight_name], model.state_dict()[weight_name])


@pytest.mark.parametrize('split_input', [False, True])
def test_torchvision_models(model_coverage: tuple, split_input: bool, request):
    model_class = request.node.callspec.id.split('-')[0]

    model, coverage = model_coverage

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)

    model.eval()
    expected_out = model(inp)

    model = symbolic_trace(model)
    # merge BN before applying channel splitting
    model = MergeBatchNorm().apply(model)

    old_state_dict = model.state_dict()

    regions = _extract_regions(model)
    regions = _clean_regions(regions)
    if model_class in no_split_models:
        assert len(regions) == 0
    else:
        model = _split(model, regions, split_ratio=SPLIT_RATIO, split_input=split_input)

        out = model(inp)
        assert torch.allclose(expected_out, out, atol=ATOL)

        modified_sources = {source for region in regions for source in region.srcs_names}
        # avoiding checking the same module multiple times
        modified_sinks = {
            sink for region in regions for sink in region.sinks_names} - modified_sources
        for module in modified_sources:
            weight_name = module + '.weight'
            assert not torch.equal(old_state_dict[weight_name], model.state_dict()[weight_name])
            bias_name = module + '.bias'
            # not all modules have bias and they only differ when splitting output channels
            if bias_name in old_state_dict.keys() and not split_input:
                assert not torch.equal(old_state_dict[bias_name], model.state_dict()[bias_name])
        for module in modified_sinks:
            weight_name = module + '.weight'
            assert not torch.equal(old_state_dict[weight_name], model.state_dict()[weight_name])
