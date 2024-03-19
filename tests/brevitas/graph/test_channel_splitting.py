# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.fx import symbolic_trace
from brevitas.graph.channel_splitting import _clean_regions
from brevitas.graph.channel_splitting import _split
from brevitas.graph.channel_splitting import GraphChannelSplitting
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.fixed_point import MergeBatchNorm
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model

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
    model_name = request.node.callspec.id.split('-')[0]

    torch.manual_seed(SEED)

    model_class = toy_model
    model = model_class()
    if 'mha' in model_name:
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
    regions = _clean_regions(regions, region_filter_func=lambda x, y: True)
    if model_name in no_split_models:
        assert len(regions) == 0
    else:
        model = _split(
            model, regions, split_input=split_input, layer_split_perc_func=lambda x: SPLIT_RATIO)

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
    model_name = request.node.callspec.id.split('-')[0]

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
    regions = _clean_regions(regions, region_filter_func=lambda x, y: True)
    if model_name in no_split_models:
        assert len(regions) == 0
    else:
        model = _split(
            model, regions, split_input=split_input, layer_split_perc_func=lambda x: SPLIT_RATIO)

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


@pytest.mark.parametrize('split_input', [False, True])
def test_quant_toymodels(toy_model, split_input, request):
    model_name = request.node.callspec.id.split('-')[0]

    torch.manual_seed(SEED)

    model_class = toy_model
    model = model_class()
    if 'mha' in model_name:
        pytest.skip('MHA not supported with this quantization method')
    else:
        inp = torch.randn(IN_SIZE_CONV)

    # preprocess model for quantization, like merge BN etc.
    model = preprocess_for_quantize(model)
    # save regions
    regions = _extract_regions(model)
    # quantize model pretty basic
    quant_model = quantize_model(
        model,
        backend='layerwise',
        weight_bit_width=8,
        act_bit_width=8,
        bias_bit_width=32,
        scale_factor_type='float_scale',
        weight_narrow_range=False,
        weight_param_method='mse',
        weight_quant_granularity='per_channel',
        weight_quant_type='sym',
        layerwise_first_last_bit_width=8,
        act_param_method='stats',
        act_quant_percentile=99.999,
        act_quant_type='sym',
        quant_format='int')

    expected_out = quant_model(inp)

    # save model's state dict to check if channel splitting was done or not
    old_state_dict = quant_model.state_dict()

    # quant_regions should be the same
    quant_regions = _extract_regions(quant_model)
    quant_regions = _clean_regions(quant_regions, region_filter_func=lambda x, y: True)

    if model_name in no_split_models:
        assert len(quant_regions) == 0
    else:
        # check regions
        assert len(quant_regions) == len(regions)

        # pass custom split function here
        quant_model = _split(
            quant_model,
            quant_regions,
            split_input=split_input,
            layer_split_perc_func=lambda x: SPLIT_RATIO)

        out = quant_model(inp)
        # checking if the outputs are all close, doesn't work for split_input = True
        assert torch.allclose(expected_out, out, atol=0.1)

        modified_sources = {source for region in quant_regions for source in region.srcs_names}
        # avoiding checking the same module multiple times
        modified_sinks = {
            sink for region in quant_regions for sink in region.sinks_names} - modified_sources
        for module in modified_sources:
            if 'mha' in module:
                module += '.out_proj'
            weight_name = module + '.weight'
            assert not torch.equal(
                old_state_dict[weight_name], quant_model.state_dict()[weight_name])
            bias_name = module + '.bias'
            # not all modules have bias and they only differ when splitting output channels
            if bias_name in old_state_dict.keys():
                assert not torch.equal(
                    old_state_dict[bias_name], quant_model.state_dict()[bias_name])
        for module in modified_sinks:
            weight_name = module + '.weight'
            assert not torch.equal(
                old_state_dict[weight_name], quant_model.state_dict()[weight_name])


@pytest.mark.parametrize('split_input', [False, True])
def test_torchvision_models_preprocessing(model_coverage: tuple, split_input: bool, request):
    model_name = request.node.callspec.id.split('-')[0]

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
    regions = _clean_regions(regions, region_filter_func=lambda x, y: True)

    # use default channel absmax for criterion and split evenly for split_func
    model, split_regions = GraphChannelSplitting(
            layer_split_perc_func=lambda x: SPLIT_RATIO,
            region_filter_func=lambda x, y: True,
            split_input=split_input).apply(model, return_regions=True)
    if model_name in no_split_models:
        assert len(regions) == 0
    else:
        # check if regions are the same
        assert len(regions) == len(split_regions)

        out = model(inp)
        assert torch.allclose(expected_out, out, atol=ATOL)

        modified_sources = {source for region in split_regions for source in region.srcs_names}
        # avoiding checking the same module multiple times
        modified_sinks = {
            sink for region in split_regions for sink in region.sinks_names} - modified_sources
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
