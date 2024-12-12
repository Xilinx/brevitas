# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from functools import partial
import itertools
from typing import List, Tuple
from unittest.mock import patch

import pytest
import torch
from torchvision import models

from brevitas.fx import symbolic_trace
# TODO: Refactor to prevent circular import
from brevitas.graph.equalize import _apply_ort_device
from brevitas.graph.equalize import _batch_norm
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.equalize import _fuse_rotations
from brevitas.graph.equalize import _get_input_axis
from brevitas.graph.equalize import _get_output_axis
from brevitas.graph.equalize import _is_supported_module
from brevitas.graph.equalize import _supported_layers
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.equalize import GraphRotationEqualization
from brevitas.graph.equalize import MergeLnAffine
from brevitas.graph.equalize import random_orthogonal_matrix
from brevitas.graph.hadamard import matmul_hadU
from brevitas.graph.standardize import DuplicateSharedStatelessModule
from brevitas.graph.standardize import TorchFunctionalToModule
from brevitas.graph.utils import get_module
from brevitas.nn.equalized_layer import RotatedModule
from brevitas.nn.equalized_layer import UnfusedRotatedModule
from tests.marker import requires_pt_ge

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
        regions, merge_bias=True, bias_shrinkage='vaiq', scale_computation_type='maxabs')
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
    # The isistance does not work after symbolic trace
    is_alexnet = isinstance(model, models.AlexNet)
    model = symbolic_trace(model)
    model = TorchFunctionalToModule().apply(model)

    expected_out = model(inp)

    supported_sinks = list(_supported_layers)
    supported_sinks = tuple([
        x for x in _supported_layers if x not in (torch.nn.LayerNorm, *_batch_norm)])
    regions = _extract_regions(model, state_impl_kwargs={'supported_sinks': supported_sinks})
    scale_factor_regions = equalize_test(
        regions, merge_bias=merge_bias, bias_shrinkage='vaiq', scale_computation_type='maxabs')
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
    if is_alexnet:
        # In AlexNet, we cannot equalize only through one region
        assert sum([shape == () for shape in shape_scale_regions]) == 1
    else:
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
        regions, merge_bias=merge_bias, bias_shrinkage='vaiq', scale_computation_type='maxabs')
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    with torch.no_grad():
        out = model(inp)
    assert len(regions) > 0
    assert torch.allclose(expected_out, out, atol=ATOL)
    # Check that at least one region performs "true" equalization
    # If all shapes are scalar, no equalization has been performed
    if 'convgroupconv' in test_id:
        with pytest.raises(AssertionError):
            assert all([shape != () for shape in shape_scale_regions])
    else:
        assert all([shape != () for shape in shape_scale_regions])


@pytest_cases.parametrize("layerwise", [True, False])
def test_act_equalization_models(toy_model, layerwise, request):
    test_id = request.node.callspec.id

    if 'mha' in test_id:
        in_shape = IN_SIZE_LINEAR
    else:
        in_shape = IN_SIZE_CONV

    model_class = toy_model
    model = model_class()
    inp = torch.randn(in_shape)

    model.eval()
    expected_out = model(inp)
    model = symbolic_trace(model)
    with torch.no_grad():
        with activation_equalization_mode(model, 0.5, True, layerwise=layerwise) as aem:
            regions = aem.graph_act_eq.regions
            model(inp)
    scale_factor_regions = aem.scale_factors
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    out = model(inp)
    assert torch.allclose(expected_out, out, atol=ATOL)

    # This region is made up of a residual branch, so no regions are found for act equalization
    if 'convgroupconv' in test_id:
        with pytest.raises(AssertionError):
            assert len(regions) > 0
            # Check that at least one region performs "true" equalization
            # If all shapes are scalar, no equalization has been performed
            assert all([shape != () for shape in shape_scale_regions])
    else:
        assert len(regions) > 0
        # Check that at least one region performs "true" equalization
        # If all shapes are scalar, no equalization has been performed
        assert all([shape != () for shape in shape_scale_regions])


@pytest_cases.parametrize(
    "model_dict", [(model_name, coverage) for model_name, coverage in MODELS.items()],
    ids=[model_name for model_name, _ in MODELS.items()])
@pytest_cases.parametrize("layerwise", [True, False])
def test_act_equalization_torchvision_models(model_dict: dict, layerwise: bool):
    model, coverage = model_dict

    if model == 'googlenet' and torch_version == version.parse('1.8.1'):
        pytest.skip(
            'Skip because of PyTorch error = AttributeError: \'function\' object has no attribute \'GoogLeNetOutputs\' '
        )
    if 'vit' in model and torch_version < version.parse('1.13'):
        pytest.skip(
            f'ViT supported from torch version 1.13, current torch version is {torch_version}')

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
        with activation_equalization_mode(model, 0.5, True, layerwise=layerwise) as aem:
            model(inp)
    scale_factor_regions = aem.scale_factors
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    out = model(inp)

    assert torch.allclose(expected_out, out, atol=ATOL)
    # Check that at least one region performs "true" equalization
    # If all shapes are scalar, no equalization has been performed
    assert any([shape != () for shape in shape_scale_regions])


@requires_pt_ge('2.4')
@pytest_cases.parametrize('partial_had', [True, False])
def test_models(rotation_fixtures, partial_had):

    in_shape = IN_SIZE_LINEAR

    model_class = rotation_fixtures
    model = model_class()
    inp = torch.ones(in_shape)

    model.eval()
    penultimate_weight = model.linear_1.weight.data
    last_weight = model.linear_2.weight.data
    with torch.no_grad():
        expected_out = model(inp)

    model = symbolic_trace(model)
    merge = MergeLnAffine()
    model = merge.apply(model)
    eq = GraphRotationEqualization(orphan_sink=partial_had)
    model = eq.apply(model)

    with torch.no_grad():
        out = model(inp)

    penultimate_weight_new = model.linear_1.weight.data

    # Invariance of the output
    assert torch.allclose(out, expected_out, atol=ATOL)
    # Rotate weights must be different
    assert not torch.allclose(penultimate_weight, penultimate_weight_new)
    # Merging affine parameters of RMS
    assert torch.allclose(model.rms.weight.data, torch.ones_like(model.rms.weight.data))
    if partial_had:
        last_weight_new = model.linear_2.layer.weight.data
        assert not torch.allclose(last_weight, last_weight_new)


def _rotate_input_output(is_source: bool, is_sink: bool, is_orphan: bool) -> Tuple[bool, bool]:
    # Verify that only one flag is enabled at the same time
    assert sum([is_source, is_sink, is_orphan]) <= 1, "Only one flag can be enabled."

    rotate_input, rotate_output = False, False
    if is_source:
        rotate_output = True
    if is_sink:
        rotate_input = True

    return rotate_input, rotate_output


def _compute_rotated_ouptut_from_matrices(
        module: nn.Module, inp: torch.Tensor, rot_mat_input: torch.Tensor,
        rot_mat_output: torch.Tensor):
    # If the node is a sink, the input is multiplied by the inverse of the rotation matrix x <- xQ^{-1}
    inp = inp @ rot_mat_input.t()
    # If the node is a source, the output is multiplied by the rotation matrix o <- oQ
    out = module(inp) @ rot_mat_output
    # Return rotated output
    return out


# NOTE: The assumption is that only one flag can be true simultaneously
# NOTE: Orphans need to be taken care of. A module can only be orphan once.
def _generate_rotation_flags(N: int) -> List[bool]:
    return [
        rotation_flags for rotation_flags in itertools.product([False, True], repeat=3 * N) if (
            all([sum(rotation_flags[i * 3:(i + 1) * 3]) <= 1 for i in range(N)]) and
            # Only outermost rotation can be orphan
            all([not rotation_flags[i * 3 + 2] for i in range(N - 1)]))]


@requires_pt_ge('2.4')
@pytest_cases.parametrize('N', [1, 2, 3], ids=lambda x: f"N={x}")
def test_composition_unfused_rotation_layer(N):
    torch.manual_seed(SEED)

    for rotation_flags in _generate_rotation_flags(N):

        in_features = IN_FEATURES_LINEAR
        module = nn.Linear(in_features=in_features, out_features=in_features)

        # Sample input to pass through the block
        sample_input = torch.rand((1, in_features),)

        # Compose rotation modules
        rotated_module = module

        # Composite rotation matrices
        rot_mat_input = torch.eye(in_features)
        rot_mat_output = torch.eye(in_features)

        for i in range(N):
            module_rotation_flags = rotation_flags[i * 3:(i + 1) * 3]
            is_source, is_sink, is_orphan = module_rotation_flags
            rotate_input, rotate_output = _rotate_input_output(is_source, is_sink, is_orphan)

            # Generate a random matrix
            rot_mat = random_orthogonal_matrix(in_features).to(dtype=torch.float32)

            # Aggregate rotation matrices
            if rotate_input:
                rot_mat_input = rot_mat_input @ rot_mat
            if rotate_output:
                rot_mat_output = rot_mat_output @ rot_mat

            # Compose rotation modules
            rotated_module = UnfusedRotatedModule(
                module=rotated_module,
                rot_func=_apply_ort_device,
                _get_input_axis=_get_input_axis,
                _get_output_axis=_get_output_axis,
                rot_mat=rot_mat,
                is_source=is_source,
                is_sink=is_sink,
                is_orphan=is_orphan,
            )

        # Compute outputs to compare
        gt_output = _compute_rotated_ouptut_from_matrices(
            module, sample_input, rot_mat_input, rot_mat_output)
        rot_output = rotated_module(sample_input)

        # Verify that the rotation operations were computed correctly
        assert torch.allclose(gt_output, rot_output, atol=ATOL)


# Adapted from https://github.com/facebookresearch/SpinQuant/blob/main/eval_utils/rotation_utils.py#L26
def _random_orthogonal_matrix(size, generator):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64, generator=generator)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0).float()
    return q


def _random_hadamard_matrix(size, device, generator):
    # See https://github.com/Cornell-RelaxML/quip-sharp , Section "Randomized Hadamard Transformation"
    Q = torch.randint(low=0, high=2, size=(size,), generator=generator).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)


def _compare_module_weights_fused_unfused(gt_module, rot_module, fused_rotations=False):
    gt_weight = gt_module.weight if isinstance(gt_module, nn.Linear) else gt_module.layer.weight
    gt_bias = gt_module.bias if isinstance(gt_module, nn.Linear) else gt_module.layer.bias
    if fused_rotations:
        rot_weight = rot_module.weight if isinstance(
            rot_module, nn.Linear) else rot_module.layer.weight
        rot_bias = rot_module.bias if isinstance(rot_module, nn.Linear) else rot_module.layer.bias
    else:
        rot_weight = rot_module.weight
        rot_bias = rot_module.bias
    assert torch.allclose(gt_weight, rot_weight, rtol=0.0, atol=0.0)
    if gt_bias is not None:
        assert torch.allclose(gt_bias, rot_bias, rtol=0.0, atol=0.0)
    # For a RotatedModule, corresponding to an orphan node, additional checks need to be done
    if isinstance(gt_module, RotatedModule):
        if not fused_rotations:
            # The outermost should be an orphan
            child_rot_module = rot_module
            assert child_rot_module.is_orphan, "Unfused rotated module needs to be an orphan."
            # Check that the inner UnfusedRotatedModules are not orphans
            while isinstance(child_rot_module.module, UnfusedRotatedModule):
                assert not child_rot_module.module.is_orphan, "Inner unfused rotated modules cannot be orphans."
                child_rot_module = child_rot_module.module
            # Verify that the rotation matrices match
            assert torch.allclose(gt_module.had_mat, rot_module.rot_mat)


# This test verifies that the weights returned by the unfused rotate modules
# match those when fusing
@requires_pt_ge('2.4')
@pytest_cases.parametrize('partial_had', [False, True])
@pytest_cases.parametrize('fused_rotations', [False, True])
def test_models_rotations(rotation_fixtures, partial_had, fused_rotations):

    in_shape = IN_SIZE_LINEAR

    model_class = rotation_fixtures
    model = model_class()

    model.eval()
    inp = torch.rand(in_shape)

    model = symbolic_trace(model)
    merge = MergeLnAffine()
    model = merge.apply(model)
    eq = GraphRotationEqualization(orphan_sink=partial_had, full_rotation_method='ort')

    # Save a copy to apply graph rotation equalization on
    model_copy = copy.deepcopy(model)

    # We need to make sure that the same random matrices are being generated
    generator = torch.Generator()
    generator.manual_seed(SEED)
    # Clone generator to make sure we can use the same rotation matrices
    generator_clone = generator.clone_state()

    # We pass the generator to make sure that we can reproduce the random orthogonal matrices that are generated
    with patch('brevitas.graph.equalize.random_orthogonal_matrix',
               partial(_random_orthogonal_matrix, generator=generator)):
        # Apply rotation equalization while controlling the random matrices that are generated
        model = eq.apply(model)

    with torch.no_grad():
        expected_out = model(inp)

    # Now rotate but without fusing the rotation matrices
    with patch('brevitas.graph.equalize.random_orthogonal_matrix',
               partial(_random_orthogonal_matrix, generator=generator_clone)):
        # Apply rotation equalization while controlling the random matrices that are generated
        model_copy = eq.apply(model_copy, fuse_rotations=False)

    # Fuse the rotations and make sure the behaviour is the same
    if fused_rotations:
        _fuse_rotations(model_copy)

    with torch.no_grad():
        out = model_copy(inp)

    # Verify that the output of the model does not change after incorporating the rotations
    assert torch.allclose(expected_out, out, rtol=0.0, atol=0.0)

    # Verify that weight matrices
    for model_node, model_copy_node in zip(model.graph.nodes, model_copy.graph.nodes):
        if model_node.op == 'call_module':
            module = get_module(model, model_node.target)
            module_copy = get_module(model_copy, model_copy_node.target)
            if isinstance(module, (nn.Linear, RotatedModule)):
                _compare_module_weights_fused_unfused(module, module_copy, fused_rotations)


def _compare_module_weights(module, module_copy):
    weight = module.weight if isinstance(module, nn.Linear) else module.layer.weight
    bias = module.bias if isinstance(module, nn.Linear) else module.layer.bias
    weight_copy = module_copy.weight
    bias_copy = module_copy.bias
    assert torch.allclose(weight, weight_copy, rtol=0.0, atol=0.0)
    if bias is not None:
        assert torch.allclose(bias, bias_copy, rtol=0.0, atol=0.0)


import logging

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from brevitas.graph.equalize import find_missing_rotation_regions
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.llm.llm_quant.data_utils import get_dataset_for_model
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_affine_merge
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_to_rmsnorm
from brevitas_examples.llm.llm_quant.ln_affine_merge import replace_rmsnorm_with_torch
from brevitas_examples.llm.llm_quant.run_utils import fix_rewriter
from brevitas_examples.llm.main import fused_optimized_rotation_no_fx
from brevitas_examples.llm.main import fused_rotation_no_fx
from tests.brevitas_examples.test_llm import default_run_args


@pytest_cases.fixture(
    ids=[
        "llama",],
    params=[
        {
            "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "input_bit_width": None,
            "fuse_sequences": False,
            "act_calibration": False,},])
def equalize_args(default_run_args, request):
    args = default_run_args
    export_dict = request.param
    args.update(**export_dict)
    yield args


@pytest.mark.llm
@requires_pt_ge('2.4')
@pytest_cases.parametrize('partial_had', [False, True])
@pytest_cases.parametrize('fused_rotations', [False, True])
def test_small_models_equalize_legacy_rotation_orthogonal(
        caplog, partial_had, fused_rotations, equalize_args):
    import os
    os.environ["HF_HUB_CACHE"] = "/scratch/hf_models/"
    caplog.set_level(logging.INFO)
    args = equalize_args
    args.rotation_orphan_sink = partial_had
    args.rotation_mode = 'ort'

    kwargs = {"torch_dtype": torch.float16}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model = replace_rmsnorm_with_torch(model, model.config)
    model.config.use_cache = False
    print("Model loaded.")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Load the data for calibration and evaluation.
    calibration_loader = get_dataset_for_model(
        args.model,
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        split="train",
        seed=args.seed,
        require_fx=False,
        device=None,
        fuse_sequences=args.fuse_sequences)

    # We need to make sure that the same random matrices are being generated
    generator = torch.Generator()
    generator.manual_seed(SEED)
    # Clone generator to make sure we can use the same rotation matrices
    generator_clone = generator.clone_state()

    # Save a copy to apply graph rotation equalization on
    model_copy = copy.deepcopy(model)

    # We pass the generator to make sure that we can reproduce the random orthogonal matrices that are generated
    with patch('brevitas.graph.equalize.random_orthogonal_matrix',
               partial(_random_orthogonal_matrix, generator=generator)):
        with patch('brevitas.graph.hadamard.random_hadamard_matrix',
                   partial(_random_hadamard_matrix, generator=generator)):
            fused_rotation_no_fx(model, calibration_loader, args, fuse_rotations=True)

    # Run model and save outputs
    with torch.no_grad():
        expected_logits = model(**calibration_loader[0]).logits

    # We pass the generator to make sure that we can reproduce the random orthogonal matrices that are generated
    with patch('brevitas.graph.equalize.random_orthogonal_matrix',
               partial(_random_orthogonal_matrix, generator=generator_clone)):
        with patch('brevitas.graph.hadamard.random_hadamard_matrix',
                   partial(_random_hadamard_matrix, generator=generator_clone)):
            fused_rotation_no_fx(model_copy, calibration_loader, args, fuse_rotations=False)

    if fused_rotations:
        _fuse_rotations(model_copy)

    # Run model and save outputs
    with torch.no_grad():
        logits = model_copy(**calibration_loader[0]).logits

    # Verify that the output is the same
    assert torch.allclose(expected_logits, logits)

    # Verify that the weights after fusing match
    for name_fused_module, fused_module in model.named_modules():
        # For linear modules verify that the weights match
        if isinstance(fused_module, (nn.Linear, RotatedModule)):
            for name_unfused_Module, unfused_module in model_copy.named_modules():
                if name_fused_module == name_unfused_Module:
                    _compare_module_weights(fused_module, unfused_module)
                    # For a RotatedModule, corresponding to an orphan node, additional checks need to be done
                    if isinstance(fused_module, RotatedModule):
                        # Verify that the outer module is an orphan
                        if fused_rotations:
                            assert isinstance(unfused_module, RotatedModule)
                            assert torch.allclose(fused_module.had_mat, unfused_module.had_mat)
                        else:
                            assert unfused_module.is_orphan
                            # Verify that the rotation matrices match
                            assert torch.allclose(fused_module.had_mat, unfused_module.rot_mat)


from itertools import product

from brevitas.graph.equalize import _apply_had_device
from brevitas.graph.hadamard import get_hadK


# NOTE: This test works because in R2 we patch the rotation method, so the appropiate matrix is not effectively used. This is because when the fast_hadamard_transform is not avai
@pytest.mark.llm
@requires_pt_ge('2.4')
@pytest_cases.parametrize(
    'partial_had, fused_rotations, add_additional_regions',
    list(product([False, True], repeat=3)),
    ids=[("fused-R1" if fused_rotations else "R1") + ("-R2" if add_additional_regions else "") +
         ("-R3" if partial_had else "") for partial_had,
         fused_rotations,
         add_additional_regions in list(product([False, True], repeat=3))],
)
@pytest_cases.parametrize('rotation_mode', ['ort', 'had'])
def test_small_models_equalize_mixed_fused_unfused(
        caplog, partial_had, fused_rotations, add_additional_regions, rotation_mode, equalize_args):
    import os
    os.environ["HF_HUB_CACHE"] = "/scratch/hf_models/"
    caplog.set_level(logging.INFO)
    args = equalize_args
    args.rotation_orphan_sink = partial_had
    args.rotation_mode = rotation_mode

    kwargs = {"torch_dtype": torch.float16}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model = replace_rmsnorm_with_torch(model, model.config)
    model.config.use_cache = False
    print("Model loaded.")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Load the data for calibration and evaluation.
    calibration_loader = get_dataset_for_model(
        args.model,
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        split="train",
        seed=args.seed,
        require_fx=False,
        device=None,
        fuse_sequences=args.fuse_sequences)

    # We need to make sure that the same random matrices are being generated
    generator = torch.Generator()
    generator.manual_seed(SEED)
    # Clone generator to make sure we can use the same rotation matrices
    generator_clone = generator.clone_state()

    # Run model and save outputs
    with torch.no_grad():
        original_logits = model(**calibration_loader[0]).logits

    # Save a copy to apply graph rotation equalization on
    model_copy = copy.deepcopy(model)

    with patch('brevitas.graph.equalize.random_orthogonal_matrix',
               partial(_random_orthogonal_matrix, generator=generator)):
        fused_optimized_rotation_no_fx(
            model,
            calibration_loader,
            args,
            fuse_rotations=True,
            add_additional_regions=add_additional_regions)

    # Run model and save outputs
    with torch.no_grad():
        expected_logits = model(**calibration_loader[0]).logits

    # Instead of random orthogonal matrices, we want to use the same ones as when the activations are not fused.
    if rotation_mode == 'had':
        with patch('brevitas.graph.equalize._apply_ort_device', _apply_had_device):
            fused_optimized_rotation_no_fx(
                model_copy,
                calibration_loader,
                args,
                fuse_rotations=False,
                add_additional_regions=add_additional_regions)
    else:
        with patch('brevitas.graph.equalize.random_orthogonal_matrix',
                   partial(_random_orthogonal_matrix, generator=generator_clone)):
            fused_optimized_rotation_no_fx(
                model_copy,
                calibration_loader,
                args,
                fuse_rotations=False,
                add_additional_regions=add_additional_regions)

    # Fuse matrices with module weights
    if fused_rotations:
        _fuse_rotations(model_copy)

    ids_rot = set()
    num_rotation_matrices = 0
    # Count the number of unique rotation matrices
    for module in model_copy.modules():
        if isinstance(module, UnfusedRotatedModule):
            if id(module.rot_mat) not in ids_rot:
                num_rotation_matrices += 1
                ids_rot.add(id(module.rot_mat))

    num_rotated_modules = 0
    # Count the number of RotatedModules
    for module in model_copy.modules():
        if isinstance(module, RotatedModule):
            num_rotated_modules += 1

    # Run model and save outputs
    with torch.no_grad():
        logits = model_copy(**calibration_loader[0]).logits

    # Verify that the number of learnable rotation matrices is the expected (R1 + one R2 per block)
    expected_number_rotation_matrices = 0 if fused_rotations else (
        1 + (model.config.num_hidden_layers if add_additional_regions else 0))
    assert num_rotation_matrices == expected_number_rotation_matrices, f"Expected {expected_number_rotation_matrices} learnable rotations, found {num_rotation_matrices}."

    # Verify that the number of rotated modules is correct
    expected_number_rotated_modules = 0 if not partial_had else (
        model.config.num_hidden_layers if add_additional_regions else 2 *
        model.config.num_hidden_layers)
    assert num_rotated_modules == expected_number_rotated_modules, f"Expected {expected_number_rotated_modules} learnable rotations, found {num_rotated_modules}."

    # Verify that the rotated module output is similar to the original FP
    assert torch.allclose(original_logits, logits, atol=ATOL)
    # Verify that the output is the same
    assert torch.allclose(expected_logits, logits, atol=0.0, rtol=0.0)

    # Verify that the weights after fusing match
    for name_fused_module, fused_module in model.named_modules():
        # For linear modules verify that the weights match
        if isinstance(fused_module, (nn.Linear, RotatedModule)):
            for name_unfused_Module, unfused_module in model_copy.named_modules():
                if name_fused_module == name_unfused_Module:
                    _compare_module_weights(fused_module, unfused_module)
                    # In case a RotatedModule is found, additional checks need to be done.
                    if isinstance(fused_module, RotatedModule):
                        if fused_rotations:
                            assert isinstance(unfused_module, RotatedModule)
                            assert torch.allclose(fused_module.had_mat, unfused_module.had_mat, rtol=0.0, atol=0.0), "The rotation matrices do not match."
                        else:
                            # Iterate over child nodes until finding the innermost RotatedModule
                            child_module = unfused_module
                            while isinstance(child_module, UnfusedRotatedModule):
                                assert not child_module.is_orphan, "UnfusedRotatedModule should not be an orphan."
                                child_module = child_module.module
                            # After finding the inner Rotated Module, they need to be compared
                            assert isinstance(child_module, RotatedModule), "Inner module should be RotatedModule."
                            assert torch.allclose(fused_module.had_mat, child_module.had_mat, rtol=0.0, atol=0.0), "The rotation matrices do not match."
