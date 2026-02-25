# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import pytest_cases
import torch

from brevitas.graph.equalize import GraphRotationEqualization
from brevitas.graph.permute import GraphPermutationEqualization
from brevitas.graph.permute import rotate_permute_mode
from tests.marker import requires_pt_ge

from .equalization_fixtures import *


def _has_tied_parameters(model: torch.nn.Module):
    """Auxiliar method to check if model has tied parameters"""
    # get all model parameters and their names
    all_named_parameters = {
        name: param for name, param in model.named_parameters(remove_duplicate=False)}

    # get only unique named parameters
    no_duplicate_named_parameters = {
        name: param for name, param in model.named_parameters(remove_duplicate=True)}

    # the difference of the two sets gives us the tied parameters
    tied_param_names = set(all_named_parameters.keys()) - set(no_duplicate_named_parameters.keys())

    return len(tied_param_names) > 0


def _setup_test_model(rotation_model, device='cpu'):
    """
    Helper function to setup a test model.

    Returns:
        tuple: (model, sample_inputs) where model is the FX-traced model on device
    """
    # Instantiate model
    model = rotation_model()

    # Skip tied parameters
    if _has_tied_parameters(model):
        pytest.skip("Skipping tests with tied parameters.")

    device = torch.device(device)
    model.to(device)

    # Sample input
    sample_inputs = torch.rand(size=(5, IN_FEATURES)).to(device)

    # Convert to FX graph
    with torch.no_grad():
        fx_model, _ = torch._dynamo.export(model)(sample_inputs)

    return fx_model, sample_inputs


@requires_pt_ge('2.3.1')
@pytest_cases.parametrize('permute_fn', ['massdiff', 'zigzag', 'absmax', 'random'])
@pytest_cases.parametrize('block_size', [8, IN_FEATURES])
@pytest_cases.parametrize('expansion_step', [0, 3])
@pytest_cases.parametrize('disable_for_fused_rotations', [True, False])
@pytest_cases.parametrize('orphan_sink', [True, False])
@pytest_cases.parametrize('device', ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
def test_rotate_permute_mode(
        rotation_model,
        permute_fn,
        block_size,
        expansion_step,
        disable_for_fused_rotations,
        orphan_sink,
        device):
    """Test rotate_permute_mode context manager with various configurations."""
    # Setup model
    model, sample_inputs = _setup_test_model(rotation_model, device)
    model.eval()
    with torch.no_grad():
        expected_output = model(sample_inputs)

    # Create rotation instance
    rotation = GraphRotationEqualization(
        expansion_step=expansion_step,
        layers_to_expand=[],
        rotation_block_size=block_size,
        orphan_sink=orphan_sink,
        disable_block_rotation_for_fused=disable_for_fused_rotations,
        return_rewriters=True,
        delay_rewriters=True)

    # Apply rotation and permutation through context manager
    with rotate_permute_mode(model,
                             rotation=rotation,
                             permute_fn=permute_fn,
                             block_size=block_size,
                             disable_for_fused_rotations=disable_for_fused_rotations) as rpm:
        permute_regions = rpm.permutation.regions
        permute_float_act_map = rpm.permutation.float_act_map
        with torch.no_grad():
            rpm.model(sample_inputs)
        # Verify activation maps were populated if regions exist
        if len(permute_regions) > 0:
            assert len(permute_float_act_map) > 0, \
                "Activation maps should be populated after forward pass"
        if (orphan_sink or not disable_for_fused_rotations) and block_size < IN_FEATURES:
            assert len(permute_regions) > 0

    # Verify output invariance
    with torch.no_grad():
        output = model(sample_inputs)
    assert torch.allclose(expected_output, output, atol=ATOL), \
        "Output mismatch with combined features"


@requires_pt_ge('2.3.1')
@pytest_cases.parametrize('block_size', [4, 8, 16, 24, 32])
@pytest_cases.parametrize('device', ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
def test_permute_block_size_compatibility(rotation_model, block_size, device):
    """
    Test block size compatibility with different model dimensions and region filtering.

    For IN_FEATURES=24, compatible block sizes are: 2, 3, 4, 6, 8, 12
    Block size of 24 is not compatible.
    Block sizes like 5, 7, 16, 32 should be incompatible and regions should be filtered.
    Verify this behavior is correct.
    """
    # Setup model
    model, sample_inputs = _setup_test_model(rotation_model, device)
    model.eval()
    with torch.no_grad():
        expected_output = model(sample_inputs)

    # Apply rotation to get regions
    rotation = GraphRotationEqualization(
        expansion_step=0,
        layers_to_expand=[],
        rotation_block_size=block_size,
        disable_block_rotation_for_fused=False,
        return_rewriters=True,
        delay_rewriters=True)

    model, rewriters = rotation.apply(model)
    regions = rotation.get_regions()

    # Setup permutation - this should handle incompatible block sizes gracefully
    permutation = GraphPermutationEqualization(block_size=block_size, permute_fn='massdiff')

    model = permutation.setup(model, regions)

    if block_size in [16, 24, 32]:
        assert len(permutation.regions) == 0

    # Verify that SDPA regions are filtered (regions with 'value_sdpa' in source names)
    for region in permutation.regions:
        assert 'value_sdpa' not in region.srcs_names, \
            "SDPA regions should be filtered out"

    # Run model to collect statistics and apply permutations
    with torch.no_grad():
        model(sample_inputs)

    model = permutation.apply(model)
    permutation.cleanup()

    # Verify output invariance
    with torch.no_grad():
        output = model(sample_inputs)
    assert torch.allclose(expected_output, output, atol=ATOL), \
        "Output changed after permutation - invariance violated"
