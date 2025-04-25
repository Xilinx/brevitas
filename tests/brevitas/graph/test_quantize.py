import copy
import platform

import pytest
import pytest_cases
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from brevitas.graph.base import _remove_parametrization_entries_state_dict
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.quantize import quantize
from brevitas.utils.parametrization_utils import RotationWeightParametrization
from brevitas.utils.python_utils import recurse_getattr
from tests.marker import requires_pt_ge
from tests.marker import requires_pt_lt


@pytest_cases.parametrize(
    'kwargs',
    [
        {
            'model': nn.Sequential(nn.Linear(2, 3)),
            'name_blacklist': [],
            'key': '0',
            'expected': "<class 'brevitas.nn.quant_linear.QuantLinear'>"},
        {
            'model': nn.Sequential(nn.Linear(2, 3)),
            'name_blacklist': ['0'],
            'key': '0',
            'expected': "<class 'torch.nn.modules.linear.Linear'>"},
        {
            'model': nn.Sequential(nn.Sequential(nn.Linear(2, 3))),
            'name_blacklist': ['0'],
            'key': '0.0',
            'expected': "<class 'torch.nn.modules.linear.Linear'>"},
        {
            'model': nn.Sequential(nn.Sequential(nn.Linear(2, 3))),
            'name_blacklist': ['0.0'],
            'key': '0.0',
            'expected': "<class 'torch.nn.modules.linear.Linear'>"},])
def test_layerwise_quantize_blacklist(kwargs):
    key = kwargs['key']
    exp = kwargs['expected']
    del kwargs['key']
    del kwargs['expected']
    qmodel = layerwise_quantize(**kwargs)
    checked = False
    found_names = []
    for n, m in qmodel.named_modules():
        found_names.append(n)
        if n == key:
            mt = str(type(m))
            assert mt == exp, f"Expect module {n} to be type: {exp}, found type {mt}"
            checked = True
    assert checked, f"Layer named {key} not found. Layer names are: {found_names}"


@pytest_cases.parametrize(
    'kwargs',
    [
        {
            'model': nn.Sequential(nn.Linear(2, 3)),
            'rot_mat': torch.tensor([[1., -1.], [1., 1.]]) / torch.sqrt(torch.tensor(2.)),
            'rot_func': lambda tensor,
                        rot_mat,
                        K: torch.matmul(tensor, rot_mat),
            'key': '0',
            'expected': "<class 'torch.nn.utils.parametrize.ParametrizedQuantLinear'>"},])
def test_layerwise_quantize_parametrized_modules(kwargs):
    key = kwargs['key']
    exp = kwargs['expected']
    rot_mat = kwargs['rot_mat']
    rot_func = kwargs['rot_func']
    del kwargs['key']
    del kwargs['expected']
    del kwargs['rot_mat']
    del kwargs['rot_func']

    model = kwargs["model"]
    module = recurse_getattr(model, key)
    # Register rotation parametrization to module
    parametrize.register_parametrization(
        module=module,
        tensor_name="weight",
        parametrization=RotationWeightParametrization(
            rot_mat=nn.Parameter(rot_mat),
            rot_func=rot_func,
            axis=1,
            K=None,
        ))
    qmodel = layerwise_quantize(**kwargs)
    checked = False
    found_names = []
    for n, m in qmodel.named_modules():
        found_names.append(n)
        if n == key:
            mt = str(type(m))
            assert mt == exp, f"Expect module {n} to be type: {exp}, found type {mt}"
            checked = True
    assert checked, f"Layer named {key} not found. Layer names are: {found_names}"


@pytest_cases.parametrize(
    'kwargs',
    [{
        'model': nn.Sequential(nn.Linear(2, 3)),
        'rot_mat': torch.tensor([[1., -1.], [1., 1.]]) / torch.sqrt(torch.tensor(2.)),
        'rot_func': lambda tensor,
                    rot_mat,
                    K: torch.matmul(tensor, rot_mat),
        'key': '0',
        'expected_state_dict_keys': ['0.weight', '0.bias'],}])
def test_remove_parametrization_entries_state_dict(kwargs):
    key = kwargs['key']
    rot_mat = kwargs['rot_mat']
    rot_func = kwargs['rot_func']
    expected_state_dict_keys = kwargs['expected_state_dict_keys']
    del kwargs['key']
    del kwargs['rot_mat']
    del kwargs['rot_func']
    del kwargs['expected_state_dict_keys']

    model = kwargs["model"]
    module = recurse_getattr(model, key)
    old_state_dict = copy.deepcopy(model.state_dict())
    # Register rotation parametrization to module
    parametrize.register_parametrization(
        module=module,
        tensor_name="weight",
        parametrization=RotationWeightParametrization(
            rot_mat=nn.Parameter(rot_mat),
            rot_func=rot_func,
            axis=1,
            K=None,
        ))
    # Retrieve state dict after parametrization
    state_dict = model.state_dict()
    # Remove parametrization entries from state dict
    state_dict = _remove_parametrization_entries_state_dict(state_dict)
    # Verify that all the expected keys in expected_state_dict_keys
    # are present in state_dict
    assert len(set(expected_state_dict_keys) - set(state_dict.keys())) == 0
    # Verify that keys match
    for key, value in state_dict.items():
        # Verify that key is in the expected keys
        assert key in expected_state_dict_keys, f"Unexpected key {key} in state_dict"
        # Compare tensor values
        assert torch.allclose(value, old_state_dict[key], rtol=0.0, atol=0.0), f"Value of tensor {value} does not match with that in the original state_dict"


@requires_pt_ge('2.3.1')
@pytest_cases.parametrize(
    'kwargs',
    [
        {
            'model': nn.Sequential(nn.Linear(2, 3)),
            'sample_input': torch.tensor([[0.8, -0.6]]),
            'rot_mat': torch.tensor([[1., -1.], [1., 1.]]) / torch.sqrt(torch.tensor(2.)),
            'rot_func': lambda tensor,
                        rot_mat,
                        K: torch.matmul(tensor, rot_mat),
            'key': '0',
            'expected': "<class 'torch.nn.utils.parametrize.ParametrizedQuantLinear'>"},])
def test_quantize_parametrized_modules(kwargs):
    if platform.system() == "Windows":
        pytest.skip("Skipping dynamo + windows")
    key = kwargs['key']
    exp = kwargs['expected']
    rot_mat = kwargs['rot_mat']
    rot_func = kwargs['rot_func']
    sample_input = kwargs['sample_input']
    model = kwargs["model"]

    graph_model, _ = torch._dynamo.export(model)(sample_input)
    orig_module = recurse_getattr(model, key)
    # Use tied weights to identify equivalent model
    key, module = [(key, module) for key, module in graph_model.named_modules() if hasattr(module, "weight") and module.weight is orig_module.weight][0]
    # Register rotation parametrization to module
    parametrize.register_parametrization(
        module=module,
        tensor_name="weight",
        parametrization=RotationWeightParametrization(
            rot_mat=nn.Parameter(rot_mat),
            rot_func=rot_func,
            axis=1,
            K=None,
        ))
    qmodel = quantize(graph_model)
    checked = False
    found_names = []
    for n, m in qmodel.named_modules():
        found_names.append(n)
        if n == key:
            mt = str(type(m))
            assert mt == exp, f"Expect module {n} to be type: {exp}, found type {mt}"
            checked = True
    assert checked, f"Layer named {key} not found. Layer names are: {found_names}"
