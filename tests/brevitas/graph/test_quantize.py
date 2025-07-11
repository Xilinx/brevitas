import copy
import inspect
from itertools import chain
import platform
import random

import pytest
import pytest_cases
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from brevitas.core.scaling.runtime import StatsFromParameterScaling
from brevitas.core.scaling.standalone import ParameterFromStatsFromParameterScaling
from brevitas.graph.base import _remove_parametrization_entries_state_dict
from brevitas.graph.quantize import LAYERWISE_COMPUTE_LAYER_MAP
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.quantize import quantize
from brevitas.inject.enum import ScalingImplType
from brevitas.utils.parametrization_utils import RotationWeightParametrization
from brevitas.utils.python_utils import recurse_getattr
from tests.conftest import SEED
from tests.marker import requires_pt_ge

MIN_INT = 2
MAX_INT = 10

RNN_INPUT_SIZE = 2
RNN_HIDDEN_SIZE = 2
MHA_EMBED_DIM = 2
MHA_NUM_HEADS = 1


class QuantModelCases:

    @pytest_cases.parametrize(
        'layer_map_item',
        LAYERWISE_COMPUTE_LAYER_MAP.items(),
        ids=[f'{c.__name__}' for c in LAYERWISE_COMPUTE_LAYER_MAP.keys()])
    def case_quant_model(self, layer_map_item, request):
        # Set seeds
        torch.manual_seed(SEED)
        random.seed(SEED)

        # Change the case_id based on current value of Parameters
        pytest_cases.set_case_id(request.node.callspec.id, QuantModelCases.case_quant_model)

        torch_layer_cls, quant_layer_cls_kwargs = layer_map_item

        if quant_layer_cls_kwargs is None:
            pytest.skip(f'There is no quant layer defined for {torch_layer_cls.__name__}')

        if torch_layer_cls in (torch.nn.LSTM, torch.nn.RNN):
            layer_kwargs = {'input_size': RNN_INPUT_SIZE, 'hidden_size': RNN_HIDDEN_SIZE}
        elif torch_layer_cls in (torch.nn.MultiheadAttention,):
            layer_kwargs = {'embed_dim': MHA_EMBED_DIM, 'num_heads': MHA_NUM_HEADS}
        else:
            # Retrieve the required parameters of the __init__ method
            layer_kwargs = {}
            for name, parameter in inspect.signature(torch_layer_cls.__init__).parameters.items():
                # Check if the parameter is required
                if name != 'self' and parameter.default is inspect.Parameter.empty and parameter.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                    # If so, check the type
                    if parameter.annotation == int or (
                            parameter.annotation.__name__ == 'Union' and
                            any([arg == int for arg in parameter.annotation.__args__])):
                        layer_kwargs[name] = random.randint(a=MIN_INT, b=MAX_INT)
                    else:
                        pytest.skip(
                            f"No strategy defined to populate the parameter {name} of {torch_layer_cls.__name__}."
                        )

        # Dummy model for testing
        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.layer = torch_layer_cls(**layer_kwargs)

            def forward(self, *args, **kwargs):
                return self.layer(*args, **kwargs)

        model = Model()
        return model


@pytest_cases.parametrize_with_cases('model', cases=QuantModelCases)
@pytest_cases.parametrize(
    'weight_scaling_impl_type', [ScalingImplType.STATS, ScalingImplType.PARAMETER_FROM_STATS])
def test_layerwise_quantize_quant_model(model, weight_scaling_impl_type, current_cases):
    torch_layer = model.layer
    torch_layer_cls = type(torch_layer)
    # Set the scaling type of weight_quant according to the parametrized value
    layer_map = copy.deepcopy(LAYERWISE_COMPUTE_LAYER_MAP)
    if 'weight_quant' in layer_map[torch_layer_cls][1]:
        layer_map[torch_layer_cls][1]['weight_quant'] = layer_map[torch_layer_cls][1][
            'weight_quant'].let(scaling_impl_type=weight_scaling_impl_type)

    # Replace torch layers by quant layers
    qmodel = layerwise_quantize(model, compute_layer_map=layer_map)

    # Verify that the layer was replaced correctly by the correct class
    assert type(qmodel.layer) == layer_map[torch_layer_cls][0]
    # Verify that all parameters and buffers were moved from the "meta" device
    assert all([
        param.device != torch.device("meta")
        for param in chain(qmodel.parameters(), qmodel.buffers())])
    # Verify that the common parameters of the torch and quant modules share the same storage position
    assert all([
        param.storage().data_ptr() == recurse_getattr(qmodel, name).storage().data_ptr() for name,
        param in chain(qmodel.named_parameters(), qmodel.named_buffers())])


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
