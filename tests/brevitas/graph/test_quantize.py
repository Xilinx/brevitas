import pytest_cases
import torch.nn as nn

from brevitas.graph.quantize import layerwise_quantize


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
