from torchvision.datasets import MNIST
import pytest
import os
import brevitas.nn as qnn
import torch
from functools import partial
from brevitas.nn.quant_activation import QuantActivation
from brevitas.nn.quant_layer import QuantLayer

weights = {
    'integer_values': [],
    'scale_factor': [],
    'output_tensor': []
}
activations = {
    'output_integers': [],
    'scale_factor': []
}

full_dictionary = {}

mnist_label_sample = [(7, 0), (2, 1), (1, 2), (0, 3), (4, 4), (9, 7), (5, 8), (6, 11), (3, 18), (8, 61)]
# weight_layers = [qnn.QuantLinear, qnn.QuantConv2d, qnn.QuantConv1d, qnn.QuantConvTranspose1d]
# activation_layers = [qnn.QuantTanh, qnn.QuantReLU, qnn.QuantHardTanh, qnn.QuantSigmoid, qnn.QuantIdentity]

def mnist_datapath():
    cwd = os.getcwd()
    datapath = os.path.join(cwd, "data")
    _ = MNIST(root=datapath, download=True)
    return datapath

@pytest.fixture(name='mnist_datapath')
def mnist_datapath_fixture():
    return mnist_datapath()

def set_and_evaluate_hooks(model, data_loader, dataset ):
    hooks = []
    for name, module in model.named_modules():
        a = register_hook(module, name)
        if a is not None:
            hooks.append(a)
    print("Ciao")
    model.eval()
    if dataset == 'mnist':
        samples = mnist_label_sample
    with torch.no_grad():
        for label, sample in samples:
            x, y = data_loader[sample]
            model(x)
    return hooks, full_dictionary


def register_hook(model, name):
    if isinstance(model, QuantActivation):
        dlb = partial(activations_hook_fn, name=name)
        activations_hook = model.register_forward_hook(dlb)
        return activations_hook
    elif isinstance(model, QuantLayer):
        dlb = partial(weights_hook_fn, name=name)
        weights_hook = model.register_forward_hook(dlb)
        return weights_hook
    return


def weights_hook_fn(module, input, output, name='None'):
    full_name = '.'.join(['weights', name, 'integer_values'])
    full_dictionary[full_name] = module.int_weight
    full_name = '.'.join(['weights', name, 'scale_factor'])
    full_dictionary[full_name] = module.quant_weight_scale
    full_name = '.'.join(['weights', name, 'output_tensor'])
    full_dictionary[full_name] = output


def activations_hook_fn(module, input, output, name='None'):
    full_name = '.'.join(['activations', name, 'scale_factor'])
    scale = module.quant_act_scale()
    output_int = torch.round(output/scale)
    full_dictionary[full_name] = scale
    full_name = '.'.join(['activations', name, 'output_integers'])
    full_dictionary[full_name] = output_int
