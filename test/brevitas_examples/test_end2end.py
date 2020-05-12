import logging
from common import mnist_datapath_fixture, set_and_evaluate_hooks_mnist, mnist_datapath
import pytest
import torch
from brevitas_examples.bnn_pynq.models import model_with_cfg

from torchvision.datasets import MNIST
from torchvision.transforms import transforms
import numpy as np
from itertools import product

@pytest.mark.parametrize("model", ["TFC", "SFC", "LFC"])
@pytest.mark.parametrize("weight_bit_width", [1, 2])
@pytest.mark.parametrize("act_bit_width", [1, 2])
def test_bnn_pynq_fc_integration_tests(model, weight_bit_width, act_bit_width, mnist_datapath):
    if model == "LFC" and weight_bit_width == 2 and act_bit_width == 2:
        pytest.skip("No pretrained LFC_W2A2 present.")
    if weight_bit_width > act_bit_width:
        pytest.skip("No weight_bit_width > act_bit_width cases.")

    network = f"{model}_{weight_bit_width}W{act_bit_width}A"

    # Load local files, to be replaced with download from release
    expected_results = np.load(network + "_integration_tests.npz")

    keys = expected_results.files
    transform_to_tensor = transforms.Compose([transforms.ToTensor()])
    test_set = MNIST(root=mnist_datapath,
                     train=False,
                     download=True,
                     transform=transform_to_tensor)
    model, cfg = model_with_cfg(network, pretrained=True)
    hooks, full_dictionary = set_and_evaluate_hooks_mnist(model, test_set)

    result = True
    for key in keys:
        is_scalefactor = key.split('.')[-1] == 'scale_factor'
        if is_scalefactor:  # if scale factors are comapared, a tolerance must be taken in account
            ATOL = 1e-8
            RTOL = 1e-8
        else:
            ATOL = 0
            RTOL = 0
        tensor_result = torch.tensor(expected_results[key])
        result = result and torch.allclose(tensor_result, full_dictionary[key], ATOL, RTOL)

    assert result


def generate_file():
    all_model = ["TFC", "SFC", "LFC"]
    all_weight_bit_width = [1, 2]
    all_act_bit_width = [1, 2]
    combinations = [all_model, all_weight_bit_width, all_act_bit_width]
    all_permutations = list(product(*combinations))
    mnist_path = mnist_datapath()
    for model, weight_bit_width, act_bit_width in all_permutations:
        if model == "LFC" and weight_bit_width == 2 and act_bit_width == 2:
            pytest.skip("No pretrained LFC_W2A2 present.")
        if weight_bit_width > act_bit_width:
            pytest.skip("No weight_bit_width > act_bit_width cases.")

        network = f"{model}_{weight_bit_width}W{act_bit_width}A"

        transform_to_tensor = transforms.Compose([transforms.ToTensor()])
        test_set = MNIST(root=mnist_path,
                         train=False,
                         download=True,
                         transform=transform_to_tensor)
        model, cfg = model_with_cfg(network, pretrained=True)
        hooks, full_dictionary = set_and_evaluate_hooks(model, test_set, 'mnist')
        filename = network + "_integration_tests"
        np.savez(filename, **full_dictionary)


if __name__ == '__main__':
    generate_file()

