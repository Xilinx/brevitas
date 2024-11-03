import torch
import torch.nn as nn
import pytest

from brevitas.nn.equalized_layer import RotationEqualizedLayer

class TestRotationEqualizedLayer:

    @pytest.fixture
    def setup(self):
        layer = nn.Linear(10, 10)
        rotation_matrix1 = torch.eye(10)
        rotation_matrix2 = torch.eye(10)
        return RotationEqualizedLayer(layer, rotation_matrix1, rotation_matrix2)

    def test_forward_pass(self, setup):
        rotation_layer = setup
        input_tensor = torch.randn(1, 10)
        output_tensor = rotation_layer(input_tensor)
        assert output_tensor.shape == (1, 10)

    def test_fuse_rotation_matrices(self, setup):
        rotation_layer = setup
        rotation_layer.fuse_rotation_matrices()
        assert torch.allclose(rotation_layer.layer.weight, torch.eye(10))
        if rotation_layer.layer.bias is not None:
            assert torch.allclose(rotation_layer.layer.bias, torch.zeros(10))
