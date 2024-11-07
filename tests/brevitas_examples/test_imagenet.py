# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import pytest
import pytest_cases
from pytest_cases import fixture
import torch
import torch.nn as nn

from brevitas_examples.imagenet_classification.ptq.utils import get_torchvision_model

DTYPE = torch.float32


class TestImageNet:

    @fixture
    def model():
        # Get the model from torchvision
        model = get_torchvision_model("resnet18")
        model = model.to(DTYPE)
        model.eval()

        return model

    def test_model_can_be_loaded(model):
        print(f"The model class IS: {type(model)}")
        assert False


if __name__ == "__main__":
    # Run pytest on the current file
    pytest.main(["-s", __file__])
