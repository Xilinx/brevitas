# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
import functools
from functools import partial
from typing import Dict, List, Optional, Union
from unittest.mock import patch

from accelerate import dispatch_model
import pytest
import pytest_cases
from pytest_cases import fixture
import torch
from torch import nn

from brevitas_examples.common.accelerate_utils.accelerate import \
    attach_update_state_dict_hook_on_modules
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.common.accelerate_utils.accelerate import update_internal_dict


@dataclass
class ModelDataClass:
    model_class: type[nn.Module]
    output: torch.Tensor
    block1_layer1_parameter: torch.Tensor
    preload_module_classes: List

class TestTiedLayer1(nn.Module):
    def __init__(self, parameter: torch.Tensor):
        super().__init__()
        self.parameter = parameter
        self.w = nn.Parameter(torch.tensor([2.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.parameter.detach().add_(x)
        return self.w*x

class TestTiedBlock1(nn.Module):
    def __init__(self):
        super().__init__()
        self.tied_parameter = nn.Parameter(torch.tensor([1.0]))
        self.layer1 = TestTiedLayer1(self.tied_parameter)
        self.layer2 = TestTiedLayer1(self.tied_parameter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        out = self.layer2(out)
        return out

class TestBlock1(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w*x

class TestModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = TestTiedBlock1()
        self.block2 = TestBlock1()

        self._no_split_modules = None

    def forward(self, x: torch.Tensor):
        out = self.block1(x)
        out = self.block2(out)
        return out

class TestTiedLayer2(nn.Module):
    def __init__(self, parameter: torch.Tensor):
        super().__init__()
        self.parameter = parameter
        self.w = nn.Parameter(torch.tensor([2.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.parameter.detach().add_(x)
        return self.w*x

class TestTiedBlock2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = TestTiedLayer2(nn.Parameter(torch.tensor([1.0])))
        self.layer2 = TestTiedLayer2(self.layer1.parameter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer1(x)
        self.layer1.parameter.detach().add_(x)
        out = self.layer2(out)
        return out

class TestBlock2(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w*x

class TestModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = TestTiedBlock2()
        self.block2 = TestBlock2()

        self._no_split_modules = None

    def forward(self, x: torch.Tensor):
        out = self.block1(x)
        out = self.block2(out)
        return out

def dispatch_model_with_preload(
        model: nn.Module,
        device_map: Dict[str, Union[str, int, torch.device]],
    ):
        return dispatch_model(
            model = model,
            device_map = device_map,
            preload_module_classes=["TestTiedBlock2"]
        )


class TestAccelerate:

    marker_model_dataclass = pytest_cases.parametrize(
        "model_dataclass", [
            ModelDataClass(
                model_class=TestModel1,
                output=torch.tensor([24.0]),
                block1_layer1_parameter=torch.tensor([7.0]),
                preload_module_classes=[],
            ),
            ModelDataClass(
                model_class=TestModel2,
                output=torch.tensor([24.0]),
                block1_layer1_parameter=torch.tensor([9.0]),
                preload_module_classes=["TestTiedBlock2"],
            )
        ]
    )

    marker_device_map = pytest_cases.parametrize("device_map", [{"": 0}, {"": "cpu"}, {"block1": "cpu", "block2": 0}])

    @pytest.mark.xfail
    @marker_model_dataclass
    @marker_device_map
    def test_accelerate_inplace_operation(self, model_dataclass, device_map):
        with patch(
            'brevitas_examples.common.accelerate_utils.accelerate.infer_auto_device_map',
            return_value=device_map,
        ) as mock_infer:
            test_model = model_dataclass.model_class()
            test_model = offload_model(test_model, preload_module_classes=model_dataclass.preload_module_classes)
            # Run forward pass through model
            out = test_model(torch.tensor([2.0])).cpu()
            # Hooks are removed and model moved to CPU, thus enabling
            # to access the model parameters easily
            remove_hooks(test_model)
            # Verify that the mocked method was called once
            mock_infer.assert_called_once()
            # Verify that output is the expected
            assert torch.allclose(out, model_dataclass.output)
            # Verify that the inplace operations were performed correctly
            assert torch.allclose(test_model.block1.layer1.parameter.detach(), model_dataclass.block1_layer1_parameter)

    @pytest.mark.xfail
    @marker_model_dataclass
    @marker_device_map
    def test_accelerate_inplace_operation_post_forward_fix(self, model_dataclass, device_map):
        with patch(
            'brevitas_examples.common.accelerate_utils.accelerate.infer_auto_device_map',
            return_value=device_map,
        ) as mock_infer:
            test_model = model_dataclass.model_class()
            test_model = offload_model(test_model, preload_module_classes=model_dataclass.preload_module_classes)

            dict_of_hooks = dict()
            def hooked_on_a_function(function, prefunction):
                @functools.wraps(function)
                def run(*args, **kwargs):
                    prefunction()
                    return function(*args, **kwargs)
                return run

            def update_params_post_init(module):
                update_internal_dict(module)

            for m in test_model.modules():
                if hasattr(m, '_hf_hook'):
                    if m._hf_hook.weights_map is not None:
                        dict_of_hooks[m] = m._hf_hook.post_forward
                        new_funct = partial(update_params_post_init, m)
                        m._hf_hook.post_forward = hooked_on_a_function(m._hf_hook.post_forward, new_funct)

            # Run forward pass through model
            out = test_model(torch.tensor([2.0])).cpu()
            # Hooks are removed and model moved to CPU, thus enabling
            # to access the model parameters easily
            for k, v in dict_of_hooks.items():
                k._hf_hook.post_forward = v

            remove_hooks(test_model)

            # Verify that the mocked method was called once
            mock_infer.assert_called_once()
            # Verify that output is the expected
            assert torch.allclose(out, model_dataclass.output)
            # Verify that the inplace operations were performed correctly
            assert torch.allclose(test_model.block1.layer1.parameter.detach(), model_dataclass.block1_layer1_parameter)

    @marker_model_dataclass
    @marker_device_map
    def test_accelerate_inplace_operation_hook_fix(self, model_dataclass, device_map):
        with (
            patch('brevitas_examples.common.accelerate_utils.accelerate.infer_auto_device_map', return_value=device_map) as mock_infer,
        ):
            test_model = model_dataclass.model_class()

            test_model = offload_model(test_model, preload_module_classes=model_dataclass.preload_module_classes)
            # Verify that the mocks were called
            mock_infer.assert_called_once()

            attach_update_state_dict_hook_on_modules(test_model)

            # Run forward pass through model
            out = test_model(torch.tensor([2.0])).cpu()

            remove_hooks(test_model)

            # Verify that the mocked method was called once
            mock_infer.assert_called_once()
            # Verify that output is the expected
            assert torch.allclose(out, model_dataclass.output)
            # Verify that the inplace operations were performed correctly
            assert torch.allclose(test_model.block1.layer1.parameter.detach(), model_dataclass.block1_layer1_parameter)
