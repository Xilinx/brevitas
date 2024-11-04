"""
Copyright (C) 2024,     Advanced Micro Devices, Inc.
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of AMD, Facebook, Deepmind Technologies, NYU,
   NEC Laboratories America and IDIAP Research Institute nor the names
   of its contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import math
import sys
from typing import List, Union
import unittest

from hypothesis import given
import pytest
import pytest_cases
from pytest_cases import fixture
import torch
from torch.nn import Parameter
import torch.nn as nn
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import PolynomialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_device_type import onlyCUDA
from torch.testing._internal.common_device_type import tol
from torch.testing._internal.common_device_type import toleranceOverride
from torch.testing._internal.common_optimizers import DecorateInfo
from torch.testing._internal.common_optimizers import optim_error_inputs_func_sgd
from torch.testing._internal.common_optimizers import optim_inputs_func_sgd
from torch.testing._internal.common_optimizers import OptimizerErrorEnum
from torch.testing._internal.common_optimizers import OptimizerInfo
from torch.testing._internal.common_optimizers import optims
from torch.testing._internal.common_optimizers import skipIfTorchDynamo
from torch.testing._internal.common_utils import markDynamoStrictTest
from torch.testing._internal.common_utils import parametrize
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_utils import skipIfTorchDynamo
from torch.testing._internal.common_utils import TEST_WITH_TORCHDYNAMO
from torch.testing._internal.common_utils import TestCase

from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.calibrate import disable_return_quant_tensor
from brevitas.graph.calibrate import DisableEnableQuantization
from brevitas.graph.calibrate import load_quant_model_mode
from brevitas.graph.calibrate import restore_return_quant_tensor
from brevitas.inject.enum import RestrictValueType
import brevitas.nn as qnn
from brevitas.optim.sign_sgd import SignSGD
from brevitas.quant import Int8ActPerTensorFixedPoint
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant_tensor import QuantTensor
# Use custom implementation of kthvalue as work around to (b)float16 kernel limitations
from brevitas.utils.torch_utils import kthvalue
from tests.brevitas.hyp_helper import float_tensor_random_size_st
from tests.conftest import SEED

torch.manual_seed(SEED)

REFERENCE_INP = torch.tensor([[-1.8645, -0.4071, 1.1971]])
REFERENCE_WEIGHTS = torch.tensor([[1.0023, 0.0205, 1.4604], [-0.2918, -1.8218, -0.7010],
                                  [1.4573, -0.9074, -0.2708]])
REFERENCE_WEIGHTS_GRAD = torch.tensor([[1.0023, 0.000, 1.4604], [-0.2918, -1.8218, -0.7010],
                                       [1.4573, -0.9074, -0.2708]])
REFERENCE_WEIGHTS_SIGN_GRAD = torch.tensor([[1.0000, 0.0000, 1.0000], [-1.0000, -1.0000, -1.0000],
                                            [1.0000, -1.0000, -1.0000]])

optim_db: List[OptimizerInfo] = [
    OptimizerInfo(
        SignSGD,
        optim_inputs_func=optim_inputs_func_sgd,
        scheduler_inputs=(
            [lambda opt: StepLR(opt, gamma=0.9, step_size=10)],
            [lambda opt: LinearLR(opt, start_factor=0.4, end_factor=0.8, total_iters=4)],
            [
                lambda opt: StepLR(opt, gamma=0.9, step_size=10),
                lambda opt: LinearLR(opt, start_factor=0.4, end_factor=0.6, total_iters=4),],
            [
                lambda opt: StepLR(opt, gamma=0.99, step_size=10),
                lambda opt: ExponentialLR(opt, gamma=0.99),
                lambda opt: ReduceLROnPlateau(opt),],
            [lambda opt: ConstantLR(opt, factor=0.4, total_iters=4)],
            [lambda opt: PolynomialLR(opt, power=0.9, total_iters=4)],
            [
                lambda opt: StepLR(opt, gamma=0.9, step_size=10),
                lambda opt: ReduceLROnPlateau(opt),],
        ),
        optim_error_inputs_func=optim_error_inputs_func_sgd,
        supported_impls=("foreach", "differentiable", "fused"),
        supports_sparse=True,
        metadata_for_sparse=(
            {
                "lr": 4.8e-3,
                "maximize": False,
                "momentum": 0,
                "nesterov": False,
                "weight_decay": 0,},
            [lambda opt: StepLR(opt, gamma=0.99999, step_size=300)],
        ),
        supports_fused_on=(
            "cpu",
            "cuda",
            "mps",
        ),
        skips=(),
    ),]


@markDynamoStrictTest
class TestOptimSignSGD(TestCase):

    @parametrize("lr", [0.1])
    @optims(optim_db, dtypes=[torch.float32])
    def test_sign_sgd_update(self, device, dtype, optim_info, lr):
        optim_cls = optim_info.optim_cls
        # Initialize weights and grads
        weights = Parameter(REFERENCE_WEIGHTS.to(device=device, dtype=dtype))
        # Initialize tensors to compute expected result
        initial_weights = REFERENCE_WEIGHTS.to(device=device, dtype=dtype, copy=True)
        weight_grad = REFERENCE_WEIGHTS_GRAD.to(device=device, dtype=dtype)
        weight_sign_grad = REFERENCE_WEIGHTS_SIGN_GRAD.to(device=device, dtype=dtype)

        optimizer = optim_cls([weights], lr=lr)

        # Perform a SignSGD update
        optimizer.zero_grad()
        weights.grad = weight_grad
        optimizer.step()

        assert torch.allclose(weights, initial_weights - lr * weight_sign_grad)

    @optims([optim for optim in optim_db if optim.optim_error_inputs_func is not None],
            dtypes=[torch.float32])
    def test_errors(self, device, dtype, optim_info):
        optim_cls = optim_info.optim_cls
        error_inputs = optim_info.optim_error_inputs_func(device=device, dtype=dtype)

        for error_input in error_inputs:
            optim_input = error_input.optimizer_error_input
            params, kwargs = optim_input.params, optim_input.kwargs
            if error_input.error_on == OptimizerErrorEnum.CONSTRUCTION_ERROR:
                if issubclass(error_input.error_type, Warning):
                    with self.assertWarnsRegex(error_input.error_type, error_input.error_regex):
                        optim_cls(params, **kwargs)
                else:
                    with self.assertRaisesRegex(error_input.error_type, error_input.error_regex):
                        optim_cls(params, **kwargs)
            elif error_input.error_on == OptimizerErrorEnum.STEP_ERROR:
                optim = optim_cls(params, **kwargs)
                if issubclass(error_input.error_type, Warning):
                    with self.assertWarnsRegex(error_input.error_type, error_input.error_regex):
                        optim.step()
                else:
                    with self.assertRaisesRegex(error_input.error_type, error_input.error_regex):
                        optim.step()
            else:
                raise NotImplementedError(f"Unknown error type {error_input.error_on}")

    @parametrize("contiguous", [True, False])
    @parametrize("with_lrsched", [True, False])
    @optims(optim_db, dtypes=[torch.float32])
    def test_forloop_goes_right_direction(
            self, device, dtype, optim_info, contiguous, with_lrsched):
        optim_cls = optim_info.optim_cls
        schedulers_constructors = (optim_info.scheduler_inputs if with_lrsched else [None])

        for schedulers_constructor in schedulers_constructors:
            # with tensor LR we need fresh inputs for each scheduler
            # or mutating it will carry across iters
            optim_inputs = optim_info.optim_inputs_func(device=device)
            for optim_input in optim_inputs:
                if "foreach" in optim_info.supported_impls:
                    optim_input.kwargs["foreach"] = False  # force forloop
                if contiguous:
                    weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
                    bias = Parameter(torch.randn((10), device=device, dtype=dtype))
                else:
                    weight = Parameter(torch.randn((10, 5, 2), device=device, dtype=dtype)[..., 0])
                    bias = Parameter(torch.randn((10, 2), device=device, dtype=dtype)[..., 0])
                input = torch.randn(5, device=device, dtype=dtype)

                optimizer = optim_cls([weight, bias], **optim_input.kwargs)
                schedulers = [
                    s(optimizer)
                    for s in (schedulers_constructor if schedulers_constructor else [])]

                def closure():
                    optimizer.zero_grad()
                    loss = (weight.mv(input) + bias).pow(2).sum()
                    loss.backward()
                    if optim_info.only_supports_sparse_grads:
                        # For this test, we naively convert the Tensor layout, which we know does
                        # NOT represent the expected use case for optims like SparseAdam!
                        weight.grad = weight.grad.to_sparse()
                        bias.grad = bias.grad.to_sparse()
                    return loss

                initial_value = closure().item()
                for _ in range(20):
                    if optim_info.step_requires_closure:
                        loss = optimizer.step(closure)
                    else:
                        loss = closure()
                        optimizer.step()

                    for scheduler in schedulers:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(loss)
                        else:
                            scheduler.step()

                if optim_input.kwargs.get("maximize", False):
                    self.assertGreater(closure().item(), initial_value)
                else:
                    self.assertLess(closure().item(), initial_value)

    @onlyCUDA
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @parametrize("with_lrsched", [True, False])
    @optims(optim_db, dtypes=[torch.float32])
    def test_forloop_goes_right_direction_multigpu(self, device, dtype, optim_info, with_lrsched):
        optim_cls = optim_info.optim_cls
        schedulers_constructors = (optim_info.scheduler_inputs if with_lrsched else [None])
        for schedulers_constructor in schedulers_constructors:
            # We need a fresh set of inputs if we have a tensor LR
            # to not carry mutations across iterations.
            optim_inputs = optim_info.optim_inputs_func(device=device)
            for optim_input in optim_inputs:
                if "foreach" in optim_info.supported_impls:
                    optim_input.kwargs["foreach"] = False  # force forloop

                weight = Parameter(torch.randn((10, 5), device="cuda:0", dtype=dtype))
                bias = Parameter(torch.randn((10), device="cuda:1", dtype=dtype))
                inpt = torch.randn(5, device="cuda:0", dtype=dtype)

                optimizer = optim_cls([weight, bias], **optim_input.kwargs)
                schedulers = [
                    s(optimizer)
                    for s in (schedulers_constructor if schedulers_constructor else [])]

                def closure():
                    optimizer.zero_grad()
                    loss = (weight.mv(inpt).cuda(1) + bias).pow(2).sum()
                    loss.backward()
                    if optim_info.only_supports_sparse_grads:
                        # For this test, we naively convert the Tensor layout, which we know does
                        # NOT represent the expected use case for optims like SparseAdam!
                        weight.grad = weight.grad.to_sparse()
                        bias.grad = bias.grad.to_sparse()
                    return loss

                initial_value = closure().item()
                for _ in range(20):
                    loss = optimizer.step(closure)
                    for scheduler in schedulers:
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(loss)
                        else:
                            scheduler.step()

                if optim_input.kwargs.get("maximize", False):
                    self.assertGreater(closure().item(), initial_value)
                else:
                    self.assertLess(closure().item(), initial_value)


instantiate_device_type_tests(TestOptimSignSGD, globals(), allow_mps=True)

if __name__ == "__main__":
    run_tests()
