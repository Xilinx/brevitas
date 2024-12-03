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

from copy import deepcopy
from itertools import product

from packaging.version import parse
import pytest
import pytest_cases
import torch
from torch.nn import Parameter
from torch.optim.lr_scheduler import LinearLR

from brevitas import torch_version
from tests.conftest import SEED
from tests.marker import requires_pt_ge

torch.manual_seed(SEED)

REFERENCE_INP = torch.tensor([[-1.8645, -0.4071, 1.1971]])
REFERENCE_WEIGHTS = torch.tensor([[1.0023, 0.0205, 1.4604], [-0.2918, -1.8218, -0.7010],
                                  [1.4573, -0.9074, -0.2708]])
REFERENCE_WEIGHTS_GRAD = torch.tensor([[1.0023, 0.000, 1.4604], [-0.2918, -1.8218, -0.7010],
                                       [1.4573, -0.9074, -0.2708]])
REFERENCE_WEIGHTS_SIGN_GRAD = torch.tensor([[1.0000, 0.0000, 1.0000], [-1.0000, -1.0000, -1.0000],
                                            [1.0000, -1.0000, -1.0000]])

OPTIMIZER_KWARGS = [{}, {"maximize": True}, {"lr": 1e-2}, {"lr": torch.tensor(0.001)}]
LR_SCHEDULER_ARGS = [
    None,
    (LinearLR, {
        "start_factor": 1.0, "end_factor": 0.0, "total_iters": 20}),]
DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
DTYPES = [torch.float16, torch.float32]

device_dtype_parametrize = pytest_cases.parametrize("device, dtype", list(product(DEVICES, DTYPES)))


class TestOptimSignSGD:

    @device_dtype_parametrize
    @pytest_cases.parametrize("lr", [0.1])
    @requires_pt_ge('2.1')  # TODO: revisit this
    def test_sign_sgd_single_update(self, device, dtype, lr):
        from brevitas.optim.sign_sgd import SignSGD

        # Initialize weights and grads
        weights = Parameter(REFERENCE_WEIGHTS.to(device=device, dtype=dtype))
        # Initialize tensors to compute expected result
        initial_weights = REFERENCE_WEIGHTS.to(device=device, dtype=dtype, copy=True)
        weight_grad = REFERENCE_WEIGHTS_GRAD.to(device=device, dtype=dtype)
        weight_sign_grad = REFERENCE_WEIGHTS_SIGN_GRAD.to(device=device, dtype=dtype)

        optimizer = SignSGD([weights], lr=lr)

        # Perform a SignSGD update
        optimizer.zero_grad()
        weights.grad = weight_grad
        optimizer.step()

        assert torch.allclose(weights, initial_weights - lr * weight_sign_grad)

    @device_dtype_parametrize
    @pytest_cases.parametrize("optimizer_kwargs", OPTIMIZER_KWARGS)
    @pytest_cases.parametrize("lr_scheduler_args", LR_SCHEDULER_ARGS)
    @requires_pt_ge('2.1')
    def test_forloop_goes_right_direction(self, device, dtype, optimizer_kwargs, lr_scheduler_args):
        from brevitas.optim.sign_sgd import SignSGD

        # PyTorch version previous to 2.3.1. might no have mv (addmv_impl_cpu) implemented for Half
        if dtype == torch.float16 and device == "cpu" and torch_version < parse('2.3.1'):
            pytest.xfail(
                "PyTorch versions previous to 2.3.1. might no have mv (addmv_impl_cpu) implemented for Half"
            )

        optim_cls = SignSGD
        weight = Parameter(torch.randn((10, 5), device=device, dtype=dtype))
        bias = Parameter(torch.randn((10), device=device, dtype=dtype))
        input = torch.randn(5, device=device, dtype=dtype)

        optimizer = optim_cls([weight, bias], **deepcopy(optimizer_kwargs))
        scheduler = None if lr_scheduler_args is None else lr_scheduler_args[0](
            optimizer, **lr_scheduler_args[1])

        def closure():
            optimizer.zero_grad()
            loss = (weight.mv(input) + bias).pow(2).sum()
            loss.backward()
            return loss

        initial_value = closure().item()
        for _ in range(20):
            closure()
            optimizer.step()
            print(bias)
            if scheduler is not None:
                scheduler.step()

            if optimizer_kwargs.get("maximize", False):
                assert closure().item() > initial_value
            else:
                assert closure().item() < initial_value

    @pytest.mark.skipif(
        torch.cuda.device_count() <= 1, reason="At least two GPUs are required for this test.")
    @pytest_cases.parametrize("optimizer_kwargs", OPTIMIZER_KWARGS)
    @pytest_cases.parametrize("lr_scheduler_args", LR_SCHEDULER_ARGS)
    @pytest_cases.parametrize("dtype", [torch.float16, torch.float32])
    @requires_pt_ge('2.1')
    def test_forloop_goes_right_direction_multigpu(
            self, dtype, optimizer_kwargs, lr_scheduler_args):
        from brevitas.optim.sign_sgd import SignSGD
        optim_cls = SignSGD
        # Learnable parameters
        weight = Parameter(torch.randn((10, 5), device="cuda:0", dtype=dtype))
        bias = Parameter(torch.randn((10), device="cuda:1", dtype=dtype))
        input = torch.randn(5, device="cuda:0", dtype=dtype)

        optimizer = optim_cls([weight, bias], **deepcopy(optimizer_kwargs))
        scheduler = None if lr_scheduler_args is None else lr_scheduler_args[0](
            optimizer, **lr_scheduler_args[1])

        def closure():
            optimizer.zero_grad()
            loss = (weight.mv(input).cuda(1) + bias).pow(2).sum()
            loss.backward()
            return loss

        initial_value = closure().item()
        for _ in range(20):
            closure()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if optimizer_kwargs.get("maximize", False):
                assert closure().item() > initial_value
            else:
                assert closure().item() < initial_value
