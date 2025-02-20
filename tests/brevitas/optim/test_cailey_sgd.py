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

from hypothesis import given
import numpy as np
from packaging import version
import pytest
import pytest_cases
from pytest_cases import fixture
from scipy.stats import ortho_group
import torch
from torch.nn import Parameter
from torch.optim.lr_scheduler import LinearLR

from brevitas import torch_version
from brevitas.optim.cailey_sgd import CaileySGD
from tests.conftest import SEED

torch.manual_seed(SEED)

OPTIMIZER_KWARGS = [{
    "stiefel": True}, {
        "stiefel": True, "lr": 0.5}, {
            "stiefel": True, "lr": torch.tensor(0.5)}]
LR_SCHEDULER_ARGS = [
    None,
    (LinearLR, {
        "start_factor": 1.0, "end_factor": 0.0, "total_iters": 20}),]
DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
DTYPES = ["float32", "float16", "bfloat16"]

device_dtype_parametrize = pytest_cases.parametrize("device, dtype", list(product(DEVICES, DTYPES)))


class TestCaileySGD:

    @device_dtype_parametrize
    @pytest_cases.parametrize("optimizer_kwargs", OPTIMIZER_KWARGS)
    @pytest_cases.parametrize("lr_scheduler_args", LR_SCHEDULER_ARGS)
    def test_forloop_goes_right_direction(self, device, dtype, optimizer_kwargs, lr_scheduler_args):
        if torch_version < version.parse('2.3.1') and dtype in ["float16", "bfloat16"]:
            pytest.skip(
                "Some operations in the CaileySGD optimizer (e.g. diag, eye) are not implemented for 'Half' or 'BFloat16' in PyTorch versions under 2.3.1."
            )
        torch.manual_seed(SEED)
        optim_cls = CaileySGD
        dtype = getattr(torch, dtype)
        # Generate a random orthogonal matrix of size NxN. Columns represent orthonormal vector in R^{N}
        N = 5
        P = 3
        weight_orthogonal = ortho_group(dim=N, seed=SEED).rvs()
        weight_orthonormal = weight_orthogonal / np.linalg.norm(weight_orthogonal, ord=2, axis=0)
        # Verify that the matrix is orthonormal
        assert np.allclose(np.matmul(weight_orthonormal.T, weight_orthonormal), np.eye(N))
        # Initialize weights, the Cailey SGD optimizer expects a matrix of size PxN, given the
        # condition unity.size()[0] <= unity.size()[1]
        weight = Parameter(
            torch.from_numpy(weight_orthonormal[:, :P].T).to(device=device, dtype=dtype))

        optimizer = optim_cls([weight], **deepcopy(optimizer_kwargs))
        scheduler = None if lr_scheduler_args is None else lr_scheduler_args[0](
            optimizer, **lr_scheduler_args[1])

        def closure():
            optimizer.zero_grad()
            # MSE between the weights and a set of orthonormal vectors
            loss = (weight - torch.eye(N, P, device=device, dtype=dtype).t()).pow(2).sum()
            loss.backward()
            return loss

        initial_value = closure().item()
        ATOL = 1e-2 if dtype == torch.float32 else 1e-1
        RTOL = 1e-3 if dtype == torch.float32 else 1e-2
        for _ in range(20):
            closure()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Verify that iterates stay within the Stiefel manifold
            assert torch.allclose(
                weight.to(dtype=torch.float32).detach().cpu()
                @ weight.to(dtype=torch.float32).detach().cpu().t(),
                torch.eye(P, P, device=device, dtype=torch.float32).detach().cpu(),
                atol=ATOL,
                rtol=RTOL)

            if optimizer_kwargs.get("maximize", False):
                assert closure().item() > initial_value
            else:
                assert closure().item() < initial_value
