# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.function.ops_ste import round_ste


def stochastic_round_ste_fn(generator):

    class StochasticRoundSteFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            floor_x = torch.floor(x)
            x_diff = torch.abs(x - floor_x)
            prob = torch.bernoulli(x_diff, generator=generator)
            out = torch.where(prob.to(torch.bool), floor_x + 1., floor_x)
            return out

        @staticmethod
        def backward(ctx, x_grad):
            return x_grad

    return StochasticRoundSteFn.apply


class StochasticRoundSte(torch.nn.Module):

    def __init__(self, deterministic_inference=True, seed=None, device=None) -> None:
        super().__init__()
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)
        self.round_fn = stochastic_round_ste_fn(self.generator)
        self.deterministic_inference = deterministic_inference
        if deterministic_inference:
            self.inference_fn = round_ste
        else:
            self.inference_fn = None

    @torch.jit.ignore
    def forward(self, x):
        if self.deterministic_inference and not self.training:
            return self.inference_fn(x)
        return self.round_fn(x)
