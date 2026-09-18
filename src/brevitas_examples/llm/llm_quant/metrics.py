# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod

import torch
from torch import nn


class MetricBase(ABC):

    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype

    @abstractmethod
    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric state with one evaluation chunk."""
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> float:
        """Return the final metric value."""
        raise NotImplementedError


class Perplexity(MetricBase):

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        self.nlls = []

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        self.nlls.append(
            nn.functional.cross_entropy(output.reshape(-1, output.shape[-1]), target.reshape(-1)))

    def finalize(self) -> float:
        nlls = torch.stack(self.nlls).to(dtype=self.dtype)
        return torch.exp(nlls.mean()).item()


class EAR(MetricBase):
    """
    Expected Acceptance Rate (EAR) as proposed in https://arxiv.org/pdf/2605.02404
    """

    def __init__(self, normalize: bool = True, dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        self.ear_sum = 0.0
        self.num_tokens = 0.0  # total number of tokens
        # Normalize by the reference top-K probability mass when requested
        self.normalize = normalize

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        output = output.to(dtype=self.dtype)
        target = target.to(dtype=self.dtype)
        if self.normalize:
            reference_mass = target.sum(dim=-1, keepdim=True)
            output = output / reference_mass
            target = target / reference_mass
        self.ear_sum += torch.minimum(output, target).sum().item()
        self.num_tokens += target.numel() // target.shape[-1]

    def finalize(self) -> float:
        return self.ear_sum / self.num_tokens


class KLD(MetricBase):

    def __init__(self, normalize: bool = True, dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        self.kld_sum = 0.0
        self.num_tokens = 0.0  # total number of tokens
        # Normalize by the reference top-K probability mass when requested
        self.normalize = normalize

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        output = output.to(dtype=self.dtype)
        target = target.to(dtype=self.dtype)
        if self.normalize:
            reference_mass = target.sum(dim=-1, keepdim=True)
            output = output / reference_mass
            target = target / reference_mass
        self.kld_sum += (target * (target.log() - output.log())).sum().item()
        self.num_tokens += target.numel() // target.shape[-1]

    def finalize(self) -> float:
        return self.kld_sum / self.num_tokens
