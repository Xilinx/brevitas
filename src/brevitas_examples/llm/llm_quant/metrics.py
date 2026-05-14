# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
import math
from typing import Optional

import torch
from torch import nn


class MetricBase(ABC):
    """Base class accumulating per-token running moments.

    Subclasses feed one value per token to ``_accumulate``. The first two
    moments are enough to report both the metric and its standard error:

        mean  = sum / count
        var   = sum_sq / count - mean ** 2
        sigma = sqrt(var / (count - 1))

    Moments are accumulated in float64 (double precision).
    """

    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0

    def _accumulate(self, values: torch.Tensor) -> None:
        values = values.detach().to(dtype=torch.float64).flatten()
        self.count += values.numel()
        self.sum += values.sum().item()
        self.sum_sq += values.square().sum().item()

    @property
    def _mean(self) -> float:
        return self.sum / self.count

    def _standard_error(self) -> Optional[float]:
        if self.count < 2:
            return None
        variance = self.sum_sq / self.count - self._mean ** 2
        # A non-positive variance is numerically meaningless, and so is the
        # standard error derived from it.
        if variance <= 0:
            return None
        return math.sqrt(variance / (self.count - 1))

    @abstractmethod
    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric state with one evaluation chunk."""
        raise NotImplementedError

    def finalize(self) -> float:
        """Return the final metric value."""
        return self._mean

    def finalize_std(self) -> Optional[float]:
        """Return the standard error of the metric, or None if undefined."""
        return self._standard_error()


class Perplexity(MetricBase):

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        nll = nn.functional.cross_entropy(
            output.reshape(-1, output.shape[-1]), target.reshape(-1), reduction="none")
        self._accumulate(nll.to(dtype=self.dtype))

    def finalize(self) -> float:
        return math.exp(self._mean)

    def finalize_std(self) -> Optional[float]:
        standard_error = self._standard_error()
        if standard_error is None:
            return None
        # Delta method: std(exp(x)) ~= exp(mean(x)) * std(x)
        return math.exp(self._mean) * standard_error


class EAR(MetricBase):
    """
    Expected Acceptance Rate (EAR) as proposed in https://arxiv.org/pdf/2605.02404
    """

    def __init__(self, normalize: bool = True, dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        # Normalize by the reference top-K probability mass when requested
        self.normalize = normalize

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        output = output.to(dtype=self.dtype)
        target = target.to(dtype=self.dtype)
        if self.normalize:
            reference_mass = target.sum(dim=-1, keepdim=True)
            output = output / reference_mass
            target = target / reference_mass
        self._accumulate(torch.minimum(output, target).sum(dim=-1))


class KLD(MetricBase):

    def __init__(self, normalize: bool = True, dtype: torch.dtype = torch.float32):
        super().__init__(dtype=dtype)
        # Normalize by the reference top-K probability mass when requested
        self.normalize = normalize

    def update(self, output: torch.Tensor, target: torch.Tensor) -> None:
        output = output.to(dtype=self.dtype)
        target = target.to(dtype=self.dtype)
        if self.normalize:
            reference_mass = target.sum(dim=-1, keepdim=True)
            output = output / reference_mass
            target = target / reference_mass
        self._accumulate((target * (target.log() - output.log())).sum(dim=-1))
