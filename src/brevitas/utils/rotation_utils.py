# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable, List, Optional

import torch
from torch import nn
from torch import Tensor


class RotationWeightParametrization(torch.nn.Module):
    r"""Rotates a tensor by a specified axis

    Args:
        rot_mat (Tensor): orthogonal matrix by which to rotate the tensor
        rot_func (Callable): function to apply the rotation. The first
            argument corresponds to the tensor to be rotated, while the
            second specifies the rotation matrix. The third argument (K) is
            useful when rotating by an Hadamard matrix and it corresponds
            to the dimensionality of the matrix up to a power of two,
            i.e. dim=(2**p)*K. See brevitas.graph.hadamard.get_hadK for details
        axis (int): axis by which to rotate the tensor
        K (int, optional): if rot_mat is an Hadamard matrix, K is the highest
            divisor of the dimensionality of the matrix, such that K, itself,
            is not divisible by 2
    """

    def __init__(
        self,
        rot_mat: Callable[[Tensor, Tensor, Optional[int]], Tensor],
        rot_func: Callable,
        axis: int,
        K: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.rot_mat = rot_mat
        self.rot_func = rot_func
        self.axis = axis
        self.K = K

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:

        if self.axis == 0:
            tensor = self.rot_func(tensor.t(), self.rot_mat, self.K).t()
        elif self.axis == 1:
            tensor = self.rot_func(tensor, self.rot_mat, self.K)
        else:
            raise RuntimeError("Not supported yet")

        return tensor


def extract_trainable_rotation_matrices(model: nn.Module) -> List[nn.Parameter]:
    trainable_rotations = []
    # IDs of the rotation matrices are tracked, as several modules can share
    # the same parametrized rotation
    ids_rot = set()
    for module in model.modules():
        if isinstance(module, RotationWeightParametrization):
            if id(module.rot_mat) not in ids_rot:
                ids_rot.add(id(module.rot_mat))
                trainable_rotations.append(module.rot_mat)
    return trainable_rotations


class ScaleWeightParametrization(torch.nn.Module):
    r"""Scales a tensor by a specified scaling factor

    Args:
        scaling_factor (Tensor): scaling factor by which to multiply
            the tensor
    """

    def __init__(
        self,
        scaling_factor: Tensor,
    ) -> None:
        super().__init__()
        self.scaling_factor = scaling_factor

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.scaling_factor
