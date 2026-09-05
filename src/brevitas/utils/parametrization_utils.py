# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable
from typing import Dict
from typing import Hashable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


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
        rotation_group_id (hashable, optional): model-instance-local identity
            shared by parametrizations that logically use the same rotation matrix
    """

    def __init__(
        self,
        rot_mat: Callable[[Tensor, Tensor, Optional[int]], Tensor],
        rot_func: Callable,
        axis: int,
        K: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        rotation_group_id: Optional[Hashable] = None,
    ) -> None:
        super().__init__()
        self.rot_mat = rot_mat
        self.rot_func = rot_func
        self.axis = axis
        self.K = K
        self.hidden_dim = hidden_dim
        # Unlike the identity of rot_mat, this remains meaningful when a wrapper such as
        # FSDP replaces one occurrence of a shared Parameter. It is deliberately plain
        # metadata rather than a buffer so that it does not affect state dicts.
        self.rotation_group_id = rotation_group_id

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.axis == 0:
            tensor = tensor.t()
            init_shape = tensor.shape
            if self.hidden_dim is not None:
                # This allows us to perform hadamard on a subset of the channel dimension
                # If init_shape[-1] == had_shape, the next reshape+squeeze is a no-op
                tensor = tensor.reshape(
                    *init_shape[:-1], init_shape[-1] // self.hidden_dim, self.hidden_dim).squeeze()
            tensor = self.rot_func(tensor, self.rot_mat, self.K)
            tensor = tensor.reshape(init_shape).t()
        elif self.axis == 1:
            init_shape = tensor.shape
            if self.hidden_dim is not None:
                # This allows us to perform hadamard on a subset of the channel dimension
                # If init_shape[-1] == had_shape, the next reshape+squeeze is a no-op
                tensor = tensor.reshape(
                    *init_shape[:-1], init_shape[-1] // self.hidden_dim, self.hidden_dim).squeeze()
            tensor = self.rot_func(tensor, self.rot_mat, self.K)
            tensor = tensor.reshape(init_shape)
        else:
            raise RuntimeError("Not supported yet")

        return tensor


class ScaleWeightParametrization(torch.nn.Module):
    r"""Scales a tensor by a specified scaling factor
    Args:
        scaling_factor (Tensor or Parameter): scaling factor by which to
            multiply the tensor
        axis (int): axis in which to apply the scaling
        start_end_idxs (tuple, optional): when a tensor is partially equalized,
            these indexes indicate the range of channels that are equalized,
            while the scaling factor is set to 1 for the rest of the channels
            (no-op for equalization)
        slice_idxs (tuple, optional): these indexes determine the slice of
            scaling_factor that must be used to equalize a given tensor
        use_inverse_scaling (bool): whether to take the inverse of the
            scaling factor
    """

    def __init__(
            self,
            scaling_factor: Union[Tensor, nn.Parameter],
            axis: int,
            start_end_idxs: Optional[Tuple[int, int]] = None,
            slice_idxs: Optional[Tuple[int, int]] = None,
            use_inverse_scaling: bool = False) -> None:
        super().__init__()
        self.scaling_factor = scaling_factor
        self.axis = axis
        self.start_end_idxs = start_end_idxs
        self.slice_idxs = slice_idxs
        self.use_inverse_scaling = use_inverse_scaling

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        # self.scaling_factor is 1D, so it needs to be reshaped to be broadcastable against tensor,
        # e.g. suppose that the tensor corresponds to the weights of a nn.Linear whose shape is
        # [output_channels, input_channels], then, if axis = 0, the scales are reshaped to [output_channels, 1]
        num_channels = tensor.size(self.axis)
        broadcast_shape = [num_channels if self.axis == i else 1 for i in range(tensor.ndim)]
        # self.scaling_factor might possible contain the scaling factors needed to equalize several
        # modules. If that is the case, an slice of self.scaling_factor must be taken, and
        # self.slice_idxs[0] and self.slice_idxs[1] indicate the starting and final positions of
        # the slice respectively
        scaling_factor = self.scaling_factor if self.slice_idxs is None else self.scaling_factor[
            self.slice_idxs[0]:self.slice_idxs[1]]
        # Sinks might have only a subset of their channels equalized. Moreover, this subset is assumed to
        # be a sequence of consecutive channels, thus they are specified by their start and end indexes
        # (self.start_end_idxs). Therefore, the scaling factor is set to 1, for the channels that are
        # not equalized
        if self.start_end_idxs is not None:
            # We replace the scaling factors of the channels we need to equalize, leaving the other to
            # one (i.e., no equalization)
            scaling_factor = F.pad(
                scaling_factor,
                pad=(self.start_end_idxs[0], num_channels - self.start_end_idxs[1]),
                value=1.)
        # Reciprocal is done on the fly as to preserve the tie between scale and its reciprocal
        scaling_factor = torch.reciprocal(
            scaling_factor) if self.use_inverse_scaling else scaling_factor
        return tensor * scaling_factor.reshape(broadcast_shape)


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


def extract_trainable_rotation_matrix_owners(model: nn.Module) -> List[nn.Parameter]:
    """Return one optimizer-owned parameter for each logical rotation group.

    FSDP rotation replication marks one physical copy in each logical group as the
    optimizer owner. Models without replicated rotations retain the existing identity-
    based extraction behavior.
    """
    owners = []
    ids_rot = set()
    for group in get_rotation_groups(model).values():
        marked_owners = {
            module.rot_mat for module in group if getattr(module, 'rotation_is_owner', False)}
        if len(marked_owners) > 1:
            raise RuntimeError("A logical rotation group has multiple optimizer owners.")
        parameter = next(iter(marked_owners)) if marked_owners else group[0].rot_mat
        if id(parameter) not in ids_rot:
            ids_rot.add(id(parameter))
            owners.append(parameter)
    return owners


def get_rotation_groups(model: nn.Module) -> Dict[Hashable, List[RotationWeightParametrization]]:
    """Return rotation parametrizations grouped by their logical rotation.

    Parametrizations carrying ``rotation_group_id`` are grouped by that model-instance-local
    metadata.
    For parametrizations created by older or external code, the rotation parameter itself
    is used as the key. The latter preserves the usual shared-parameter semantics.

    The occurrence lists follow ``model.modules()`` traversal order.
    """
    groups = {}
    for module in model.modules():
        if isinstance(module, RotationWeightParametrization):
            # getattr keeps discovery compatible with parametrizations deserialized from
            # checkpoints created before rotation_group_id was introduced.
            key = getattr(module, 'rotation_group_id', None)
            if key is None:
                key = module.rot_mat
            groups.setdefault(key, []).append(module)
    return groups
