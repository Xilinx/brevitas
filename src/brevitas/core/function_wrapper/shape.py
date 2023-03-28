# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
ScriptModule classes to compute the view of a tensor according to various different criteria.
"""

from typing import Optional, Tuple

import torch

import brevitas
from brevitas.core.function_wrapper import Identity
from brevitas.function.shape import over_batch_over_output_channels
from brevitas.function.shape import over_batch_over_tensor
from brevitas.function.shape import over_output_channels
from brevitas.function.shape import over_tensor


class PermuteDims(brevitas.jit.ScriptModule):

    def __init__(self, permute_dims: Tuple[int, ...]) -> None:
        super(PermuteDims, self).__init__()
        self.permute_dims = permute_dims

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        return x.permute(*self.permute_dims).contiguous()


class OverTensorView(brevitas.jit.ScriptModule):
    """
    ScriptMoodule to compute the :func:`~brevitas.function.shape.over_tensor` view of an input tensor.

    Examples:
        >>> view_module = OverTensorView()
        >>> y = view_module(torch.empty(size=[16, 6, 5, 5]))
        >>> y.shape
        torch.Size([2400])
    """

    def __init__(self) -> None:
        super(OverTensorView, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        shape = over_tensor(x)
        return x.reshape(shape)


class OverOutputChannelView(brevitas.jit.ScriptModule):
    """
    ScriptMoodule to compute the :func:`~brevitas.function.shape.over_output_channels` view of an
    input tensor.

    Examples:
        >>> view_module = OverOutputChannelView(permute_dims=None)
        >>> y = view_module(torch.empty(size=[16, 8, 5, 5]))
        >>> y.shape
        torch.Size([16, 200])
    """

    def __init__(self, permute_dims: Optional[Tuple[int, ...]]) -> None:
        super(OverOutputChannelView, self).__init__()
        if permute_dims is not None:
            self.permute_impl = PermuteDims(permute_dims)
        else:
            self.permute_impl = Identity()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        y = self.permute_impl(x)
        shape = over_output_channels(y)
        return y.reshape(shape)


class OverBatchOverTensorView(brevitas.jit.ScriptModule):
    """
    ScriptMoodule to compute the :func:`~brevitas.function.shape.over_batch_over_tensor` view of an
    input tensor.

    Examples:
        >>> view_module = OverBatchOverTensorView()
        >>> y = view_module(torch.empty(size=[8, 10, 5, 5]))
        >>> y.shape
        torch.Size([8, 250])
    """

    def __init__(self) -> None:
        super(OverBatchOverTensorView, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        shape = over_batch_over_tensor(x)
        return x.reshape(shape)


class OverBatchOverOutputChannelView(brevitas.jit.ScriptModule):
    """
    ScriptModule to compute the :func:`~brevitas.function.shape.over_batch_over_output_channels`
    view of an input tensor.

    Examples:
        >>> view_module = OverBatchOverOutputChannelView()
        >>> y = view_module(torch.empty(size=[8, 10, 5, 5]))
        >>> y.shape
        torch.Size([8, 10, 25])
    """

    def __init__(self) -> None:
        super(OverBatchOverOutputChannelView, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        shape = over_batch_over_output_channels(x)
        return x.reshape(shape)


class StatsInputViewShapeImpl(object):
    """
    Enum-like object to collect pointers to variants of ScriptModules that perform a view on a tensor.
    All adhere to the same interface.
    """
    OVER_TENSOR = OverTensorView
    OVER_OUTPUT_CHANNELS = OverOutputChannelView
    OVER_BATCH_OVER_TENSOR = OverBatchOverTensorView
    OVER_BATCH_OVER_OUTPUT_CHANNELS = OverBatchOverOutputChannelView
