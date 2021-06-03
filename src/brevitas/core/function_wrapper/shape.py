# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
ScriptModule classes to compute the view of a tensor according to various different criteria.
"""

from typing import Optional, Tuple

import torch

import brevitas
from brevitas.function.shape import over_tensor, over_output_channels, over_batch_over_tensor
from brevitas.function.shape import over_batch_over_output_channels
from brevitas.core.function_wrapper import Identity


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
