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
A collection of miscellaneous ScriptModule used in various quantizers.
"""

import brevitas
import torch


class InplaceNoOp(brevitas.jit.ScriptModule):
    """
    ScriptModule in-place no-op placehoder.

    Examples:
        >>> x = torch.tensor(1.0)
        >>> no_op = InplaceNoOp()
        >>> no_op(x)
    """

    def __init__(self):
        super(InplaceNoOp, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor) -> None:
        return None


class Identity(brevitas.jit.ScriptModule):
    """
    Identity ScriptModule.

    Examples:
        >>> identity = Identity()
        >>> x = torch.randn(size=[10,])
        >>> y = identity(x)
        >>> y is x
        True
    """

    def __init__(self) -> None:
        super(Identity, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PowerOfTwo(brevitas.jit.ScriptModule):
    """
    ScriptModule implementation of 2.0 ** x.

    Examples:
        >>> power_of_two = PowerOfTwo()
        >>> x = torch.tensor(5.0)
        >>> power_of_two(x)
        tensor(32.)
    """

    def __init__(self) -> None:
        super(PowerOfTwo, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 ** x


class LogTwo(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~torch.log2`.

    Examples:
        >>> log_two = LogTwo()
        >>> x = torch.tensor(8.0)
        >>> log_two(x)
        tensor(3.)
    """

    def __init__(self) -> None:
        super(LogTwo, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log2(x)


class InplaceLogTwo(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~torch.log2_`.

    Examples:
        >>> inplace_log_two = InplaceLogTwo()
        >>> x = torch.tensor(8.0)
        >>> inplace_log_two(x)
        >>> x
        tensor(3.)

    Note:
        The forward method returns None.
    """

    def __init__(self) -> None:
        super(InplaceLogTwo, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor) -> None:
        x.log2_()






