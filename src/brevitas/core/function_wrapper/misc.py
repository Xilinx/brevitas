# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
A collection of miscellaneous ScriptModule used in various quantizers.
"""

import torch

import brevitas


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


class InplaceLogTwo(torch.nn.Module):
    """
    Module wrapper for :func:`~torch.log2_`.

    Examples:
        >>> inplace_log_two = InplaceLogTwo()
        >>> x = torch.tensor(8.0)
        >>> inplace_log_two(x)
        >>> x
        tensor(3.)

    Notes:
        Inplace operations in TorchScript can be problematic, compilation is disabled.
    """

    def __init__(self) -> None:
        super(InplaceLogTwo, self).__init__()

    @torch.jit.ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.log2_()
        return x
