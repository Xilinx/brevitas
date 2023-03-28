# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
ScriptModule wrappers of various functions defined in :obj:`~brevitas.function.ops_ste`.
"""

import torch

import brevitas
from brevitas.function.ops_ste import *


class RoundSte(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~brevitas.function.ops_ste.round_ste`.
    """

    def __init__(self) -> None:
        super(RoundSte, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        return round_ste(x)


class FloorSte(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~brevitas.function.ops_ste.floor_ste`.
    """

    def __init__(self) -> None:
        super(FloorSte, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        return floor_ste(x)


class RoundToZeroSte(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~brevitas.function.ops_ste.round_to_zero_ste`.
    """

    def __init__(self) -> None:
        super(RoundToZeroSte, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        return round_to_zero_ste(x)


class DPURoundSte(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~brevitas.function.ops_ste.dpu_round_ste`.
    """

    def __init__(self) -> None:
        super(DPURoundSte, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        return dpu_round_ste(x)


class CeilSte(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~brevitas.function.ops_ste.ceil_ste`.
    """

    def __init__(self) -> None:
        super(CeilSte, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        return ceil_ste(x)


class ScalarClampMinSte(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~brevitas.function.ops_ste.scalar_clamp_min_ste`.
    """

    __constants__ = ['min_val']

    def __init__(self, min_val: float) -> None:
        super(ScalarClampMinSte, self).__init__()
        self.min_val = min_val

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        return scalar_clamp_min_ste(x, self.min_val)


class TensorClampSte(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~brevitas.function.ops_ste.tensor_clamp_ste`.
    """

    def __init__(self) -> None:
        super(TensorClampSte, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
        return tensor_clamp_ste(x, min_val, max_val)


class InplaceTensorClampSte(brevitas.jit.ScriptModule):
    """
    ScriptModule wrapper for :func:`~brevitas.function.ops_ste.tensor_clamp_ste_`.
    """

    def __init__(self) -> None:
        super(InplaceTensorClampSte, self).__init__()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
        return tensor_clamp_ste_(x, min_val, max_val)
