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
ScriptModule wrappers of various functions defined in :obj:`~brevitas.function.ops_ste`.
"""

import brevitas
import torch
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