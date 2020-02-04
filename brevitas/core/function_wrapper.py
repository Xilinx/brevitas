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

from brevitas.function.ops_ste import round_ste, tensor_clamp_ste, ceil_ste, floor_ste
from brevitas.function.shape import *
from brevitas.function import tensor_clamp


class Identity(torch.jit.ScriptModule):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return x


class RoundSte(torch.jit.ScriptModule):
    def __init__(self) -> None:
        super(RoundSte, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return round_ste(x)


class FloorSte(torch.jit.ScriptModule):
    def __init__(self) -> None:
        super(FloorSte, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return floor_ste(x)


class CeilSte(torch.jit.ScriptModule):
    def __init__(self) -> None:
        super(CeilSte, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return ceil_ste(x)


class PowerOfTwo(torch.jit.ScriptModule):
    def __init__(self) -> None:
        super(PowerOfTwo, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return 2.0 ** x


class LogTwo(torch.jit.ScriptModule):
    def __init__(self) -> None:
        super(LogTwo, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return torch.log2(x)


class TensorClampSte(torch.jit.ScriptModule):
    def __init__(self) -> None:
        super(TensorClampSte, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
        return tensor_clamp_ste(x, min_val=min_val, max_val=max_val)


class TensorClamp(torch.jit.ScriptModule):
    def __init__(self) -> None:
        super(TensorClamp, self).__init__()

    @torch.jit.script_method
    def forward(self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor):
        return tensor_clamp(x, min_val=min_val, max_val=max_val)


class ConstScalarClamp(torch.jit.ScriptModule):
    __constants__ = ['min_val', 'max_val']

    def __init__(self, min_val, max_val) -> None:
        super(ConstScalarClamp, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return torch.clamp(x, min=self.min_val, max=self.max_val)


class ClampMin(torch.jit.ScriptModule):
    __constants__ = ['min_val']

    def __init__(self, min_val: float) -> None:
        super(ClampMin, self).__init__()
        self.min_val = min_val

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        return x.clamp_min(self.min_val)


class OverTensorView(torch.jit.ScriptModule):

    def __init__(self) -> None:
        super(OverTensorView, self).__init__()

    @torch.jit.script_method
    def shape(self, x: torch.Tensor):
        return over_tensor(x)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        shape = self.shape(x)
        return x.view(shape)


class OverOutputChannelView(torch.jit.ScriptModule):

    def __init__(self) -> None:
        super(OverOutputChannelView, self).__init__()

    @torch.jit.script_method
    def shape(self, x: torch.Tensor):
        return over_output_channels(x)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        shape = self.shape(x)
        return x.view(shape)


class OverBatchOverTensorView(torch.jit.ScriptModule):

    def __init__(self) -> None:
        super(OverBatchOverTensorView, self).__init__()

    @torch.jit.script_method
    def shape(self, x: torch.Tensor):
        return over_batch_over_tensor(x)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        shape = self.shape(x)
        return x.view(shape)


class OverBatchOverOutputChannelView(torch.jit.ScriptModule):

    def __init__(self) -> None:
        super(OverBatchOverOutputChannelView, self).__init__()

    @torch.jit.script_method
    def shape(self, x: torch.Tensor):
        return over_batch_over_output_channels(x)

    @torch.jit.script_method
    def forward(self, x: torch.Tensor):
        shape = self.shape(x)
        return x.view(shape)
