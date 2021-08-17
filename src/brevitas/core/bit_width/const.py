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

import torch
from torch import Tensor
from torch.nn import Module

import brevitas
from brevitas.function.ops_ste import tensor_clamp_ste
from brevitas.core.utils import StatelessBuffer


class BitWidthConst(brevitas.jit.ScriptModule):
    """ 
    ScriptModule that returns a constant bit-width wrapped in a float torch.tensor.

    Args:
        bit_width (int): bit-width value.

    Examples:
        >>> bit_width = BitWidthConst(8)
        >>> bit_width()
        tensor(8.)

    Note:
        The bit-width is not part of the Module's state, meaning that it won't be saved as part of
        a checkpoint.

    Note:
        Maps to bit_width_impl_type == BitWidthImplType.CONST == 'CONST' == 'const' in higher-level APIs.
    """
    def __init__(self, bit_width: int) -> None:
        super(BitWidthConst, self).__init__()
        assert isinstance(bit_width, int)
        self.bit_width = StatelessBuffer(torch.tensor(float(bit_width)))

    @brevitas.jit.script_method
    def forward(self) -> Tensor:
        return self.bit_width()


class MsbClampBitWidth(brevitas.jit.ScriptModule):

    def __init__(
            self,
            bit_width_to_remove_impl: Module,
            min_overall_bit_width: int,
            max_overall_bit_width: int) -> None:
        super(MsbClampBitWidth, self).__init__()

        self.min_overall_bit_width = BitWidthConst(min_overall_bit_width)
        self.max_overall_bit_width = BitWidthConst(max_overall_bit_width)
        self.bit_width_to_remove_impl = bit_width_to_remove_impl

    @brevitas.jit.script_method
    def forward(self, input_bit_width: Tensor) -> Tensor:
        bit_width_to_remove = self.bit_width_to_remove_impl()
        output_bit_width = torch.abs(input_bit_width - bit_width_to_remove)
        output_bit_width = tensor_clamp_ste(
            output_bit_width,
            self.min_overall_bit_width(),
            self.max_overall_bit_width())
        return output_bit_width