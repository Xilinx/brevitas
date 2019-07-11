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
from torch.nn import Parameter

from brevitas.proxy.max_bit_width_mod import MaxLinearOutputBitWidth
from brevitas.core.quant import QuantType, IntQuant
from brevitas.proxy.parameter_quant import WeightQuant, BiasQuant
from brevitas.core.bit_width import BitWidthParameter, BitWidthConst, BitWidthImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType, SCALING_SCALAR_SHAPE
from brevitas.core.stats import StatsInputViewShapeImpl
from brevitas.core.stats import StatsOp
from brevitas.nn.quant_activation import QuantAccumulator, QuantSigmoid, QuantTanh
from brevitas.proxy.parameter_quant import WeightQuant, BiasQuant


class LSTMCell(torch.jit.ScriptModule):

    def __init__(self,
                 input_size,
                 hidden_size,
                 weight_quant_type: QuantType = QuantType.FP,
                 weight_narrow_range: bool = False,
                 weight_bit_width_impl_override: Union[BitWidthParameter, BitWidthConst] = None,
                 weight_bit_width_impl_type: BitWidthImplType = BitWidthImplType.CONST,
                 weight_restrict_bit_width_type: RestrictValueType = RestrictValueType.INT,
                 weight_bit_width: int = 32,
                 weight_min_overall_bit_width: Optional[int] = 2,
                 weight_max_overall_bit_width: Optional[int] = None,
                 weight_scaling_impl_type: ScalingImplType = ScalingImplType.STATS,
                 weight_scaling_stats_op: StatsOp = StatsOp.MAX,
                 weight_scaling_per_output_channel: bool = False,
                 weight_ternary_threshold: float = 0.5,
                 weight_restrict_scaling_type: RestrictValueType = RestrictValueType.LOG_FP,
                 weight_scaling_stats_sigma: float = 3.0):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))


        self.weight_ih_quant = WeightQuant()
        self.weight_hh_quant = WeightQuant()
        self.bias_ih_quant = BiasQuant()
        self.bias_hh_quant = BiasQuant()
        self.ingate_quant_sigmoid = torch.jit.trace(QuantSigmoid(), torch.randn(size=hidden_size))
        self.forgetgate_quant_sigmoid = torch.jit.trace(QuantSigmoid(), torch.randn(size=hidden_size))
        self.cellgate_quant_tanh = torch.jit.trace(QuantTanh(), torch.randn(size=hidden_size))
        self.outgate_quant_sigmoid = torch.jit.trace(QuantSigmoid(), torch.randn(size=hidden_size))

    @torch.jit.script_method
    def forward(self, input, state):
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)
