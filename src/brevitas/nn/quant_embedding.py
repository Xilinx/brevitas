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

from typing import Union, Type, Optional

import torch
from torch import Tensor
from torch.nn import Embedding, EmbeddingBag
from torch.nn.functional import embedding

from brevitas.function.ops_ste import ceil_ste
from brevitas.function.ops import max_int
from brevitas.quant_tensor import QuantTensor
from brevitas.inject.defaults import Int8WeightPerTensorFloat
from .mixin.parameter import QuantWeightMixin
from .quant_layer import WeightQuantType

__all__ = ['QuantEmbedding']


class QuantEmbedding(QuantWeightMixin, Embedding):

    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int, 
            padding_idx: Optional[int] = None,
            max_norm: Optional[float] = None, 
            norm_type: float = 2., 
            scale_grad_by_freq: bool = False,
            sparse: bool = False, 
            _weight: Optional[Tensor] = None,
            weight_quant: WeightQuantType = Int8WeightPerTensorFloat,
            return_quant_tensor = False,
            **kwargs) -> None:
        Embedding.__init__(
            self,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight)
        QuantWeightMixin.__init__(
            self,
            weight_quant=weight_quant,
            **kwargs)
        self.accept_quant_tensor = False
        self.return_quant_tensor = return_quant_tensor

    def forward(self, inp):
        quant_weight = self.quant_weight()
        out = embedding(
            inp, 
            quant_weight.value, 
            self.padding_idx, 
            self.max_norm, 
            self.norm_type, 
            self.scale_grad_by_freq, 
            self.sparse)
        if self.return_quant_tensor:
            scale = quant_weight.scale
            zero_point = quant_weight.zero_point
            bit_width = quant_weight.bit_width
            if any(t.numel() > 1 for t in [scale, zero_point, bit_width]):
                raise RuntimeError("Only per-tensor quantization is supported.")
            signed = quant_weight.signed
            training = quant_weight.training
            out = QuantTensor(out, scale, zero_point, bit_width, signed, training)
        return out
                
        