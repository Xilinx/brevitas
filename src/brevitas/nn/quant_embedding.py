# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Type, Union

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.nn import EmbeddingBag
from torch.nn.functional import embedding

from brevitas.function.ops import max_int
from brevitas.function.ops_ste import ceil_ste
from brevitas.inject.defaults import Int8WeightPerTensorFloat
from brevitas.quant_tensor import QuantTensor

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
            return_quant_tensor=False,
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
        QuantWeightMixin.__init__(self, weight_quant=weight_quant, **kwargs)
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
