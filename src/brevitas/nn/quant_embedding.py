# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Embedding
from torch.nn.functional import embedding

from brevitas.inject.defaults import Int8WeightPerTensorFloat
from brevitas.quant_tensor import _unpack_quant_tensor
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
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
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
            _weight=_weight,
            device=device,
            dtype=dtype)
        QuantWeightMixin.__init__(self, weight_quant=weight_quant, **kwargs)
        self.accept_quant_tensor = False
        self.return_quant_tensor = return_quant_tensor

    @property
    def output_channel_dim(self) -> int:
        return 0

    @property
    def out_channels(self) -> int:
        return self.num_embeddings

    def forward(self, inp):
        quant_weight = self.quant_weight()
        out = embedding(
            inp,
            quant_weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse)
        if self.return_quant_tensor:
            assert isinstance(out, QuantTensor), "Enable weight quantization to return QuantTensor"
            return out
        else:
            out = _unpack_quant_tensor(out)

        return out
