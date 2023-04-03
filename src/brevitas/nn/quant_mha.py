"""
Copyright (C) 2023,     Advanced Micro Devices, Inc.
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of AMD, Facebook, Deepmind Technologies, NYU,
   NEC Laboratories America and IDIAP Research Institute nor the names
   of its contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import math
from typing import Optional, Tuple, Union
import warnings

from packaging import version
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.init import xavier_uniform_

from brevitas import torch_version
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
from brevitas.nn.utils import check_tensors_same_ptr
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int32Bias
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat
from brevitas.quant_tensor import QuantTensor


class QuantMultiheadAttention(Module):
    """"
    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            packed_in_proj=True,
            in_proj_input_quant=Int8ActPerTensorFloat,
            in_proj_weight_quant=Int8WeightPerTensorFloat,
            in_proj_bias_quant=Int32Bias,
            softmax_input_quant=None,
            attn_output_weights_quant=Uint8ActPerTensorFloat,
            q_scaled_quant=Int8ActPerTensorFloat,
            k_transposed_quant=Int8ActPerTensorFloat,
            v_quant=Int8ActPerTensorFloat,
            out_proj_input_quant=Int8ActPerTensorFloat,
            out_proj_weight_quant=Int8WeightPerTensorFloat,
            out_proj_bias_quant=Int32Bias,
            out_proj_output_quant=None,
            batch_first=False,
            return_quant_tensor=False,
            **kwargs) -> None:
        super(QuantMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        def filter_kwargs(prefix):
            return {k[len(prefix):]: v for k, v in kwargs.items() if k.startswith(prefix)}

        if self._qkv_same_embed_dim and packed_in_proj:
            self.in_proj = QuantLinear(
                out_features=3 * embed_dim,
                in_features=embed_dim,
                bias=bias,
                input_quant=in_proj_input_quant,
                weight_quant=in_proj_weight_quant,
                bias_quant=in_proj_bias_quant,
                **filter_kwargs('in_proj_'))
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            self.q_proj = QuantLinear(
                out_features=embed_dim,
                in_features=embed_dim,
                bias=bias,
                input_quant=in_proj_input_quant,
                weight_quant=in_proj_weight_quant,
                bias_quant=in_proj_bias_quant,
                **filter_kwargs('in_proj_'))
            self.k_proj = QuantLinear(
                out_features=embed_dim,
                in_features=self.kdim,
                bias=bias,
                input_quant=in_proj_input_quant,
                weight_quant=in_proj_weight_quant,
                bias_quant=in_proj_bias_quant,
                **filter_kwargs('in_proj_'))
            self.v_proj = QuantLinear(
                out_features=embed_dim,
                in_features=self.vdim,
                bias=bias,
                input_quant=in_proj_input_quant,
                weight_quant=in_proj_weight_quant,
                bias_quant=in_proj_bias_quant,
                **filter_kwargs('in_proj_'))
            self.in_proj = None

        # Keep compatibility with this regression between 1.6.0 and 1.8.2, where bias is always enabled
        # https://github.com/pytorch/pytorch/issues/52257
        out_proj_bias = bias or (version.parse('1.8.2') >= torch_version >= version.parse('1.6.0'))

        self.out_proj = QuantLinear(
            embed_dim,
            embed_dim,
            bias=out_proj_bias,
            input_quant=out_proj_input_quant,
            weight_quant=out_proj_weight_quant,
            bias_quant=out_proj_bias_quant,
            output_quant=out_proj_output_quant,
            return_quant_tensor=return_quant_tensor,
            **filter_kwargs('out_proj_'))

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim)))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim)))
        else:
            self.bias_k = self.bias_v = None

        self.softmax_input_quant = QuantIdentity(
            act_quant=softmax_input_quant, **filter_kwargs('softmax_input_'))
        self.attn_output_weights_quant = QuantIdentity(
            act_quant=attn_output_weights_quant, **filter_kwargs('attn_output_weights_'))
        self.q_scaled_quant = QuantIdentity(act_quant=q_scaled_quant, **filter_kwargs('q_scaled_'))
        self.k_transposed_quant = QuantIdentity(
            act_quant=k_transposed_quant, **filter_kwargs('k_transposed_'))
        self.v_quant = QuantIdentity(act_quant=v_quant, **filter_kwargs('v_'))

        self.add_zero_attn = add_zero_attn
        self._reset_parameters()

    def _reset_parameters(self):
        if self.in_proj is not None:
            xavier_uniform_(self.in_proj.weight)
            if self.in_proj.bias is not None:
                constant_(self.in_proj.bias, 0.)
        else:
            xavier_uniform_(self.q_proj.weight)
            xavier_uniform_(self.k_proj.weight)
            xavier_uniform_(self.v_proj.weight)
            if self.q_proj.bias is not None:
                constant_(self.q_proj.bias, 0.)
            if self.k_proj.bias is not None:
                constant_(self.k_proj.bias, 0.)
            if self.v_proj.bias is not None:
                constant_(self.v_proj.bias, 0.)

        if self.out_proj.bias is not None:
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def mha_shape_check(
            self,
            query: Union[Tensor, QuantTensor],
            key: Union[Tensor, QuantTensor],
            value: Union[Tensor, QuantTensor],
            key_padding_mask: Optional[Tensor],
            attn_mask: Optional[Tensor],
            num_heads: int):
        # Verifies the expected shape for `query, `key`, `value`, `key_padding_mask` and `attn_mask`
        # and returns if the input is batched or not.
        # Raises an error if `query` is not 2-D (unbatched) or 3-D (batched) tensor.

        # Shape check.
        if query.dim() == 3:
            # Batched Inputs
            is_batched = True
            assert key.dim() == 3 and value.dim() == 3, \
                ("For batched (3-D) `query`, expected `key` and `value` to be 3-D"
                 f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")
            if key_padding_mask is not None:
                assert key_padding_mask.dim() == 2, \
                    ("For batched (3-D) `query`, expected `key_padding_mask` to be `None` or 2-D"
                     f" but found {key_padding_mask.dim()}-D tensor instead")
            if attn_mask is not None:
                assert attn_mask.dim() in (2, 3), \
                    ("For batched (3-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                     f" but found {attn_mask.dim()}-D tensor instead")
        elif query.dim() == 2:
            # Unbatched Inputs
            is_batched = False
            assert key.dim() == 2 and value.dim() == 2, \
                ("For unbatched (2-D) `query`, expected `key` and `value` to be 2-D"
                 f" but found {key.dim()}-D and {value.dim()}-D tensors respectively")

            if key_padding_mask is not None:
                assert key_padding_mask.dim() == 1, \
                    ("For unbatched (2-D) `query`, expected `key_padding_mask` to be `None` or 1-D"
                     f" but found {key_padding_mask.dim()}-D tensor instead")

            if attn_mask is not None:
                assert attn_mask.dim() in (2, 3), \
                    ("For unbatched (2-D) `query`, expected `attn_mask` to be `None`, 2-D or 3-D"
                     f" but found {attn_mask.dim()}-D tensor instead")
                if attn_mask.dim() == 3:
                    expected_shape = (num_heads, query.shape[0], key.shape[0])
                    assert attn_mask.shape == expected_shape, \
                        (f"Expected `attn_mask` shape to be {expected_shape} but got {attn_mask.shape}")
        else:
            raise AssertionError(
                f"query should be unbatched 2D or batched 3D tensor but received {query.dim()}-D query tensor"
            )

        return is_batched

    def multi_head_attention(
            self,
            query: Union[Tensor, QuantTensor],
            key: Union[Tensor, QuantTensor],
            value: Union[Tensor, QuantTensor],
            embed_dim_to_check: int,
            num_heads: int,
            bias_k: Optional[Tensor],
            bias_v: Optional[Tensor],
            add_zero_attn: bool,
            dropout_p: float,
            training: bool = True,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            use_separate_proj_weight: bool = False,
            static_k: Optional[Tensor] = None,
            static_v: Optional[Tensor] = None,
            average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            embed_dim_to_check: total dimension of the model.
            num_heads: parallel attention heads.
            in_proj_weight, in_proj_bias: input projection weight and bias.
            bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
            add_zero_attn: add a new batch of zeros to the key and
                           value sequences at dim=1.
            dropout_p: probability of an element to be zeroed.
            out_proj_weight, out_proj_bias: the output projection weight and bias.
            training: apply dropout if is ``True``.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
            use_separate_proj_weight: the function accept the proj. weights for query, key,
                and value in different forms. If false, in_proj_weight will be used, which is
                a combination of q_proj_weight, k_proj_weight, v_proj_weight.
            q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
            static_k, static_v: static key and value used for attention operators.
            average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
                Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
                when ``need_weights=True.``. Default: True


        Shape:
            Inputs:
            - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a FloatTensor is provided, it will be directly added to the value.
              If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
              positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
              while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
              are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.
            - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
              N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
            - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
              N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

            Outputs:
            - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
              attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
              :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
              :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
        """

        is_batched = self.mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

        # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
        # is batched, run the computation and before returning squeeze the
        # batch dimension so that the output doesn't carry this temporary batch dimension.
        if not is_batched:
            # unsqueeze if the input is unbatched
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        assert embed_dim == embed_dim_to_check, \
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        if use_separate_proj_weight:
            # allow MHA to have different embedding dimensions when separate projection weights are used
            assert key.shape[:2] == value.shape[:2], \
                f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        else:
            assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

        #
        # compute in-projection
        #
        if self.in_proj is not None:
            if check_tensors_same_ptr([key, query, value]):
                # self-attention
                q, k, v = self.in_proj(query).chunk(3, dim=-1)
            else:
                raise RuntimeError(
                    "Packed in_proj is supported only for self-attention with k is v is q. Set packed_in_proj=False."
                )
        else:
            assert self.q_proj is not None, "use_separate_proj_weight is True but q_proj is None"
            assert self.k_proj is not None, "use_separate_proj_weight is True but k_proj is None"
            assert self.v_proj is not None, "use_separate_proj_weight is True but v_proj is None"
            q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)

        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                    f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # add bias along batch dimension (currently second)
        if bias_k is not None and bias_v is not None:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert bias_k is None
            assert bias_v is None

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        if static_k is None:
            k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_k.size(0) == bsz * num_heads, \
                f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
            assert static_k.size(2) == head_dim, \
                f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
            k = static_k
        if static_v is None:
            v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
        else:
            # TODO finish disentangling control flow so we don't do in-projections when statics are passed
            assert static_v.size(0) == bsz * num_heads, \
                f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
            assert static_v.size(2) == head_dim, \
                f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
            v = static_v

        # add zero attention along batch dimension (now first)
        if add_zero_attn:
            zero_attn_shape = (bsz * num_heads, 1, head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        # update source sequence length after adjustments
        src_len = k.size(1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
                expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        # adjust dropout probability
        if not training:
            dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #

        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)
        k_transposed = k.transpose(-2, -1)

        # Quantize q_scaled and k_transposed
        q_scaled = self.q_scaled_quant(q_scaled)
        k_transposed = self.k_transposed_quant(k_transposed)

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k_transposed)
        else:
            attn_output_weights = torch.bmm(q_scaled, k_transposed)

        # Quantize the input to softmax, if any
        attn_output_weights = self.softmax_input_quant(attn_output_weights)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

        # Quantize attn_output_weights and value
        attn_output_weights = self.attn_output_weights_quant(attn_output_weights)
        v = self.v_quant(v)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)

        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        if need_weights:
            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            return attn_output, attn_output_weights
        else:
            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
            return attn_output, None

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """
        is_batched = query.dim() == 3
        if any([hasattr(t, 'is_nested') and t.is_nested for t in (query, key, value)]):
            raise RuntimeError("Nested inputs not supported for quantization.")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        attn_output, attn_output_weights = self.multi_head_attention(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):

        def set_bias(proj_name, value):
            bias_name = f'{prefix}{proj_name}_proj.bias'
            state_dict[bias_name] = value

        def set_weight(proj_name, value):
            key = f'{prefix}{proj_name}_proj.weight'
            state_dict[key] = value

        for name, value in list(state_dict.items()):
            if prefix + 'in_proj_weight' in name:
                if self.in_proj is not None:
                    set_weight('in', value)
                # We might have set packed_in_proj=False, which is absent in the original float implementation
                else:
                    if not value.size(0) % 3 == 0:
                        raise RuntimeError("in_proj dim 0 doesn't divide evenly into 3 tensors.")
                    q_proj, k_proj, v_proj = torch.chunk(value, 3, dim=0)
                    set_weight('q', q_proj)
                    set_weight('k', k_proj)
                    set_weight('v', v_proj)
                del state_dict[name]
            elif prefix + 'q_proj_weight' in name:
                assert prefix + 'k_proj_weight' in state_dict.keys(), 'k_proj_weight is missing.'
                assert prefix + 'v_proj_weight' in state_dict.keys(), 'v_proj_weight is missing.'
                set_weight('q', value)
                del state_dict[name]
            elif prefix + 'k_proj_weight' in name:
                assert prefix + 'q_proj_weight' in state_dict.keys(), 'q_proj_weight is missing.'
                assert prefix + 'v_proj_weight' in state_dict.keys(), 'v_proj_weight is missing.'
                set_weight('k', value)
                del state_dict[name]
            elif prefix + 'v_proj_weight' in name:
                assert prefix + 'q_proj_weight' in state_dict.keys(), 'q_proj_weight is missing.'
                assert prefix + 'k_proj_weight' in state_dict.keys(), 'k_proj_weight is missing.'
                set_weight('v', value)
            elif prefix + 'in_proj_bias' in name:
                if self.in_proj is not None:
                    set_bias('in', value)
                else:
                    q_proj, k_proj, v_proj = torch.chunk(value, 3, dim=0)
                    set_bias('q', q_proj)
                    set_bias('k', k_proj)
                    set_bias('v', v_proj)
                del state_dict[name]
        super(QuantMultiheadAttention, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
