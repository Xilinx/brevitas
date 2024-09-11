# This code was taken and modified from the Hugging Face Diffusers repository under the following
# LICENSE:

# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Mapping, Optional

import diffusers
from diffusers.models.attention_processor import Attention
from diffusers.models.lora import LoRACompatibleLinear
import packaging
import packaging.version
import torch
import torch.nn.functional as F

from brevitas.graph.base import ModuleInstanceToModuleInstance
from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.nn.quant_activation import QuantIdentity
from brevitas.nn.quant_scale_bias import ScaleBias
from brevitas.quant_tensor import _unpack_quant_tensor


class QuantizableAttention(Attention):

    def __init__(
            self,
            query_dim: int,
            cross_attention_dim: Optional[int] = None,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.0,
            bias=False,
            upcast_attention: bool = False,
            upcast_softmax: bool = False,
            cross_attention_norm: Optional[str] = None,
            cross_attention_norm_num_groups: int = 32,
            added_kv_proj_dim: Optional[int] = None,
            norm_num_groups: Optional[int] = None,
            spatial_norm_dim: Optional[int] = None,
            out_bias: bool = True,
            scale_qk: bool = True,
            only_cross_attention: bool = False,
            eps: float = 1e-5,
            rescale_output_factor: float = 1.0,
            residual_connection: bool = False,
            _from_deprecated_attn_block=False,
            dtype=torch.float32,
            processor: Optional["AttnProcessor"] = None):

        super().__init__(
            query_dim,
            cross_attention_dim,
            heads,
            dim_head,
            dropout,
            bias,
            upcast_attention,
            upcast_softmax,
            cross_attention_norm,
            cross_attention_norm_num_groups,
            added_kv_proj_dim,
            norm_num_groups,
            spatial_norm_dim,
            out_bias,
            scale_qk,
            only_cross_attention,
            eps,
            rescale_output_factor,
            residual_connection,
            _from_deprecated_attn_block,
            processor,
        )
        if self.to_q.weight.shape == self.to_k.weight.shape:
            self.to_qkv = LoRACompatibleLinear(
                query_dim, 3 * self.inner_dim, bias=bias, dtype=dtype)

            del self.to_q
            del self.to_k
            del self.to_v

        else:
            self.to_q = LoRACompatibleLinear(query_dim, self.inner_dim, bias=bias, dtype=dtype)
            self.to_kv = LoRACompatibleLinear(
                self.cross_attention_dim, 2 * self.inner_dim, bias=bias, dtype=dtype)

            del self.to_k
            del self.to_v

        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(
            LoRACompatibleLinear(self.inner_dim, query_dim, bias=out_bias, dtype=dtype))
        self.to_out.append(torch.nn.Dropout(dropout))

    def load_state_dict(
            self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        if hasattr(self, 'to_qkv') and 'to_q.weight' in state_dict:
            new_weights = torch.cat(
                [state_dict['to_q.weight'], state_dict['to_k.weight'], state_dict['to_v.weight']],
                dim=0)
            state_dict['to_qkv.weight'] = new_weights

            del state_dict['to_q.weight']
            del state_dict['to_k.weight']
            del state_dict['to_v.weight']
        elif hasattr(self, 'to_kv') and 'to_k.weight' in state_dict:
            new_weights = torch.cat([state_dict['to_k.weight'], state_dict['to_v.weight']], dim=0)
            state_dict['to_kv.weight'] = new_weights
            del state_dict['to_k.weight']
            del state_dict['to_v.weight']
        return super().load_state_dict(state_dict, strict, assign)


class QuantAttentionLast(Attention):

    def __init__(
            self,
            query_dim: int,
            cross_attention_dim: Optional[int] = None,
            heads: int = 8,
            kv_heads: Optional[int] = None,
            dim_head: int = 64,
            dropout: float = 0.0,
            bias: bool = False,
            upcast_attention: bool = False,
            upcast_softmax: bool = False,
            cross_attention_norm: Optional[str] = None,
            cross_attention_norm_num_groups: int = 32,
            qk_norm: Optional[str] = None,
            added_kv_proj_dim: Optional[int] = None,
            added_proj_bias: Optional[bool] = True,
            norm_num_groups: Optional[int] = None,
            spatial_norm_dim: Optional[int] = None,
            out_bias: bool = True,
            scale_qk: bool = True,
            only_cross_attention: bool = False,
            eps: float = 1e-5,
            rescale_output_factor: float = 1.0,
            residual_connection: bool = False,
            _from_deprecated_attn_block: bool = False,
            processor: Optional["AttnProcessor"] = None,
            out_dim: int = None,
            context_pre_only=None,
            pre_only=False,
            matmul_input_quant=None,
            is_equalized=False,
            fuse_qkv=False):

        super().__init__(
            query_dim,
            cross_attention_dim,
            heads,
            kv_heads,
            dim_head,
            dropout,
            bias,
            upcast_attention,
            upcast_softmax,
            cross_attention_norm,
            cross_attention_norm_num_groups,
            qk_norm,
            added_kv_proj_dim,
            added_proj_bias,
            norm_num_groups,
            spatial_norm_dim,
            out_bias,
            scale_qk,
            only_cross_attention,
            eps,
            rescale_output_factor,
            residual_connection,
            _from_deprecated_attn_block,
            processor,
            out_dim,
            context_pre_only,
            pre_only,
        )
        if fuse_qkv:
            self.fuse_projections()

        self.output_softmax_quant = QuantIdentity(matmul_input_quant)
        self.out_q = QuantIdentity(matmul_input_quant)
        self.out_k = QuantIdentity(matmul_input_quant)
        self.out_v = QuantIdentity(matmul_input_quant)
        if is_equalized:
            replacements = []
            for n, m in self.named_modules():
                if isinstance(m, torch.nn.Linear):
                    in_channels = m.in_features
                    eq_m = EqualizedModule(ScaleBias(in_channels, False, (1, 1, -1)), m)
                    r = ModuleInstanceToModuleInstance(m, eq_m)
                    replacements.append(r)
            for r in replacements:
                r.apply(self)

    def get_attention_scores(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            attention_mask: torch.Tensor = None) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device)
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        attention_probs = _unpack_quant_tensor(self.output_softmax_quant(attention_probs))
        return attention_probs


class QuantAttention(QuantizableAttention):

    def __init__(
            self,
            query_dim: int,
            cross_attention_dim: Optional[int] = None,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.0,
            bias=False,
            upcast_attention: bool = False,
            upcast_softmax: bool = False,
            cross_attention_norm: Optional[str] = None,
            cross_attention_norm_num_groups: int = 32,
            added_kv_proj_dim: Optional[int] = None,
            norm_num_groups: Optional[int] = None,
            spatial_norm_dim: Optional[int] = None,
            out_bias: bool = True,
            scale_qk: bool = True,
            only_cross_attention: bool = False,
            eps: float = 1e-5,
            rescale_output_factor: float = 1.0,
            residual_connection: bool = False,
            _from_deprecated_attn_block=False,
            processor: Optional["AttnProcessor"] = None,
            matmul_input_quant=None,
            dtype=torch.float32,
            is_equalized=False):

        super().__init__(
            query_dim,
            cross_attention_dim,
            heads,
            dim_head,
            dropout,
            bias,
            upcast_attention,
            upcast_softmax,
            cross_attention_norm,
            cross_attention_norm_num_groups,
            added_kv_proj_dim,
            norm_num_groups,
            spatial_norm_dim,
            out_bias,
            scale_qk,
            only_cross_attention,
            eps,
            rescale_output_factor,
            residual_connection,
            _from_deprecated_attn_block,
            dtype,
            processor,
        )
        self.output_softmax_quant = QuantIdentity(matmul_input_quant)
        self.out_q = QuantIdentity(matmul_input_quant)
        self.out_k = QuantIdentity(matmul_input_quant)
        self.out_v = QuantIdentity(matmul_input_quant)
        if is_equalized:
            replacements = []
            for n, m in self.named_modules():
                if isinstance(m, torch.nn.Linear):
                    in_channels = m.in_features
                    eq_m = EqualizedModule(ScaleBias(in_channels, False, (1, 1, -1)), m)
                    r = ModuleInstanceToModuleInstance(m, eq_m)
                    replacements.append(r)
            for r in replacements:
                r.apply(self)

    def get_attention_scores(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            attention_mask: torch.Tensor = None) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device)
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        attention_probs = _unpack_quant_tensor(self.output_softmax_quant(attention_probs))
        return attention_probs


class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            assert attn.norm_cross is None, "Not supported"
            query, key, value = attn.to_qkv(hidden_states, scale=scale).chunk(3, dim=-1)

        else:
            assert not hasattr(attn, 'to_qkv'), 'Model not created correctly'
            query = attn.to_q(hidden_states, scale=scale)
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            key, value = attn.to_kv(encoder_hidden_states, scale=scale).chunk(2, dim=-1)
        if hasattr(attn, 'out_q'):
            query = _unpack_quant_tensor(attn.out_q(query))
        if hasattr(attn, 'out_k'):
            key = _unpack_quant_tensor(attn.out_k(key))
        if hasattr(attn, 'out_v'):
            value = _unpack_quant_tensor(attn.out_v(value))
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1,
                                                    -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale: float = 1.0,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            assert attn.norm_cross is None, "Not supported"
            query, key, value = attn.to_qkv(hidden_states, scale=scale).chunk(3, dim=-1)

        else:
            assert not hasattr(attn, 'to_qkv'), 'Model not created correctly'
            query = attn.to_q(hidden_states, scale=scale)
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            key, value = attn.to_kv(encoder_hidden_states, scale=scale).chunk(2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, scale=scale)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1,
                                                    -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
