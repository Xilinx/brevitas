from typing import Any
from typing import Optional
from typing import Tuple

import torch
from torch import nn
from transformers.integrations.sdpa_attention import repeat_kv

from brevitas.nn import ScaledDotProductAttention
from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.utils.torch_utils import KwargsForwardHook

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    from transformers.models.llama.modeling_llama import LlamaAttention
except:
    apply_rotary_pos_emb = None
    LlamaAttention = object


class LlamaQuantAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def __init__(self, *args, **kwargs):
        super.__init__(**args, **kwargs)
        self.attn = ScaledDotProductAttention()

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor,
                                            torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                         self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        attn_output, _ = quant_sdpa_attention_forward(self, query_states, key_states, value_states, attention_mask, self.attention_dropout if self.training else 0.0)
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


def quant_sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key.shape[-2]]

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()
    attn_output = module.attn(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


def attention_mask_handler(
        attention_mask, batch_size, num_heads, query_seq_length, key_value_seq_length):
    """Re-arrange attention mask to go from 4D to 3D (explicit batch_size and n_heads) or 2D
    (implicit batch_size and n_heads)."""
    if len(attention_mask.shape) == 4:
        if attention_mask.shape[0] == 1:
            attention_mask = attention_mask.repeat(batch_size, 1, 1, 1)
        if attention_mask.shape[1] == 1:
            attention_mask = attention_mask.repeat(1, num_heads, 1, 1)
        if attention_mask.shape[2] == 1:
            attention_mask = attention_mask.repeat(1, 1, query_seq_length, 1)
        attention_mask = attention_mask.view(
            batch_size * num_heads, query_seq_length, key_value_seq_length)
    elif len(attention_mask.shape) == 2 and attention_mask.shape[0] == 1:
        # This could happen in Encoder-like architecture
        assert query_seq_length == key_value_seq_length
        attention_mask = attention_mask.repeat(query_seq_length, 1)
    return attention_mask


class MultiheadAttentionWrapper(nn.Module):

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
            batch_first=False,
            device=None,
            dtype=None) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            device,
            dtype)

    @property
    def wrapped_mha(self):
        mha = self.mha
        # Workaround for activation equalization for when mha is wrapped
        # KwargsForwardHook is inserted during act equalization
        # EqualizedModule is inserted after act equalization
        if isinstance(mha, KwargsForwardHook):
            mha = mha.module
        if isinstance(mha, EqualizedModule):
            mha = mha.layer
        return mha

    @property
    def num_heads(self):
        return self.wrapped_mha.num_heads

    @property
    def batch_first(self):
        return self.wrapped_mha.batch_first

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):

        def set_bias(value):
            bias_name = f'{prefix}mha.in_proj_bias'
            if bias_name in state_dict:
                state_dict[bias_name] += value
            else:
                state_dict[bias_name] = value

        def set_weight(value):
            weight_name = f'{prefix}mha.in_proj_weight'
            if weight_name in state_dict:
                state_dict[weight_name] += value
            else:
                state_dict[weight_name] = value

        embed_dim = self.mha.embed_dim
        for name, value in list(state_dict.items()):
            if prefix + 'q_proj.weight' in name:
                weight = torch.zeros((3 * embed_dim, embed_dim),
                                     device=value.device,
                                     dtype=value.dtype)
                weight[:embed_dim] = value
                set_weight(weight)
                del state_dict[name]
            elif prefix + 'k_proj.weight' in name:
                weight = torch.zeros((3 * embed_dim, embed_dim),
                                     device=value.device,
                                     dtype=value.dtype)
                weight[embed_dim:2 * embed_dim] = value
                set_weight(weight)
                del state_dict[name]
            elif prefix + 'v_proj.weight' in name:
                weight = torch.zeros((3 * embed_dim, embed_dim),
                                     device=value.device,
                                     dtype=value.dtype)
                weight[2 * embed_dim:3 * embed_dim] = value
                set_weight(weight)
                del state_dict[name]
            if prefix + 'q_proj.bias' in name:
                bias = torch.zeros(3 * embed_dim, device=value.device, dtype=value.dtype)
                bias[:embed_dim] = value
                set_bias(bias)
                del state_dict[name]
            elif prefix + 'k_proj.bias' in name:
                bias = torch.zeros(3 * embed_dim, device=value.device, dtype=value.dtype)
                bias[embed_dim:2 * embed_dim] = value
                set_bias(bias)
                del state_dict[name]
            elif prefix + 'v_proj.bias' in name:
                bias = torch.zeros(3 * embed_dim, device=value.device, dtype=value.dtype)
                bias[2 * embed_dim:3 * embed_dim] = value
                set_bias(bias)
                del state_dict[name]
            elif prefix + 'out_proj.weight' in name:
                state_dict[prefix + 'mha.out_proj.weight'] = value
                del state_dict[name]
            elif prefix + 'out_proj.bias' in name:
                state_dict[prefix + 'mha.out_proj.bias'] = value
                del state_dict[name]
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


class QuantizableOPTAttention(MultiheadAttentionWrapper):

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if key_value_states is None:
            key_value_states = hidden_states
        if layer_head_mask is not None:
            raise RuntimeError("layer_head_mask is not supported.")
        if self.batch_first:
            batch_size, query_seq_length = hidden_states.shape[:2]
            key_value_seq_length = key_value_states.shape[1]
        else:
            query_seq_length, batch_size = hidden_states.shape[:2]
            key_value_seq_length = key_value_states.shape[0]
        num_heads = self.num_heads
        attention_mask = (
            attention_mask_handler(
                attention_mask, batch_size, num_heads, query_seq_length, key_value_seq_length)
            if attention_mask is not None else None)
        attn_output, attn_output_weights = self.mha(
            hidden_states,
            key_value_states,
            key_value_states,
            attn_mask=attention_mask,
            need_weights=output_attentions,
            average_attn_weights=False,
        )
        past_key_value = None
        return attn_output, attn_output_weights, past_key_value
