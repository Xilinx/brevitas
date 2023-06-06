from typing import Optional, Tuple

import torch
from torch import nn


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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if key_value_states is None:
            key_value_states = hidden_states
        if layer_head_mask is not None:
            raise RuntimeError("layer_head_mask is not supported.")
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze()
        attn_output, attn_output_weights = self.mha(
            hidden_states,
            key_value_states,
            key_value_states,
            attn_mask=attention_mask,
            need_weights=output_attentions,
            average_attn_weights=False)
        past_key_value = None
        return attn_output, attn_output_weights, past_key_value
