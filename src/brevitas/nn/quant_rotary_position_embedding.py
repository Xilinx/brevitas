import torch
import torch.nn as nn
from .mixin.base import QuantLayerMixin
from brevitas.common import ExportMixin
from torch.autograd import Function


## Define the actual RoPE layer
def rotate_half(x):
    """Rotates half the hidden dims of the input using transposes."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(inp, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    embed = (inp * cos) + (rotate_half(inp) * sin)
    return embed


def frequency_position_encoding( dim, seq_len, base, position_ids=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
    inv_freq = inv_freq[None, :, None] 
    # x: [bs, num_attention_heads, seq_len, head_size]
    if position_ids is None:
        # position_ids = torch.arange(seq_len)
        ## Where do we get Batch size here? 
        position_ids = torch.arange(seq_len).expand((1, seq_len)).float()
        position_ids_expanded = position_ids[:, None, :] 
    # ## Reshape the inv_freq tensor to match the shape of the position_ids tensor
    inv_freq_expanded = inv_freq 
    # Use broadcasting to match the shape instead of the expand function

    # Calculate the frequency embeddings
    freqs = torch.matmul(inv_freq_expanded, position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000):
        super(RotaryPositionEmbedding, self).__init__()
        self.cos, self.sin = frequency_position_encoding(dim, max_position_embeddings, base)

    def forward(self, inp, position_ids=None):
        out = apply_rotary_pos_emb(inp, self.cos, self.sin, position_ids)
        return out
    
    
## Define Brevitas version of the RoPE layer
# class RotaryPositionEmbeddingFN(Function):
    # @staticmethod
    # def symbolic(g,inp, position_ids=None):
    #     ret = g.op(
    #         ## How does this map to the MSFT CustomOp attributes and inputs?
    #         'com.microsoft.RotaryEmbedding',
    #         inp)
    #     return ret
class QuantRotaryPositionEmbedding(QuantLayerMixin, ExportMixin, RotaryPositionEmbedding):

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 return_quant_tensor: bool = True,):
        QuantLayerMixin.__init__(self, return_quant_tensor)
        ExportMixin.__init__(self)
        RotaryPositionEmbedding.__init__(self, dim, max_position_embeddings, base)

    def forward(self, inp, position_ids=None):
        if self.export_mode:
            out = self.export_handler(inp)
            return out
        # super().forward(*args, **kwargs)
        return super().forward(inp, position_ids)
