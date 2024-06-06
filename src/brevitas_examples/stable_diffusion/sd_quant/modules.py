from typing import Optional

from diffusers import DiffusionPipeline
from diffusers.models.attention_processor import Attention
import torch

from brevitas.graph.base import ModuleInstanceToModuleInstance
from brevitas.graph.base import ModuleToModuleByClass
from brevitas.nn.equalized_layer import EqualizedModule
from brevitas.nn.quant_activation import QuantIdentity
from brevitas.nn.quant_scale_bias import ScaleBias
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat
from brevitas.quant_tensor import _unpack_quant_tensor


class QuantAttention(Attention):

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
            softmax_output_quant=None,
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
            processor,
        )
        # self.output_q_quant = QuantIdentity(qkv_output_quant)
        # self.output_k_quant = QuantIdentity(qkv_output_quant)
        # self.output_v_quant = QuantIdentity(qkv_output_quant)
        self.output_softmax_quant = QuantIdentity(softmax_output_quant)
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
