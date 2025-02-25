"""
Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Adapted from https://github.com/mit-han-lab/llm-awq, released under the following LICENSE:

MIT License

Copyright (c) 2023 MIT HAN Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import List

from torch import nn
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from brevitas.graph.equalize import _get_input_axis
from brevitas.graph.equalize import _get_output_axis
from brevitas.graph.equalize import EqualizationIndexes
from brevitas_examples.llm.llm_quant.awq.graph import RegionAWQ


def retrieve_block_awq_regions(block: nn.Module) -> List[RegionAWQ]:
    block_regions = []
    if isinstance(block, OPTDecoderLayer):
        # attention input
        block_regions.append(
            RegionAWQ(
                srcs={"self_attn_layer_norm": None},
                sinks={
                    "self_attn.q_proj": None, "self_attn.k_proj": None, "self_attn.v_proj": None},
                acts=(),
                name_to_module={
                    "self_attn_layer_norm": block.self_attn_layer_norm,
                    "self_attn.q_proj": block.self_attn.q_proj,
                    "self_attn.k_proj": block.self_attn.k_proj,
                    "self_attn.v_proj": block.self_attn.v_proj,},
                block=block.self_attn,
                use_kwargs=True,
            ))
        # attn out
        block_regions.append(
            RegionAWQ(
                srcs={"self_attn.v_proj": None},
                sinks={"self_attn.out_proj": None},
                acts=(),
                name_to_module={
                    "self_attn.v_proj": block.self_attn.v_proj,
                    "self_attn.out_proj": block.self_attn.out_proj,},
            ))
        # fc1
        block_regions.append(
            RegionAWQ(
                srcs={"final_layer_norm": None},
                sinks={"fc1": None},
                acts=(),
                name_to_module={
                    "final_layer_norm": block.final_layer_norm,
                    "fc1": block.fc1,},
            ))
        # fc2
        block_regions.append(
            RegionAWQ(
                srcs={"fc1": None},
                sinks={"fc2": None},
                acts=(),
                name_to_module={
                    "fc1": block.fc1,
                    "fc2": block.fc2,},
            ))
    elif isinstance(block, (LlamaDecoderLayer, Qwen2DecoderLayer)):
        # attention input
        block_regions.append(
            RegionAWQ(
                srcs={"input_layernorm": None},
                sinks={
                    "self_attn.q_proj": None,
                    "self_attn.k_proj": None,
                    "self_attn.v_proj": None,},
                acts=(),
                name_to_module={
                    "input_layernorm": block.input_layernorm,
                    "self_attn.q_proj": block.self_attn.q_proj,
                    "self_attn.k_proj": block.self_attn.k_proj,
                    "self_attn.v_proj": block.self_attn.v_proj,},
                block=block.self_attn,
                use_kwargs=True,
            ))
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if block.self_attn.v_proj.weight.shape == block.self_attn.o_proj.weight.shape:
            block_regions.append(
                RegionAWQ(
                    srcs={"self_attn.v_proj": None},
                    sinks={"self_attn.o_proj": None},
                    acts=(),
                    name_to_module={
                        "self_attn.v_proj": block.self_attn.v_proj,
                        "self_attn.o_proj": block.self_attn.o_proj,},
                ))
        # fc1
        block_regions.append(
            RegionAWQ(
                srcs={"post_attention_layernorm": None},
                sinks={
                    "mlp.gate_proj": None, "mlp.up_proj": None},
                acts=(),
                name_to_module={
                    "post_attention_layernorm": block.post_attention_layernorm,
                    "mlp.gate_proj": block.mlp.gate_proj,
                    "mlp.up_proj": block.mlp.up_proj,},
                block=block.mlp,
            ))
        # fc2
        block_regions.append(
            RegionAWQ(
                srcs={"mlp.up_proj": None},
                sinks={"mlp.down_proj": None},
                acts=(),
                name_to_module={
                    "mlp.up_proj": block.mlp.up_proj,
                    "mlp.down_proj": block.mlp.down_proj,},
            ))
    elif isinstance(block, BloomBlock):
        # attention input
        block_regions.append(
            RegionAWQ(
                srcs={"input_layernorm": None},
                sinks={"self_attention.query_key_value": None},
                acts=(),
                name_to_module={
                    "input_layernorm": block.input_layernorm,
                    "self_attention.query_key_value": block.self_attention.query_key_value,},
                block=block,
                use_kwargs=True,
            ))
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469

        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1
        block_regions.append(
            RegionAWQ(
                srcs={"post_attention_layernorm": None},
                sinks={"mlp.dense_h_to_4h": None},
                acts=(),
                name_to_module={
                    "post_attention_layernorm": block.post_attention_layernorm,
                    "mlp.dense_h_to_4h": block.mlp.dense_h_to_4h,},
                block=block,
                use_kwargs=True,
            ))
        # fc2
        block_regions.append(
            RegionAWQ(
                srcs={"mlp.gelu_impl": None},
                sinks={"mlp.dense_4h_to_h": None},
                acts=(),
                name_to_module={
                    "mlp.gelu_impl": block.mlp.gelu_impl,
                    "mlp.dense_4h_to_h": block.mlp.dense_4h_to_h,},
            ))
    elif "mpt" in str(block.__class__).lower():
        # attention input
        block_regions.append(
            RegionAWQ(
                srcs={"norm_1": None},
                sinks={"attn.Wqkv": None},
                acts=(),
                name_to_module={
                    "norm_1": block.norm_1,
                    "attn.Wqkv": block.attn.Wqkv,},
                block=block.attn,
                use_kwargs=True,
            ))
        # attn out
        block_regions.append(
            RegionAWQ(
                srcs={"attn.Wqkv": None},
                sinks={"attn.out_proj": None},
                acts=(),
                name_to_module={
                    "attn.Wqkv": block.attn.Wqkv,
                    "attn.out_proj": block.attn.out_proj,},
            ))
        # fc1
        block_regions.append(
            RegionAWQ(
                srcs={"norm_2": None},
                sinks={"ffn.up_proj": None},
                acts=(),
                name_to_module={
                    "norm_2": block.norm_2,
                    "ffn.up_proj": block.ffn.up_proj,},
                block=block.ffn,
            ))

        # fc2
        block_regions.append(
            RegionAWQ(
                srcs={"ffn.act": None},
                sinks={"ffn.down_proj": None},
                acts=(),
                name_to_module={
                    "ffn.act": block.ffn.act,
                    "ffn.down_proj": block.ffn.down_proj,},
            ))
    elif "falcon" in str(block.__class__).lower():
        # attn out
        # Haotian: TBD: need to handle repeated scales for MQ

        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1, as long as it is scaled, everything is screwed up
        if "falcon-7b" in str(block.__class__).lower():
            block_regions.append(
                RegionAWQ(
                    srcs={"input_layernorm": None},
                    sinks={
                        "mlp.dense_h_to_4h": None,
                        "self_attention.query_key_value": None,},
                    acts=(),
                    name_to_module={
                        "input_layernorm": block.input_layernorm,
                        "mlp.dense_h_to_4h": block.mlp.dense_h_to_4h,
                        "self_attention.query_key_value": block.self_attention.query_key_value,},
                    block=block,
                    use_kwargs=True,
                ))
        elif "falcon-40b" in str(block.__class__).lower():
            block_regions.append(
                RegionAWQ(
                    srcs={"ln_attn": None},
                    sinks={"self_attention.query_key_value": None},
                    acts=(),
                    name_to_module={
                        "ln_attn": block.ln_attn,
                        "self_attention.query_key_value": block.self_attention.query_key_value,},
                    block=block,
                    use_kwargs=True,
                ))
            block_regions.append(
                RegionAWQ(
                    srcs={"ln_mlp": None},
                    sinks={"mlp.dense_h_to_4h": None},
                    acts=(),
                    name_to_module={
                        "ln_mlp": block.ln_mlp,
                        "mlp.dense_h_to_4h": block.mlp.dense_h_to_4h,},
                    block=block,
                    use_kwargs=True,
                ))
        else:
            raise NotImplementedError(
                "Unknown Falcon architecture, currently only falcon-7b and falcon-40b are supported"
            )
        # fc2
        block_regions.append(
            RegionAWQ(
                srcs={"mlp.act": None},
                sinks={"mlp.dense_4h_to_h": None},
                acts=(),
                name_to_module={
                    "mlp.act": block.mlp.act,
                    "mlp.dense_4h_to_h": block.mlp.dense_4h_to_h,},
                block=block,
                use_kwargs=True,
            ))
    elif "bigcode" in str(block.__class__).lower():
        block_regions.append(
            RegionAWQ(
                srcs={"ln_1": None},
                sinks={"attn.c_attn": None},
                acts=(),
                name_to_module={
                    "ln_1": block.ln_1,
                    "attn.c_attn": block.attn.c_attn,},
                block=block.attn,
                use_kwargs=True,
            ))
        # fc1
        block_regions.append(
            RegionAWQ(
                srcs={"ln_2": None},
                sinks={"mlp.c_fc": None},
                acts=(),
                name_to_module={
                    "ln_2": block.ln_2,
                    "mlp.c_fc": block.mlp.c_fc,},
                block=block.mlp,
            ))
        # fc2
        block_regions.append(
            RegionAWQ(
                srcs={"mlp.act": None},
                sinks={"mlp.c_proj": None},
                acts=(),
                name_to_module={
                    "mlp.act": block.mlp.act,
                    "mlp.c_proj": block.mlp.c_proj,},
            ))
    elif "neox" in str(block.__class__).lower():
        block_regions.append(
            RegionAWQ(
                srcs={"input_layernorm": None},
                sinks={"attention.query_key_value": None},
                acts=(),
                name_to_module={
                    "input_layernorm": block.input_layernorm,
                    "attention.query_key_value": block.attention.query_key_value,},
                block=block.attention,
                use_kwargs=True,
            ))
        # fc1
        block_regions.append(
            RegionAWQ(
                srcs={"post_attention_layernorm": None},
                sinks={"mlp.dense_h_to_4h": None},
                acts=(),
                name_to_module={
                    "post_attention_layernorm": block.post_attention_layernorm,
                    "mlp.dense_h_to_4h": block.mlp.dense_h_to_4h,},
                block=block.mlp,
            ))
        # fc2
        block_regions.append(
            RegionAWQ(
                srcs={"mlp.act": None},
                sinks={"mlp.dense_4h_to_h": None},
                acts=(),
                name_to_module={
                    "mlp.act": block.mlp.act,
                    "mlp.dense_4h_to_h": block.mlp.dense_4h_to_h,},
            ))
    else:
        raise NotImplementedError(f"{type(block)} not supported yet!")

    # Insert EqualizationIndexes for sources and sinks
    for region in block_regions:
        for src_full_name, src_name in zip(region.srcs, region.srcs_names):
            m = region.name_to_module[src_name]
            axis = _get_output_axis(m)
            if hasattr(m, "weight"):
                region.srcs[src_full_name] = EqualizationIndexes(0, m.weight.shape[axis], 0)

        for sink_full_name, sink_name in zip(region.sinks, region.sinks_names):
            m = region.name_to_module[sink_name]
            axis = _get_input_axis(m)
            if hasattr(m, "weight"):
                region.sinks[sink_full_name] = EqualizationIndexes(0, m.weight.shape[axis], 0)

    return block_regions
