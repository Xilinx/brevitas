"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Based on https://github.com/nod-ai/SHARK/blob/main/apps/language_models/scripts/sharded_vicuna_fp32.py

Copyright 2023 Nod.ai

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.


---- LLVM Exceptions to the Apache 2.0 License ----

As an exception, if, as a result of your compiling your source code, portions
of this Software are embedded into an Object form of such source code, you
may redistribute such embedded portions in such Object form without complying
with the conditions of Sections 4(a), 4(b) and 4(d) of the License.

In addition, if you combine or link compiled forms of this Software with
software that is licensed under the GPLv2 ("Combined Software") and if a
court of competent jurisdiction determines that the patent provision (Section
3), the indemnity provision (Section 9) or other Section of the License
conflicts with the conditions of the GPLv2, you may retroactively and
prospectively choose to deem waived or otherwise exclude such Section(s) of
the License, but only in their entirety and only with respect to the Combined
Software.
"""
import argparse
from io import BytesIO
from pathlib import Path
import re
from typing import List

import torch
from torch._decomp import get_decompositions
import torch_mlir
from torch_mlir import TensorPlaceholder
from tqdm import tqdm

from brevitas.backport.fx._symbolic_trace import wrap
from brevitas.backport.fx.experimental.proxy_tensor import make_fx
from brevitas_examples.llm.llm_quant.export import block_quant_layer_level_manager
from brevitas_examples.llm.llm_quant.export import BlockQuantProxyLevelManager
from brevitas_examples.llm.llm_quant.export import brevitas_layer_export_mode
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode
from brevitas_examples.llm.llm_quant.export import LinearWeightBlockQuantHandler
from brevitas_examples.llm.llm_quant.export import replace_call_fn_target
from brevitas_examples.llm.llm_quant.mlir_custom_mm import brevitas_matmul_rhs_group_quant_library


# Due a tracing issue this annotation needs to be
# in the same module (== file) from which make_fx is called
# We also can't directly annotate torch.ops.quant.matmul_rhs_group_quant
# and so we trace a placeholder first and then replace it post tracing
@wrap(visible_to_make_fx=True)
def matmul_rhs_group_quant_placeholder(*args, **kwargs):
    return torch.ops.quant.matmul_rhs_group_quant(*args, **kwargs)


class LinearWeightBlockQuantHandlerFwd(LinearWeightBlockQuantHandler):

    def forward(self, x):
        # Due a tracing issue the call to this fn needs to be
        # in the same module (== file) from which make_fx is called
        out = matmul_rhs_group_quant_placeholder(
            x, self.int_weight, self.scale, self.zero_point, self.bit_width, self.group_size)
        if self.bias is not None:
            out = out + self.bias.view(1, -1)
        return out


class FirstVicunaLayer(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, hidden_states, attention_mask, position_ids):
        outputs = self.model(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids, use_cache=True)
        next_hidden_states = outputs[0]
        past_key_value_out0, past_key_value_out1 = (outputs[-1][0], outputs[-1][1])

        return (next_hidden_states, past_key_value_out0, past_key_value_out1)


class SecondVicunaLayer(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
            self, hidden_states, attention_mask, position_ids, past_key_value0, past_key_value1):
        outputs = self.model(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=(past_key_value0, past_key_value1),
            use_cache=True)
        next_hidden_states = outputs[0]
        past_key_value_out0, past_key_value_out1 = (outputs[-1][0], outputs[-1][1])

        return (next_hidden_states, past_key_value_out0, past_key_value_out1)


def write_in_dynamic_inputs0(module, dynamic_input_size):
    new_lines = []
    for line in module.splitlines():
        line = re.sub(f"{dynamic_input_size}x", "?x", line)
        if "?x" in line:
            line = re.sub("tensor.empty\(\)", "tensor.empty(%dim)", line)
        line = re.sub(f" {dynamic_input_size},", " %dim,", line)
        if "tensor.empty" in line and "?x?" in line:
            line = re.sub("tensor.empty\(%dim\)", "tensor.empty(%dim, %dim)", line)
        if "arith.cmpi" in line:
            line = re.sub(f"c{dynamic_input_size}", "dim", line)
        new_lines.append(line)
    new_module = "\n".join(new_lines)
    return new_module


def write_in_dynamic_inputs1(module, dynamic_input_size):
    new_lines = []
    for line in module.splitlines():
        if "dim_42 =" in line:
            continue
        if f"%c{dynamic_input_size}_i64 =" in line:
            new_lines.append("%dim_42 = tensor.dim %arg1, %c3 : tensor<1x1x1x?xf32>")
            new_lines.append(f"%dim_42_i64 = arith.index_cast %dim_42 : index to i64")
            continue
        line = re.sub(f"{dynamic_input_size}x", "?x", line)
        if "?x" in line:
            line = re.sub("tensor.empty\(\)", "tensor.empty(%dim_42)", line)
        line = re.sub(f" {dynamic_input_size},", " %dim_42,", line)
        if "tensor.empty" in line and "?x?" in line:
            line = re.sub(
                "tensor.empty\(%dim_42\)",
                "tensor.empty(%dim_42, %dim_42)",
                line,
            )
        if "arith.cmpi" in line:
            line = re.sub(f"c{dynamic_input_size}", "dim_42", line)
        new_lines.append(line)
    new_module = "\n".join(new_lines)
    return new_module


def compile_vicuna_layer(
    export_context_manager,
    export_class,
    vicuna_layer,
    hidden_states,
    attention_mask,
    position_ids,
    past_key_value0=None,
    past_key_value1=None,
):
    hidden_states_placeholder = TensorPlaceholder.like(hidden_states, dynamic_axes=[1])
    attention_mask_placeholder = TensorPlaceholder.like(attention_mask, dynamic_axes=[2, 3])
    position_ids_placeholder = TensorPlaceholder.like(position_ids, dynamic_axes=[1])

    if past_key_value0 is None and past_key_value1 is None:
        with export_context_manager(vicuna_layer, export_class):
            fx_g = make_fx(
                vicuna_layer,
                decomposition_table=get_decompositions([
                    torch.ops.aten.embedding_dense_backward,
                    torch.ops.aten.native_layer_norm_backward,
                    torch.ops.aten.slice_backward,
                    torch.ops.aten.select_backward,
                    torch.ops.aten.norm.ScalarOpt_dim,
                    torch.ops.aten.native_group_norm,
                    torch.ops.aten.upsample_bilinear2d.vec,
                    torch.ops.aten.split.Tensor,
                    torch.ops.aten.split_with_sizes,]),
            )(hidden_states, attention_mask, position_ids)
    else:
        with export_context_manager(vicuna_layer, export_class):
            fx_g = make_fx(
                vicuna_layer,
                decomposition_table=get_decompositions([
                    torch.ops.aten.embedding_dense_backward,
                    torch.ops.aten.native_layer_norm_backward,
                    torch.ops.aten.slice_backward,
                    torch.ops.aten.select_backward,
                    torch.ops.aten.norm.ScalarOpt_dim,
                    torch.ops.aten.native_group_norm,
                    torch.ops.aten.upsample_bilinear2d.vec,
                    torch.ops.aten.split.Tensor,
                    torch.ops.aten.split_with_sizes,]),
            )(hidden_states, attention_mask, position_ids, past_key_value0, past_key_value1)

    def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
        removed_indexes = []
        for node in fx_g.graph.nodes:
            if node.op == "output":
                assert (len(node.args) == 1), "Output node must have a single argument"
                node_arg = node.args[0]
                if isinstance(node_arg, (list, tuple)):
                    node_arg = list(node_arg)
                    node_args_len = len(node_arg)
                    for i in range(node_args_len):
                        curr_index = node_args_len - (i + 1)
                        if node_arg[curr_index] is None:
                            removed_indexes.append(curr_index)
                            node_arg.pop(curr_index)
                    node.args = (tuple(node_arg),)
                    break

        if len(removed_indexes) > 0:
            fx_g.graph.lint()
            fx_g.graph.eliminate_dead_code()
            fx_g.recompile()
        removed_indexes.sort()
        return removed_indexes

    def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
        """
        Replace tuple with tuple element in functions that return one-element tuples.
        Returns true if an unwrapping took place, and false otherwise.
        """
        unwrapped_tuple = False
        for node in fx_g.graph.nodes:
            if node.op == "output":
                assert (len(node.args) == 1), "Output node must have a single argument"
                node_arg = node.args[0]
                if isinstance(node_arg, tuple):
                    if len(node_arg) == 1:
                        node.args = (node_arg[0],)
                        unwrapped_tuple = True
                        break

        if unwrapped_tuple:
            fx_g.graph.lint()
            fx_g.recompile()
        return unwrapped_tuple

    def transform_fx(fx_g):
        for node in fx_g.graph.nodes:
            if node.op == "call_function":
                if node.target in [torch.ops.aten.empty]:
                    # aten.empty should be filled with zeros.
                    with fx_g.graph.inserting_after(node):
                        new_node = fx_g.graph.call_function(torch.ops.aten.zero_, args=(node,))
                        node.append(new_node)
                        node.replace_all_uses_with(new_node)
                        new_node.args = (node,)
        fx_g.graph.lint()

    transform_fx(fx_g)
    replace_call_fn_target(
        fx_g, src=matmul_rhs_group_quant_placeholder, target=torch.ops.quant.matmul_rhs_group_quant)

    fx_g.recompile()
    removed_none_indexes = _remove_nones(fx_g)
    was_unwrapped = _unwrap_single_tuple_return(fx_g)

    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()

    print("FX_G recompile")

    def strip_overloads(gm):
        """
        Modifies the target of graph nodes in :attr:`gm` to strip overloads.
        Args:
            gm(fx.GraphModule): The input Fx graph module to be modified
        """
        for node in gm.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload):
                node.target = node.target.overloadpacket
        gm.recompile()

    strip_overloads(fx_g)
    ts_g = torch.jit.script(fx_g)
    return ts_g


def compile_to_vmfb(inputs, layers, export_context_manager, export_class, is_first=True):
    mlirs = []
    for idx, layer in tqdm(enumerate(layers), desc="Getting mlirs"):
        if is_first:
            mlir_path = Path(f"{idx}_0.mlir")
            vmfb_path = Path(f"{idx}_0.vmfb")
        else:
            mlir_path = Path(f"{idx}_1.mlir")
            vmfb_path = Path(f"{idx}_1.vmfb")
        if vmfb_path.exists():
            continue
        if mlir_path.exists():
            f_ = open(mlir_path, "rb")
            bytecode = f_.read()
            f_.close()
        else:
            hidden_states_placeholder = TensorPlaceholder.like(inputs[0], dynamic_axes=[1])
            attention_mask_placeholder = TensorPlaceholder.like(inputs[1], dynamic_axes=[3])
            position_ids_placeholder = TensorPlaceholder.like(inputs[2], dynamic_axes=[1])
            if not is_first:
                pkv0_placeholder = TensorPlaceholder.like(inputs[3], dynamic_axes=[2])
                pkv1_placeholder = TensorPlaceholder.like(inputs[4], dynamic_axes=[2])
            print(f"Compiling layer {idx} mlir")
            if is_first:
                ts_g = compile_vicuna_layer(
                    export_context_manager, export_class, layer, inputs[0], inputs[1], inputs[2])
                module = torch_mlir.compile(
                    ts_g, (hidden_states_placeholder, inputs[1], inputs[2]),
                    output_type="torch",
                    backend_legal_ops=["quant.matmul_rhs_group_quant"],
                    extra_library=brevitas_matmul_rhs_group_quant_library,
                    use_tracing=False,
                    verbose=False)
            else:
                ts_g = compile_vicuna_layer(
                    export_context_manager,
                    export_class,
                    layer,
                    inputs[0],
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    inputs[4])
                module = torch_mlir.compile(
                    ts_g,
                    (
                        inputs[0],
                        attention_mask_placeholder,
                        inputs[2],
                        pkv0_placeholder,
                        pkv1_placeholder),
                    output_type="torch",
                    backend_legal_ops=["quant.matmul_rhs_group_quant"],
                    extra_library=brevitas_matmul_rhs_group_quant_library,
                    use_tracing=False,
                    verbose=False)

            if is_first:
                module = write_in_dynamic_inputs0(str(module), 137)
                bytecode = module.encode("UTF-8")
                bytecode_stream = BytesIO(bytecode)
                bytecode = bytecode_stream.read()

            else:
                module = write_in_dynamic_inputs1(str(module), 138)
                if idx in [0, 5, 6, 7]:
                    module_str = module
                    module_str = module_str.splitlines()
                    new_lines = []
                    for line in module_str:
                        if len(line) < 1000:
                            new_lines.append(line)
                        else:
                            new_lines.append(line[:999])
                    module_str = "\n".join(new_lines)
                    f1_ = open(f"{idx}_1_test.mlir", "w+")
                    f1_.write(module_str)
                    f1_.close()

                bytecode = module.encode("UTF-8")
                bytecode_stream = BytesIO(bytecode)
                bytecode = bytecode_stream.read()

            f_ = open(mlir_path, "wb")
            f_.write(bytecode)
            f_.close()
        mlirs.append(bytecode)

    return mlirs


def sharded_weight_group_export(model, no_custom_packed_export):

    # SAMPLE_INPUT_LEN is used for creating mlir with dynamic inputs,
    # which is currently an increadibly hacky proccess
    # please don't change it
    SAMPLE_INPUT_LEN = 137

    placeholder_input0 = (
        torch.zeros([1, SAMPLE_INPUT_LEN, 4096]),
        torch.zeros([1, 1, SAMPLE_INPUT_LEN, SAMPLE_INPUT_LEN]),
        torch.zeros([1, SAMPLE_INPUT_LEN], dtype=torch.int64))

    placeholder_input1 = (
        torch.zeros([1, 1, 4096]),
        torch.zeros([1, 1, 1, SAMPLE_INPUT_LEN + 1]),
        torch.zeros([1, 1], dtype=torch.int64),
        torch.zeros([1, 32, SAMPLE_INPUT_LEN, 128]),
        torch.zeros([1, 32, SAMPLE_INPUT_LEN, 128]))

    if no_custom_packed_export:
        export_context_manager = brevitas_proxy_export_mode
        export_class = BlockQuantProxyLevelManager
    else:
        export_context_manager = brevitas_layer_export_mode
        # generate an export_class with the handler declared above
        export_class = block_quant_layer_level_manager(
            export_handlers=[LinearWeightBlockQuantHandlerFwd])

    layers0 = [FirstVicunaLayer(layer) for layer in model.model.layers]
    mlirs0 = compile_to_vmfb(
        placeholder_input0, layers0, export_context_manager, export_class, is_first=True)

    layers1 = [SecondVicunaLayer(layer) for layer in model.model.layers]
    mlirs1 = compile_to_vmfb(
        placeholder_input1, layers1, export_context_manager, export_class, is_first=False)
