# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from inspect import signature
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn

from brevitas import nn as qnn
from brevitas.fx import map_arg
from brevitas.fx import Node
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL

__all__ = [
    'module_class_name',
    'replace_all_uses_except',
    'signature_keys',
    'is_subseq',
    'find_node_for_module',
    'get_module_name_and_parent',
    'set_module',
    'get_module',
    'del_module',
    'replace_module',
    'remove_weight_orig',
    'name_from_module',
    'matches_module_pattern',
    'get_output_channels',
    'get_output_channel_dim',
    'power_iteration']

CONV_TRANSPOSED = (
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    qnn.QuantConvTranspose1d,
    qnn.QuantConvTranspose2d,
    qnn.QuantConvTranspose3d)


def module_class_name(m: torch.nn.Module):
    module = m.__class__.__module__
    if module is None or module == str.__class__.__module__:
        full_name = m.__class__.__name__
    else:
        full_name = module + '.' + m.__class__.__name__
    return full_name


def replace_all_uses_except(to_replace: Node, replace_with: 'Node', exceptions=()):
    """
    Replace all users of ``to_replace`` with the Node ``replace_with``, except when
    the user is in exceptions.

    Args:
        to_replace (Node): The node to replace all uses of.
        replace_with (Node): The node to replace all uses of ``to_replace`` with.
        exceptions (List[Node]): The user nodes that should be affected.

    Returns:
        The list of Nodes on which this change was made.
    """
    to_process = list(to_replace.users)
    for use_node in to_process:

        def maybe_replace_node(n: Node) -> Node:
            if n == to_replace and use_node not in exceptions:
                return replace_with
            else:
                return n

        new_args = map_arg(use_node.args, maybe_replace_node)
        new_kwargs = map_arg(use_node.kwargs, maybe_replace_node)
        assert isinstance(new_args, tuple)
        assert isinstance(new_kwargs, dict)
        if hasattr(use_node, '_update_args_kwargs'):
            use_node._update_args_kwargs(new_args, new_kwargs)
        elif hasattr(use_node, '_Node__update_args_kwargs'):
            use_node._Node__update_args_kwargs(new_args, new_kwargs)
        else:
            raise RuntimeError("Cannot update args-kwargs. Please open an issue to report this")
    return to_process


def signature_keys(module_class):
    return signature(module_class).parameters.keys()


def is_subseq(seq, subseq):
    return any(subseq == seq[i:len(subseq) + i] for i in range(len(seq) - len(subseq) + 1))


def find_node_for_module(graph_model, target_module) -> Optional[Node]:
    """
    Find the graph node corresponding to a module instance by matching its identity.
    """
    for node in graph_model.graph.nodes:
        if node.op == 'call_module':
            module = get_module(graph_model, node.target)
            if id(module) == id(target_module):
                return node
    return None


def get_module_name_and_parent(model, fully_qualified_module_name):
    supermodule = model
    prefix_list = fully_qualified_module_name.split('.')
    module_name = prefix_list[-1]
    prefix_list = prefix_list[:-1]  # exclude module name
    for prefix in prefix_list:
        if prefix:  # exclude empty prefix
            supermodule = getattr(supermodule, prefix)
    return module_name, supermodule


def set_module(model, module, fully_qualified_module_name):
    module_name, supermodule = get_module_name_and_parent(model, fully_qualified_module_name)
    setattr(supermodule, module_name, module)


def get_module(model, fully_qualified_module_name):
    name_atoms = fully_qualified_module_name.split('.')
    attr_itr = model
    for i, atom in enumerate(name_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Nonexistent module {'.'.join(name_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def del_module(model, fully_qualified_module_name):
    module_name, supermodule = get_module_name_and_parent(model, fully_qualified_module_name)
    del supermodule._modules[module_name]


def name_from_module(model, module):
    for name, m in model.named_modules():
        if m is module:
            return name
    return None


def replace_module(model, old_module, new_module):
    if isinstance(new_module, nn.Module):
        new_module = new_module.train() if old_module.training else new_module.eval()
    name = name_from_module(model, old_module)
    set_module(model, new_module, name)


# https://github.com/pytorch/pytorch/blob/v1.8.1/torch/fx/_experimental/fuser.py
# Works for length 2 patterns with 2 modules
def matches_module_pattern(pattern: Iterable, node: Node, modules: Dict[str, Any]):
    if len(node.args) == 0:
        return False
    nodes: Tuple[Any, Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


def is_conv_transposed(module):
    return isinstance(module, CONV_TRANSPOSED)


def get_output_channel_dim(module):
    if is_conv_transposed(module):
        return 1
    else:
        return 0


def get_output_channels(module):
    return module.weight.shape[get_output_channel_dim(module)]


def get_node(graph_model, name):
    for node in graph_model.graph.nodes:
        if node.target == name:
            return node


def is_quant_module(module):
    return isinstance(module, QuantWBIOL)


def remove_weight_orig(model: nn.Module):
    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig'):
            del module.weight_orig


def power_iteration(
        H: torch.Tensor,
        steps: int,
        eps: float = 1e-12,
        device: Union[str, torch.device] = 'cpu',
        seed: int = 42) -> torch.Tensor:
    """
    Power iteration to estimate the dominant eigenvalue of the Hessian.
    Accuracy improves with `steps`. Several choices below reduce run-to-run variation.
    """
    # device='cpu' by default; CPU reductions tend to be more deterministic than on GPU
    if not isinstance(device, torch.device):
        device = torch.device(device)
    # fixing generator mitigates run-to-run variation with negligible impact on convergence
    g = torch.Generator(device=device).manual_seed(seed)
    b_k = torch.rand(H.shape[1], device=device, dtype=H.dtype, generator=g)
    # normalize H by its absmax for numerical stability; rescale the eigenvalue before returning
    c_k = H.max().abs()
    H_k = (H / c_k).to(device)
    for _ in range(steps):
        b_k1 = torch.mv(H_k, b_k)
        b_k1_norm = torch.norm(b_k1)
        b_k = b_k1 / (b_k1_norm + eps)
    # Rayleigh quotient on the normalized H, then undo the absmax scaling; return on H.device
    max_eigenval = torch.dot(b_k, torch.mv(H_k, b_k)).to(c_k.device) * c_k
    return max_eigenval
