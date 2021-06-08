from inspect import signature
from typing import Tuple, Any, Iterable, Dict

import torch

from brevitas.fx import Node, map_arg

__all__ = [
    'module_class_name',
    'replace_all_users_except',
    'signature_keys',
    'is_subseq',
    'get_module_name_and_parent',
    'set_module',
    'get_module',
    'del_module',
    'replace_module',
    'name_from_module',
    'matches_module_pattern'
]


def module_class_name(m: torch.nn.Module):
    module = m.__class__.__module__
    if module is None or module == str.__class__.__module__:
        full_name = m.__class__.__name__
    else:
        full_name = module + '.' + m.__class__.__name__
    return full_name


def replace_all_users_except(to_replace: Node, replace_with: 'Node', exceptions=()):
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
        use_node._Node__update_args_kwargs(new_args, new_kwargs)
    return to_process


def signature_keys(module_class):
    return signature(module_class).parameters.keys()


def is_subseq(seq, subseq):
    return any(subseq == seq[i:len(subseq) + i] for i in range(len(seq) - len(subseq) + 1))


def get_module_name_and_parent(model, fully_qualified_module_name):
    supermodule = model
    prefix_list = fully_qualified_module_name.split('.')
    module_name = prefix_list[-1]
    prefix_list = prefix_list[:-1]  # exclude module name
    for prefix in prefix_list:
        if prefix:  # exclude empty prefix
            supermodule = supermodule._modules[prefix]
    return module_name, supermodule


def set_module(model, module, fully_qualified_module_name):
    module_name, supermodule = get_module_name_and_parent(model, fully_qualified_module_name)
    supermodule._modules[module_name] = module


def get_module(model, fully_qualified_module_name):
    named_modules = dict(model.named_modules())
    return named_modules[fully_qualified_module_name]


def del_module(model, fully_qualified_module_name):
    module_name, supermodule = get_module_name_and_parent(model, fully_qualified_module_name)
    del supermodule._modules[module_name]


def name_from_module(model, module):
    for name, m in model.named_modules():
        if m is module:
            return name
    return None


def replace_module(model, old_module, new_module):
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
