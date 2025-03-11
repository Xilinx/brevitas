# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from inspect import isclass
import operator
from typing import Dict, List, Optional

import torch
import torch.nn as nn

# TODO: Deprecate PyTorch 1.11
try:
    from torch.nn.utils.parametrize import type_before_parametrizations
except ImportError:
    from brevitas.utils.torch_utils import type_before_parametrizations

from brevitas.graph.base import InsertModuleCallAfter
from brevitas.graph.base import ModuleInstanceToModuleInstance
from brevitas.graph.base import ModuleToModuleByInstance
from brevitas.graph.utils import del_module
from brevitas.graph.utils import get_module
from brevitas.utils.logging import setup_logger

logging = setup_logger(__name__)

ADD_FNS = [torch.add, operator.add, operator.iadd]

ADD_METHODS = ['add', 'add_']

SIGN_PRESERVING_MODULES = (
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.PixelShuffle,
    nn.PixelUnshuffle,
    nn.Identity)

PRECISION_PRESERVING_MODULES = (
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.PixelShuffle,
    nn.PixelUnshuffle,
    nn.Identity)

MAX_RESIDUAL_ITERS = 9999

BATCH_NORM = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def inp_placeholder_handler(model, input_quantizer):
    """
    Add Quantization step at the input of the network.
    """
    rewriters = []
    if input_quantizer is None:
        return model
    for node in model.graph.nodes:
        if node.op == 'placeholder':
            act_quant, kwargs_act_quant = input_quantizer
            inp_quant = act_quant(**kwargs_act_quant)
            name = node.name + '_quant'
            model.add_module(name, inp_quant)
            rewriters.append(InsertModuleCallAfter(name, node))
    for rewriter in rewriters:
        model = rewriter.apply(model)
    return model


def are_inputs_unsigned(model, node, is_unsigned_list, quant_act_map, unsigned_act_tuple):
    for inp_node in node.all_input_nodes:
        if inp_node.op == 'call_module':
            inp_module = get_module(model, inp_node.target)
            if isinstance(inp_module, tuple(quant_act_map.keys())) and isinstance(
                    inp_module, unsigned_act_tuple):
                is_unsigned_list.append(True)
            elif isinstance(inp_module, tuple(SIGN_PRESERVING_MODULES)):
                are_inputs_unsigned(
                    model, inp_node, is_unsigned_list, quant_act_map, unsigned_act_tuple)
            elif hasattr(inp_module, 'input_quant'):
                is_unsigned_list.append(not inp_module.input_quant.is_signed)
            else:
                is_unsigned_list.append(False)
        elif inp_node.op == 'call_function':
            if inp_node.target in [torch.reshape, torch.flatten, torch.transpose, torch.cat
                                  ] + ADD_FNS:
                are_inputs_unsigned(
                    model, inp_node, is_unsigned_list, quant_act_map, unsigned_act_tuple)
            else:
                is_unsigned_list.append(False)
        elif inp_node.op == 'call_method':
            if inp_node.target in ['view', 'reshape', 'flatten', 't', 'permute'] + ADD_METHODS:
                are_inputs_unsigned(
                    model, inp_node, is_unsigned_list, quant_act_map, unsigned_act_tuple)
            else:
                is_unsigned_list.append(False)
    return all(is_unsigned_list)


def _tensor_quant_in_list(act_quant, module_list, same_sign):
    tq = act_quant.fused_activation_quant_proxy.tensor_quant
    for m in module_list:
        if m is None:
            continue
        m_tq = m.fused_activation_quant_proxy.tensor_quant
        if same_sign and m_tq is tq:
            return True
        elif not same_sign and m_tq.scaling_impl is tq.scaling_impl and m_tq.int_scaling_impl is tq.int_scaling_impl:
            return True
    return False


def are_inputs_quantized_and_aligned(model, node, quantized_modules_list, quant_act_map, same_sign):
    """
    Check if the inputs to `node` are quantized and aligned.
    If same_sign=True, aligned means that the inputs should have same sign and scale factor.
    Otherwise, they need to have only the same scale factors.
    If none of the previous conditions are met (e.g., FP input, or not aligned scales), the function
    returns False.
    """
    for inp_node in node.all_input_nodes:
        if inp_node.op == 'call_module':
            inp_module = get_module(model, inp_node.target)
            if isinstance(inp_module, tuple(PRECISION_PRESERVING_MODULES)) and (
                    not same_sign or
                (same_sign and isinstance(inp_module, tuple(SIGN_PRESERVING_MODULES)))):
                are_inputs_quantized_and_aligned(
                    model, inp_node, quantized_modules_list, quant_act_map, same_sign)
            elif hasattr(inp_module, 'act_quant'):
                aq = inp_module.act_quant
                if _tensor_quant_in_list(aq, quantized_modules_list, same_sign):
                    continue
                quantized_modules_list.append(aq)
            else:
                quantized_modules_list.append(None)
        elif inp_node.op == 'call_function':
            if inp_node.target in [torch.reshape, torch.flatten, torch.transpose]:
                are_inputs_quantized_and_aligned(
                    model, inp_node, quantized_modules_list, quant_act_map, same_sign)
            elif inp_node.target is torch.cat:
                are_inputs_quantized_and_aligned(
                    model, inp_node, quantized_modules_list, quant_act_map, True)
            elif inp_node.target in ADD_FNS:
                are_inputs_quantized_and_aligned(
                    model, inp_node, quantized_modules_list, quant_act_map, False)
            else:
                quantized_modules_list.append(None)
        elif inp_node.op == 'call_method':
            if inp_node.target in ['view', 'reshape', 'flatten', 't', 'permute']:
                are_inputs_quantized_and_aligned(
                    model, inp_node, quantized_modules_list, quant_act_map, same_sign)
            elif inp_node.target in ADD_METHODS:
                are_inputs_quantized_and_aligned(
                    model, inp_node, quantized_modules_list, quant_act_map, False)
            else:
                quantized_modules_list.append(None)
    if None in quantized_modules_list:
        return False
    elif len(quantized_modules_list) > 1:
        return False
    else:
        return True


def output_quant_handler(
        model,
        node,
        rewriters,
        is_sign_preserving,
        quant_identity_map,
        quant_act_map,
        unsigned_act_tuple=None):
    """
    Starting from `node`, check if any of the users requires requantization (i.e., it does not have
    an act_quant attribute). In that case, the functions adds a requantization step to which all the
    branches are connected. If another branch has its own requantization step, there will be two
    consecutive for that branch.
    """
    if is_sign_preserving and unsigned_act_tuple is None:
        raise RuntimeError("Missing information for output_quant_handler")
    quant_module = None
    quant_module_name = None
    for user in node.users:
        output_quant = True
        if user.op == 'call_module':
            user_module = get_module(model, user.target)
            if hasattr(user_module, 'act_quant'):
                output_quant = False
            elif isinstance(user_module, BATCH_NORM):
                # If the user is BatchNorm, check BN's users and potentially requentize at
                # the output of BN
                output_quant = False
                output_quant_handler(
                    model,
                    user,
                    rewriters,
                    is_sign_preserving,
                    quant_identity_map,
                    quant_act_map,
                    unsigned_act_tuple)
        if output_quant:
            if quant_module_name is None and quant_module is None:
                if is_sign_preserving and are_inputs_unsigned(
                        model, node, [], quant_act_map, unsigned_act_tuple):
                    quant_module_class, quant_module_kwargs = quant_identity_map['unsigned']
                else:
                    quant_module_class, quant_module_kwargs = quant_identity_map['signed']
                quant_module = quant_module_class(**quant_module_kwargs)
                quant_module_name = node.name + '_output_quant'
                model.add_module(quant_module_name, quant_module)
                rewriters.append(InsertModuleCallAfter(quant_module_name, node))


def recursive_input_handler(
        model,
        node,
        shared_quant_identity_name,
        shared_quant_identity,
        rewriters,
        quant_identity_map,
        align_input_quant_fn,
        align_sign):
    """
    For a given CAT or ADD node, iterate through its inputs to make sure they are correctly aligned.
    """
    for inp_node in node.all_input_nodes:
        if inp_node.op == 'call_module':
            module = get_module(model, inp_node.target)
            # Precision preserving modules can be safely traversed
            # In case align_sign is True, the modules should also be sign preserving
            if isinstance(module, tuple(PRECISION_PRESERVING_MODULES)) and (
                    not align_sign or
                (align_sign and isinstance(module, tuple(SIGN_PRESERVING_MODULES)))):
                recursive_input_handler(
                    model,
                    inp_node,
                    shared_quant_identity_name,
                    shared_quant_identity,
                    rewriters,
                    quant_identity_map,
                    align_input_quant_fn,
                    align_sign)
            else:
                # Based on the current module, generate an align_output object
                align_output = align_input_quant_fn(
                    module,
                    shared_quant_identity,
                    shared_quant_identity_name,
                    quant_identity_map,
                    align_sign)
                # If it is a tuple, it is a combination of QuantAct and its configuration that
                # will replace the current inp_node module
                if isinstance(align_output, tuple):
                    quant_module_class, quant_module_kwargs = align_output
                    rewriter = ModuleToModuleByInstance(
                        module, quant_module_class, **quant_module_kwargs)
                    rewriters.append(rewriter)
                # If it is a nn.Module, it is an already instatiated QuantAct that will replace
                # the current inp_node module
                elif isinstance(align_output, torch.nn.Module):
                    rewriter = ModuleInstanceToModuleInstance(module, align_output)
                    rewriters.append(rewriter)
                # If it is a string, we simply add a requantization activation after the current
                # inp_node module
                elif isinstance(align_output, str):
                    rewriters.append(InsertModuleCallAfter(shared_quant_identity_name, inp_node))
                else:
                    assert align_output is None, f"align_output {str(align_output)} not supported."
        elif inp_node.op == 'call_function' and inp_node.target in [
                torch.flatten, torch.reshape, torch.transpose]:
            recursive_input_handler(
                model,
                inp_node,
                shared_quant_identity_name,
                shared_quant_identity,
                rewriters,
                quant_identity_map,
                align_input_quant_fn,
                align_sign)
        elif inp_node.op == 'call_function' and inp_node.target is torch.cat:
            recursive_input_handler(
                model,
                inp_node,
                shared_quant_identity_name,
                shared_quant_identity,
                rewriters,
                quant_identity_map,
                align_input_quant_fn,
                align_sign=True)
        elif inp_node.op == 'call_method' and inp_node.target in [
                'view', 'reshape', 'flatten', 'transpose']:
            recursive_input_handler(
                model,
                inp_node,
                shared_quant_identity_name,
                shared_quant_identity,
                rewriters,
                quant_identity_map,
                align_input_quant_fn,
                align_sign)
        else:
            rewriters.append(InsertModuleCallAfter(shared_quant_identity_name, inp_node))


def _get_quant_module(model, node, quant_identity_map, quant_act_map, unsigned_act_tuple):
    """
    Generate a QuantIdentity node that will be used for requantization step around a node.
    If all inputs to that node are unsigned, the QuantIdentity will also be unsigned, otherwise it
    is signed.
    """
    if are_inputs_unsigned(model, node, [], quant_act_map, unsigned_act_tuple):
        quant_module_class, quant_module_kwargs = quant_identity_map['unsigned']
    else:
        quant_module_class, quant_module_kwargs = quant_identity_map['signed']
    quant_module = quant_module_class(**quant_module_kwargs)
    quant_module_name = node.name + '_quant'
    model.add_module(quant_module_name, quant_module)
    return quant_module, quant_module_name


def residual_handler(
        model, quant_identity_map, quant_act_map, unsigned_act_tuple, align_input_quant_fn):
    iter = 0

    def is_converged(model):

        for node in model.graph.nodes:
            if (node.op == 'call_function' and node.target in ADD_FNS + [torch.cat] or
                    node.op == 'call_method' and node.target in ADD_METHODS):
                rewriters = []
                # If the op is CAT, check that inputs have same sign, and in recursive_input_handler
                # force that the sign is aligned
                same_sign = node.target is torch.cat

                # If input to the CAT or ADD node are quantized and aligned correctly, continue to
                # the next node
                if are_inputs_quantized_and_aligned(model,
                                                    node, [],
                                                    quant_act_map,
                                                    same_sign=same_sign):
                    continue

                # Generate a QuantIdentity module to use for alignement of the inputs
                shared_quant_identity, shared_quant_identity_name = _get_quant_module(
                    model, node, quant_identity_map, quant_act_map, unsigned_act_tuple)

                # Recursively, for every input node, traverse the graph to determine how to quantize
                # and align that input node.
                recursive_input_handler(
                    model,
                    node,
                    shared_quant_identity_name,
                    shared_quant_identity,
                    rewriters,
                    quant_identity_map,
                    align_input_quant_fn,
                    align_sign=same_sign)
                for rewriter in rewriters:
                    model = rewriter.apply(model)
                model.graph.lint()
                model.recompile()
                return False
        return True

    while not is_converged(model):
        iter += 1
        if iter == MAX_RESIDUAL_ITERS:
            raise RuntimeError(
                "Residual handler could not find a solution to align scale factors "
                "across ADDs and CATs")

    return model


def add_output_quant_handler(model, quant_identity_map, quant_act_map, unsigned_act_tuple):
    """
    Check the output of every add node, to decide whether it needs to be requantized or not, based
    on the logic of output_quant_handler.
    """
    rewriters = []
    for node in model.graph.nodes:
        if (node.op == 'call_function' and node.target in ADD_FNS or
                node.op == 'call_method' and node.target in ADD_METHODS):
            output_quant_handler(
                model,
                node,
                rewriters,
                is_sign_preserving=True,
                quant_identity_map=quant_identity_map,
                quant_act_map=quant_act_map,
                unsigned_act_tuple=unsigned_act_tuple)
    for rewriter in rewriters:
        model = rewriter.apply(model)
    return model


def act_handler(model, layer_map):
    for node in model.graph.nodes:
        rewriters = []
        if node.op == 'call_module':
            module = get_module(model, node.target)
            if isinstance(module, tuple(layer_map.keys())):
                if layer_map[type_before_parametrizations(module)] is not None:
                    quant_module_class, quant_module_kwargs = layer_map[type_before_parametrizations(module)]
                    quant_module = quant_module_class(**quant_module_kwargs)
                    # Check for activation equalization mul nodes
                    if len(node.users) == 1:
                        user_node = list(node.users.keys())[0]
                        if user_node.name.endswith('act_eq_mul'):
                            # We update activation_impl so that the mul node is executed before quantization
                            act_module = quant_module.act_quant.fused_activation_quant_proxy.activation_impl
                            mul_module = get_module(model, user_node.target)
                            quant_module.act_quant.fused_activation_quant_proxy.activation_impl = torch.nn.Sequential(
                                *[act_module, mul_module])
                            # The mul node added during equalization is removed
                            user_node.replace_all_uses_with(node)
                            model.graph.erase_node(user_node)
                            del_module(model, user_node.target)
                    rewriter = ModuleInstanceToModuleInstance(module, quant_module)
                    rewriters.append(rewriter)
        for rewriter in rewriters:
            model = rewriter.apply(model)
    return model


def layer_handler(
        model,
        layer_map,
        requantize_output,
        quant_identity_map=None,
        quant_act_map=None,
        unsigned_act_tuple=None):
    """
    Replace FP weight layers with their corresponding quantized version
    """
    if requantize_output and (quant_identity_map is None or quant_act_map is None or
                              unsigned_act_tuple is None):
        raise RuntimeError("Missing information to requantize output")
    for node in model.graph.nodes:
        rewriters = []
        if node.op == 'call_module':
            module = get_module(model, node.target)
            if isinstance(module, tuple(layer_map.keys())):
                if requantize_output:
                    if len(node.users) > 1 and all(['getitem' in n.name for n in node.users]):
                        for n in node.users:
                            if len(n.users) > 0:
                                output_quant_handler(
                                    model,
                                    n,
                                    rewriters,
                                    is_sign_preserving=isinstance(module, SIGN_PRESERVING_MODULES),
                                    quant_identity_map=quant_identity_map,
                                    quant_act_map=quant_act_map,
                                    unsigned_act_tuple=unsigned_act_tuple)
                    else:
                        output_quant_handler(
                            model,
                            node,
                            rewriters,
                            is_sign_preserving=isinstance(module, SIGN_PRESERVING_MODULES),
                            quant_identity_map=quant_identity_map,
                            quant_act_map=quant_act_map,
                            unsigned_act_tuple=unsigned_act_tuple)
                if layer_map[type_before_parametrizations(module)] is not None:
                    quant_module_class, quant_module_kwargs = layer_map[type_before_parametrizations(module)]
                    # Quantize the input if is not quantized, input_quant is not specified,
                    # and the quant_identity_map is provided.
                    if not are_inputs_quantized_and_aligned(
                            model, node, [], quant_act_map, same_sign=False
                    ) and not 'input_quant' in quant_module_kwargs and len(quant_identity_map) > 0:
                        # Define the source node where to add the requantization step
                        previous_node = node.all_input_nodes[0]
                        # Exclude all the other possible users
                        previous_node_users = list(previous_node.users.keys())
                        previous_node_users.remove(node)

                        act_quant, kwargs_act_quant = quant_identity_map['signed']
                        inp_quant = act_quant(**kwargs_act_quant)
                        name = node.name + '_input_quant'
                        model.add_module(name, inp_quant)
                        rewriter = InsertModuleCallAfter(
                            name, previous_node, tuple(previous_node_users))
                        rewriters.append(rewriter)
                    rewriter = ModuleToModuleByInstance(
                        module, quant_module_class, **quant_module_kwargs)
                    rewriters.append(rewriter)
        for rewriter in rewriters:
            model = rewriter.apply(model)
    return model


def _module_class_name(module_class_or_str):
    name = module_class_or_str.__module__ + '.' + module_class_or_str.__name__ if isclass(
        module_class_or_str) else module_class_or_str
    return name


def find_module(
        model: nn.Module,
        layer_map: Dict[nn.Module, Optional[Dict]],
        module_to_replace: List,
        name_blacklist,
        prefix=''):
    """
    Iterate through the model looking at immediate children of every module to look for supported modules.
    This allows us to stop the search when we meet a top-level module that is supported.
    Specifically, it allows to map nn.MultiheadAttetion to its quantized counterpart and not its
    Linear submodules.
    """
    if _module_class_name(type_before_parametrizations(model)) in layer_map.keys():
        module_to_replace.append(model)
    else:
        for name, module in model.named_children():
            full_name = prefix + '.' + name if prefix != '' else name
            if name_blacklist is not None and full_name in name_blacklist:
                logging.info(f"Skipping {full_name} module from quantization")
                continue
            find_module(module, layer_map, module_to_replace, name_blacklist, full_name)


def layerwise_layer_handler(
        model: nn.Module, layer_map: Dict[nn.Module, Optional[Dict]], name_blacklist=None):
    """
    Replace FP weight layers with their corresponding quantized version
    """
    # Normalize all module lookups to fully qualified strings
    layer_map = {_module_class_name(m): v for m, v in layer_map.items()}
    module_to_replace = []
    find_module(model, layer_map, module_to_replace, name_blacklist)
    rewriters = []
    for module in module_to_replace:
        if layer_map[_module_class_name(type_before_parametrizations(module))] is not None:
            quant_module_class, quant_module_kwargs = layer_map[_module_class_name(type_before_parametrizations(module))]
            rewriter = ModuleToModuleByInstance(module, quant_module_class, **quant_module_kwargs)
            rewriters.append(rewriter)
    for rewriter in rewriters:
        model = rewriter.apply(model)
    return model
