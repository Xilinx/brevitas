# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
import operator

import torch
import torch.nn as nn

import brevitas
from brevitas.graph.base import InsertModuleCallAfter
from brevitas.graph.base import ModuleInstanceToModuleInstance
from brevitas.graph.base import ModuleToModuleByInstance
from brevitas.graph.utils import get_module

ADD_FNS = [torch.add, operator.add, operator.iadd]

ADD_METHODS = ['add', 'add_']
CAT = brevitas.original_cat

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
    nn.AdaptiveAvgPool3d)

PRECISION_PRESERVING_MODULES = (
    nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)


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
            elif hasattr(inp_module, 'is_quant_act_signed'):
                is_unsigned_list.append(not inp_module.is_quant_act_signed)
            else:
                is_unsigned_list.append(False)
        elif inp_node.op == 'call_function':
            if inp_node.target in [torch.reshape, torch.flatten, torch.transpose, CAT] + ADD_FNS:
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
            if isinstance(inp_module, tuple(quant_act_map.keys())):
                quantized_modules_list.append(None)
            elif isinstance(inp_module, tuple(PRECISION_PRESERVING_MODULES)) and (
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
            elif inp_node.target is CAT:
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
        quant_act_map=None,
        unsigned_act_tuple=None):
    """
    Starting from `node`, check if any of the users requires requantization (i.e., it does not have
    an act_quant attribute). In that case, the functions adds a requantization step to which all the
    branches are connected. If another branch has its own requantization step, there will be two
    consecutive for that branch.
    """
    if is_sign_preserving and (quant_act_map is None or unsigned_act_tuple is None):
        raise RuntimeError("Missing information for output_quant_handler")
    quant_module = None
    quant_module_name = None
    for user in node.users:
        output_quant = True
        if user.op == 'call_module':
            user_module = get_module(model, user.target)
            if hasattr(user_module, 'act_quant'):
                output_quant = False
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
                torch.flatten, torch.reshape, torch.transpose, operator.getitem,
                operator.__getitem__]:
            recursive_input_handler(
                model,
                inp_node,
                shared_quant_identity_name,
                shared_quant_identity,
                rewriters,
                quant_identity_map,
                align_input_quant_fn,
                align_sign)
        elif inp_node.op == 'call_function' and inp_node.target is CAT:
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

    def is_converged(model):

        for node in model.graph.nodes:
            if (node.op == 'call_function' and node.target in ADD_FNS + [CAT] or
                    node.op == 'call_method' and node.target in ADD_METHODS):
                rewriters = []
                # If the op is CAT, check that inputs have same sign, and in recursive_input_handler
                # force that the sign is aligned
                same_sign = node.target is CAT

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
        continue

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


def layer_handler(
    model,
    layer_map,
    requantize_output,
    quant_identity_map=dict(),
    quant_act_map=dict(),
    unsigned_act_tuple=dict()):
    """
    Replace FP weight layers with their corresponding quantized version
    """
    if requantize_output and (len(quant_identity_map) == 0 or len(quant_act_map) == 0 or
                              len(unsigned_act_tuple) == 0):
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
                if layer_map[type(module)] is not None:
                    quant_module_class, quant_module_kwargs = layer_map[type(module)]
                    # Quantize the input if is not quantized, input_quant is not specified,
                    # and the quant_identity_map is provided.
                    # The last requirement is needed to avoid requantizing the input to activations
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
