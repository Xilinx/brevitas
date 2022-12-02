# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import operator

import torch
from torch import nn

import brevitas
from brevitas import config
from brevitas.core.scaling.standalone import ConstScaling
from brevitas.core.scaling.standalone import ParameterScaling
from brevitas.graph.base import InsertModuleCallAfter
from brevitas.graph.base import ModuleInstanceToModuleInstance
from brevitas.graph.base import ModuleToModuleByInstance
from brevitas.graph.per_input import AvgPoolToQuantDepthwiseConv
from brevitas.graph.standardize import DisableLastReturnQuantTensor
from brevitas.graph.utils import get_module
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import Uint8ActPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloatMaxInit
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat

QUANT_AVGPOOL_MAP = {nn.AvgPool2d: qnn.flexml.FlexMLQuantAvgPool2d}

ADD_FNS = [torch.add, operator.add, operator.iadd]

ADD_METHODS = ['add', 'add_']
CAT = brevitas.original_cat

SIGN_PRESERVING_MODULES = [
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d]

QUANT_WBIOL_MAP = {
    nn.Conv1d: (
        qnn.QuantConv1d,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.Conv2d: (
        qnn.QuantConv2d,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.ConvTranspose1d: (
        qnn.QuantConvTranspose1d,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.ConvTranspose2d: (
        qnn.QuantConvTranspose2d,
        {
            'weight_quant': Int8WeightPerTensorFloat,
            'bias_quant': Int32Bias,
            'return_quant_tensor': True}),
    nn.Linear:
        (qnn.QuantLinear, {
            'weight_quant': Int8WeightPerTensorFloat, 'return_quant_tensor': True})}

UNSIGNED_ACT_TUPLE = (nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Hardsigmoid)

QUANT_ACT_MAP = {
    nn.ReLU: (qnn.QuantReLU, {
        'act_quant': Uint8ActPerTensorFloat, 'return_quant_tensor': True}),
    nn.ReLU6: (
        qnn.QuantReLU, {
            'act_quant': Uint8ActPerTensorFloatMaxInit, 'max_val': 6.,
            'return_quant_tensor': True}),
    nn.Hardtanh: (
        qnn.QuantHardTanh,
        {
            'act_quant': Int8ActPerTensorFloatMinMaxInit,
            'max_val': lambda module: module.max_val,
            'min_val': lambda module: module.min_val}),
    nn.Sigmoid:
        (qnn.QuantSigmoid, {
            'act_quant': Uint8ActPerTensorFloat,
            'return_quant_tensor': True,}),}

QUANT_IDENTITY_MAP = {
    'signed':
        (qnn.QuantIdentity, {
            'act_quant': Int8ActPerTensorFloat, 'return_quant_tensor': True}),
    'unsigned':
        (qnn.QuantIdentity, {
            'act_quant': Uint8ActPerTensorFloat, 'return_quant_tensor': True}),}


def align_input_quant(
        module, shared_quant_identity, shared_quant_identity_name, quant_identity_map, align_sign):
    """
    Based on the input module, the function decides how to align its output.
    """
    # If it is a QuantIdentity already, simply modify tensor_quant or the scaling implementations
    # based on whether we need to align the sign or not
    if isinstance(module, qnn.QuantIdentity):
        if align_sign or module.is_quant_act_signed == shared_quant_identity.is_quant_act_signed:
            return shared_quant_identity
        else:
            assert not module.is_quant_act_signed and shared_quant_identity.is_quant_act_signed
            quant_module_class, quant_module_kwargs = quant_identity_map['unsigned']
            return (
                quant_module_class,
                {
                    **quant_module_kwargs,
                    'scaling_impl':
                        shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                        .scaling_impl,
                    'int_scaling_impl':
                        shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                        .int_scaling_impl})
    elif hasattr(module, 'output_quant'):
        return (type(module), {'output_quant': shared_quant_identity})
    # If it is a QuantAct where the scaling can be determined through stats (thus through calibration),
    # then adapt its act_quant according to align_sign.
    elif hasattr(module, 'act_quant') and not isinstance(
            module.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl,
        (ParameterScaling, ConstScaling)):
        module_type = type(module)
        if align_sign:
            partial_config = {
                'signed':
                    shared_quant_identity.act_quant.is_signed,
                'tensor_quant':
                    shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant}
        else:
            partial_config = {
                'scaling_impl':
                    shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                    .scaling_impl,
                'int_scaling_impl':
                    shared_quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant
                    .int_scaling_impl}
        injector = module.act_quant.quant_injector.let(**partial_config)
        return module_type(act_quant=injector, return_quant_tensor=True)
    # In all other cases, return the name of the QuantIdentity that will be added at the output of
    # the module
    else:
        return shared_quant_identity_name


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
            if isinstance(inp_module, tuple(quant_act_map.keys())):
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
            elif isinstance(inp_module, tuple(SIGN_PRESERVING_MODULES)):
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
            # Sign preserving modules can be safely traversed
            if isinstance(module, tuple(SIGN_PRESERVING_MODULES)):
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


def act_handler(model, quant_act_map):
    """
    Replace FP activations with their corresponding quantized version
    """
    rewriters = []
    for node in model.graph.nodes:
        if node.op == 'call_module':
            module = get_module(model, node.target)
            for module_class in quant_act_map.keys():
                if isinstance(module, module_class):
                    quant_module_class, quant_module_kwargs = quant_act_map[type(module)]
                    rewriter = ModuleToModuleByInstance(
                        module, quant_module_class, **quant_module_kwargs)
                    rewriters.append(rewriter)
    for rewriter in rewriters:
        model = rewriter.apply(model)
    return model


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


def wbiol_handler(model, float_to_quant_wbiol_map, quant_identity_map):
    """
    Replace FP weight layers with their corresponding quantized version
    """
    rewriters = []
    for node in model.graph.nodes:
        if node.op == 'call_module':
            module = get_module(model, node.target)
            if isinstance(module, tuple(float_to_quant_wbiol_map.keys())):
                output_quant_handler(
                    model,
                    node,
                    rewriters,
                    is_sign_preserving=False,
                    quant_identity_map=quant_identity_map)
                quant_module_class, quant_module_kwargs = float_to_quant_wbiol_map[type(module)]
                rewriter = ModuleToModuleByInstance(
                    module, quant_module_class, **quant_module_kwargs)
                rewriters.append(rewriter)
    for rewriter in rewriters:
        model = rewriter.apply(model)
    return model


def avgpool_handler(
        model,
        *model_args,
        quant_identity_map,
        quant_act_map,
        quant_avg_pool_map,
        unsigned_act_tuple,
        avgpool_to_depthwise_conv=False,
        **model_kwargs):
    rewriters = []
    for node in model.graph.nodes:
        if node.op == 'call_module':
            module = get_module(model, node.target)
            if isinstance(module, tuple(quant_avg_pool_map.keys())):
                output_quant_handler(
                    model,
                    node,
                    rewriters,
                    is_sign_preserving=True,
                    quant_identity_map=quant_identity_map,
                    quant_act_map=quant_act_map,
                    unsigned_act_tuple=unsigned_act_tuple)
                if avgpool_to_depthwise_conv:
                    rewriter = AvgPoolToQuantDepthwiseConv(
                        weight_quant=Int8WeightPerTensorFixedPoint,
                        bias_quant=Int32Bias,
                        return_quant_tensor=True)
                else:
                    rewriter = ModuleToModuleByInstance(
                        module, quant_avg_pool_map[type(module)], return_quant_tensor=True)
                rewriters.append(rewriter)
    for rewriter in rewriters:
        if isinstance(rewriter, AvgPoolToQuantDepthwiseConv):
            model = rewriter.apply(model, *model_args, **model_kwargs)
        else:
            model = rewriter.apply(model)
    return model


def quantize(
        graph_model,
        *model_args,
        avgpool_to_depthwise_conv=False,
        quant_identity_map=QUANT_IDENTITY_MAP,
        quant_wbiol_map=QUANT_WBIOL_MAP,
        quant_act_map=QUANT_ACT_MAP,
        quant_avg_pool_map=QUANT_AVGPOOL_MAP,
        unsigned_act_tuple=UNSIGNED_ACT_TUPLE,
        **model_kwargs):
    ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
    config.IGNORE_MISSING_KEYS = True
    training_state = graph_model.training
    graph_model.eval()
    graph_model = inp_placeholder_handler(
        graph_model, input_quantizer=quant_identity_map.get('signed', None))
    graph_model = act_handler(graph_model, quant_act_map)
    graph_model = add_output_quant_handler(
        graph_model, quant_identity_map, quant_act_map, unsigned_act_tuple)
    graph_model = wbiol_handler(graph_model, quant_wbiol_map, quant_identity_map)
    graph_model = avgpool_handler(
        graph_model,
        *model_args,
        quant_identity_map=quant_identity_map,
        quant_act_map=quant_act_map,
        quant_avg_pool_map=quant_avg_pool_map,
        unsigned_act_tuple=unsigned_act_tuple,
        avgpool_to_depthwise_conv=avgpool_to_depthwise_conv,
        **model_kwargs)
    graph_model = residual_handler(
        graph_model, quant_identity_map, quant_act_map, unsigned_act_tuple, align_input_quant)
    graph_model = DisableLastReturnQuantTensor().apply(graph_model)
    graph_model.train(training_state)
    config.IGNORE_MISSING_KEYS = ignore_missing_keys_state
    return graph_model
