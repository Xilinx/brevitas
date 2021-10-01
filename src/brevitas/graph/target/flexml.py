import operator

import torch
from torch import nn

from brevitas import config
import brevitas.nn as qnn
from brevitas.core.utils import StatelessBuffer
from brevitas.quant import Int16Bias
from brevitas.quant import Int8WeightPerTensorFixedPoint
from brevitas.quant import Int8ActPerTensorFixedPoint
from brevitas.quant import Uint8ActPerTensorFixedPoint
from brevitas.quant import Uint8ActPerTensorFixedPointMaxInit
from brevitas.graph.base import ModuleToModuleByClass, ModuleToModuleByInstance
from brevitas.graph.base import ModuleInstanceToModuleInstance
from brevitas.graph.base import InsertModuleCallAfter
from brevitas.graph.standardize import TorchFunctionalToModule
from brevitas.graph.standardize import DuplicateSharedStatelessModule
from brevitas.graph.standardize import MeanMethodToAdaptiveAvgPool2d
from brevitas.graph.standardize import DisableLastReturnQuantTensor
from brevitas.graph.per_input import AdaptiveAvgPoolToAvgPool
from brevitas.graph.per_input import AvgPoolToDepthwiseConv
from brevitas.graph.fixed_point import MergeBatchNorm
from brevitas.graph.fixed_point import MoveSplitBatchNormBeforeCat
from brevitas.graph.fixed_point import CollapseConsecutiveConcats
from brevitas.graph.equalize import EqualizeGraph
from brevitas.fx import value_trace
from brevitas.graph.utils import get_module


ADD_FNS = [torch.add, operator.add, operator.iadd]

ADD_METHODS = ['add', 'add_']


QUANT_WBIOL_MAP = {
    nn.Conv1d: qnn.QuantConv1d,
    nn.Conv2d: qnn.QuantConv2d,
    nn.ConvTranspose1d: qnn.QuantConvTranspose1d,
    nn.ConvTranspose2d: qnn.QuantConvTranspose2d,
    nn.BatchNorm1d: qnn.BatchNorm1dToQuantScaleBias,
    nn.BatchNorm2d: qnn.BatchNorm2dToQuantScaleBias,
    nn.Linear: qnn.QuantLinear}


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


class FlexMLQuantLeakyReLU(nn.Module):

    def __init__(
            self,
            negative_slope,
            alpha_quant=qnn.QuantIdentity(Uint8ActPerTensorFixedPoint, bit_width=16),
            input_quant=qnn.QuantIdentity(Int8ActPerTensorFixedPoint, bit_width=16, scaling_stats_momentum = None),
            output_quant=qnn.QuantIdentity(Int8ActPerTensorFixedPoint, return_quant_tensor=True)):
        super(FlexMLQuantLeakyReLU, self).__init__()
        self.alpha_quant = alpha_quant
        self.input_quant = input_quant
        self.output_quant = output_quant
        self.negative_slope = StatelessBuffer(torch.tensor(negative_slope))

    def forward(self, inp):
        quant_inp = self.input_quant(inp)
        quant_alpha = self.alpha_quant(self.negative_slope())
        quant_alpha_out = self.input_quant(quant_inp * quant_alpha)
        out = torch.max(quant_inp, quant_alpha_out)
        out = self.output_quant(out)
        return out


def flexml_inp_placeholder_handler(model):
    rewriters = []
    for node in model.graph.nodes:
        if node.op == 'placeholder':
            inp_quant = qnn.QuantIdentity(Int8ActPerTensorFixedPoint, return_quant_tensor=True)
            name = node.name + '_quant'
            model.add_module(name, inp_quant)
            rewriters.append(InsertModuleCallAfter(name, node))
    for rewriter in rewriters:
        model = rewriter.apply(model)
    return model


def are_inputs_unsigned(model, node, is_unsigned_list):
    for inp_node in node.all_input_nodes:
        if inp_node.op == 'call_module':
            inp_module = get_module(model, inp_node.target)
            if isinstance(inp_module, (nn.ReLU, nn.ReLU6)):
                is_unsigned_list.append(True)
            elif isinstance(inp_module, tuple(SIGN_PRESERVING_MODULES)):
                are_inputs_unsigned(model, inp_node, is_unsigned_list)
            elif isinstance(inp_module, (qnn.QuantReLU, qnn.QuantIdentity, qnn.QuantHardTanh)):
                is_unsigned_list.append(not inp_module.is_quant_act_signed)
            else:
                is_unsigned_list.append(False)
        elif inp_node.op == 'call_function':
            if inp_node.target in [torch.reshape, torch.flatten, torch.transpose, torch.cat] + ADD_FNS:
                are_inputs_unsigned(model, inp_node, is_unsigned_list)
            else:
                is_unsigned_list.append(False)
        elif inp_node.op == 'call_method':
            if inp_node.target in ['view', 'reshape', 'flatten', 't', 'permute'] + ADD_METHODS:
                are_inputs_unsigned(model, inp_node, is_unsigned_list)
            else:
                is_unsigned_list.append(False)
    return all(is_unsigned_list)


def _tensor_quant_in_list(tensor_quant, module_list, same_sign):
    for m in module_list:
        if m is None:
            continue
        if same_sign and m is tensor_quant:
            return True
        elif not same_sign and m.scaling_impl is tensor_quant.scaling_impl and m.int_scaling_impl is tensor_quant.int_scaling_impl:
            return True
    return False


def are_inputs_quantized(model, node, quantized_modules_list, same_sign):
    for inp_node in node.all_input_nodes:
        if inp_node.op == 'call_module':
            inp_module = get_module(model, inp_node.target)
            if isinstance(inp_module, (nn.ReLU, nn.ReLU6)):
                quantized_modules_list.append(None)
            elif isinstance(inp_module, tuple(SIGN_PRESERVING_MODULES)):
                are_inputs_quantized(model, inp_node, quantized_modules_list, same_sign)
            elif isinstance(inp_module, (qnn.QuantReLU, qnn.QuantIdentity, qnn.QuantHardTanh)):
                tq = inp_module.act_quant.fused_activation_quant_proxy.tensor_quant
                if _tensor_quant_in_list(tq, quantized_modules_list, same_sign):
                    continue
                quantized_modules_list.append(tq)
            elif isinstance(inp_module, (FlexMLQuantLeakyReLU)):
                tq = inp_module.output_quant.act_quant.fused_activation_quant_proxy.tensor_quant
                if _tensor_quant_in_list(tq, quantized_modules_list, same_sign):
                    continue
                quantized_modules_list.append(tq)
            else:
                quantized_modules_list.append(None)
        elif inp_node.op == 'call_function':
            if inp_node.target in [torch.reshape, torch.flatten, torch.transpose]:
                are_inputs_quantized(model, inp_node, quantized_modules_list, same_sign)
            elif inp_node.target is torch.cat:
                are_inputs_quantized(model, inp_node, quantized_modules_list, True)
            elif inp_node.target in ADD_FNS:
                are_inputs_quantized(model, inp_node, quantized_modules_list, False)
            else:
                quantized_modules_list.append(None)
        elif inp_node.op == 'call_method':
            if inp_node.target in ['view', 'reshape', 'flatten', 't', 'permute']:
                are_inputs_quantized(model, inp_node, quantized_modules_list, same_sign)
            elif inp_node.target in ADD_METHODS:
                are_inputs_quantized(model, inp_node, quantized_modules_list, False)
            else:
                quantized_modules_list.append(None)
    if None in quantized_modules_list:
        return False
    elif len(quantized_modules_list) > 1:
        return False
    else:
        return True


def output_quant_handler(model, node, rewriters, is_sign_preserving):
    quant_module = None
    quant_module_name = None
    for user in node.users:
        output_quant = True
        if user.op == 'call_module':
            user_module = get_module(model, user.target)
            if isinstance(user_module, (qnn.QuantReLU, qnn.QuantIdentity, FlexMLQuantLeakyReLU)):
                output_quant = False
        if output_quant:
            if quant_module_name is None and quant_module is None:
                if is_sign_preserving and are_inputs_unsigned(model, node, []):
                    quant_module = qnn.QuantIdentity(
                        act_quant=Uint8ActPerTensorFixedPoint, return_quant_tensor=True)
                else:
                    quant_module = qnn.QuantIdentity(
                        act_quant=Int8ActPerTensorFixedPoint, return_quant_tensor=True)
                quant_module_name = node.name + '_output_quant'
                model.add_module(quant_module_name, quant_module)
            rewriters.append(InsertModuleCallAfter(quant_module_name, node))


def cat_input_handler(model, node, quant_identity_name, quant_identity, rewriters):
    for inp_node in node.all_input_nodes:
        if inp_node.op == 'call_module':
            module = get_module(model, inp_node.target)
            if isinstance(module, tuple(SIGN_PRESERVING_MODULES)):
                cat_input_handler(model, inp_node, quant_identity_name, quant_identity, rewriters)
            elif isinstance(module, qnn.QuantReLU):
                rewriter = ModuleToModuleByInstance(
                    module, qnn.QuantReLU,
                    # WORKAROUND
                    # TODO act_quant=quant_identity.act_quant is currently broken
                    # because it overrides act_impl even though it shouldn't
                    signed=quant_identity.act_quant.is_signed,
                    narrow_range=quant_identity.act_quant.is_narrow_range,
                    tensor_quant=quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant)
                rewriters.append(rewriter)
            elif isinstance(module, qnn.QuantIdentity):
                rewriter = ModuleInstanceToModuleInstance(
                    module, quant_identity)
                rewriters.append(rewriter)
            elif isinstance(module, FlexMLQuantLeakyReLU):
                rewriter = ModuleToModuleByInstance(
                    module, FlexMLQuantLeakyReLU, output_quant=quant_identity)
                rewriters.append(rewriter)
            else:
                rewriters.append(InsertModuleCallAfter(quant_identity_name, inp_node))
        elif inp_node.op == 'call_function' and inp_node.target in [torch.flatten, torch.reshape, torch.transpose]:
            cat_input_handler(model, inp_node, quant_identity_name, quant_identity, rewriters)
        elif inp_node.op == 'call_function' and inp_node.target is torch.cat:
            cat_input_handler(model, inp_node, quant_identity_name, quant_identity, rewriters)
        elif inp_node.op == 'call_method' and inp_node.target in ['view', 'reshape', 'flatten', 'transpose']:
            cat_input_handler(model, inp_node, quant_identity_name, quant_identity, rewriters)
        else:
            rewriters.append(InsertModuleCallAfter(quant_identity_name, inp_node))


def add_input_handler(model, node, quant_identity_name, quant_identity, rewriters):
    for inp_node in node.all_input_nodes:
        if inp_node.op == 'call_module':
            module = get_module(model, inp_node.target)
            if isinstance(module, tuple(SIGN_PRESERVING_MODULES)):
                add_input_handler(model, inp_node, quant_identity_name, quant_identity, rewriters)
            elif isinstance(module, qnn.QuantReLU):
                rewriter = ModuleToModuleByInstance(
                    module, qnn.QuantReLU,
                    act_quant=Uint8ActPerTensorFixedPoint,
                    scaling_impl=quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl,
                    int_scaling_impl=quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl,
                    return_quant_tensor=True)
                rewriters.append(rewriter)
            elif isinstance(module, qnn.QuantIdentity):
                if module.is_quant_act_signed == quant_identity.is_quant_act_signed:
                    rewriters.append(ModuleInstanceToModuleInstance(module, quant_identity))
                else:
                    assert not module.is_quant_act_signed and quant_identity.is_quant_act_signed
                    rewriter = ModuleToModuleByInstance(
                        module, qnn.QuantIdentity,
                        act_quant=Uint8ActPerTensorFixedPoint,
                        scaling_impl=quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl,
                        int_scaling_impl=quant_identity.act_quant.fused_activation_quant_proxy.tensor_quant.int_scaling_impl,
                        return_quant_tensor=True)
                    rewriters.append(rewriter)
            elif isinstance(module, FlexMLQuantLeakyReLU):
                rewriter = ModuleToModuleByInstance(
                    module, FlexMLQuantLeakyReLU, output_quant=quant_identity)
                rewriters.append(rewriter)
            else:
                rewriters.append(InsertModuleCallAfter(quant_identity_name, inp_node))
        elif inp_node.op == 'call_function' and inp_node.target in [torch.flatten, torch.reshape, torch.transpose]:
            add_input_handler(model, inp_node, quant_identity_name, quant_identity, rewriters)
        elif inp_node.op == 'call_function' and inp_node.target is torch.cat:
            cat_input_handler(model, inp_node, quant_identity_name, quant_identity, rewriters)
        elif inp_node.op == 'call_method' and inp_node.target in ['view', 'reshape', 'flatten', 'transpose']:
            add_input_handler(model, inp_node, quant_identity_name, quant_identity, rewriters)
        else:
            rewriters.append(InsertModuleCallAfter(quant_identity_name, inp_node))


def flexml_act_handler(model):
    rewriters = []
    for node in model.graph.nodes:
        if node.op == 'call_module':
            module = get_module(model, node.target)
            if isinstance(module, nn.ReLU):
                rewriter = ModuleToModuleByInstance(
                    module, qnn.QuantReLU,
                    act_quant=Uint8ActPerTensorFixedPoint, return_quant_tensor=True)
                rewriters.append(rewriter)
            elif isinstance(module, nn.ReLU6):
                rewriter = ModuleToModuleByInstance(
                    module, qnn.QuantReLU,
                    act_quant=Uint8ActPerTensorFixedPointMaxInit,
                    max_val=6., return_quant_tensor=True)
                rewriters.append(rewriter)
            elif isinstance(module, nn.LeakyReLU):
                rewriter = ModuleToModuleByInstance(
                    module, FlexMLQuantLeakyReLU)
                rewriters.append(rewriter)
    for rewriter in rewriters:
        model = rewriter.apply(model)
    return model


def _get_quant_module(model, node):
    if are_inputs_unsigned(model, node, []):
        quant_module = qnn.QuantIdentity(Uint8ActPerTensorFixedPoint,
                                         return_quant_tensor=True)
    else:
        quant_module = qnn.QuantIdentity(Int8ActPerTensorFixedPoint,
                                         return_quant_tensor=True)
    quant_module_name = node.name + '_quant'
    model.add_module(quant_module_name, quant_module)
    return quant_module, quant_module_name


def flexml_residual_handler(model):

    def is_converged(model):

        for node in model.graph.nodes:
            if (node.op == 'call_function' and node.target in ADD_FNS + [torch.cat]
                    or node.op == 'call_method' and node.target in ADD_METHODS):
                rewriters = []
                if node.target is torch.cat:
                    if are_inputs_quantized(model, node, [], True):
                        continue
                    quant_module, quant_module_name = _get_quant_module(model, node)
                    cat_input_handler(model, node, quant_module_name, quant_module, rewriters)
                else:
                    if are_inputs_quantized(model, node, [], False):
                        continue
                    quant_module, quant_module_name = _get_quant_module(model, node)
                    add_input_handler(model, node, quant_module_name, quant_module, rewriters)
                for rewriter in rewriters:
                    model = rewriter.apply(model)
                model.graph.lint()
                model.recompile()
                return False
        return True

    while not is_converged(model):
        continue

    return model


def flexml_add_output_quant_handler(model):
    rewriters = []
    for node in model.graph.nodes:
        if (node.op == 'call_function' and node.target in ADD_FNS
                or node.op == 'call_method' and node.target in ADD_METHODS):
            output_quant_handler(model, node, rewriters, is_sign_preserving=True)
    for rewriter in rewriters:
        model = rewriter.apply(model)
    return model


def flexml_wbiol_handler(model):
    rewriters = []
    for node in model.graph.nodes:
        if node.op == 'call_module':
            module = get_module(model, node.target)
            if isinstance(module, tuple(QUANT_WBIOL_MAP.keys())):
                output_quant_handler(model, node, rewriters, is_sign_preserving=False)
                rewriter = ModuleToModuleByInstance(
                    module, QUANT_WBIOL_MAP[type(module)],
                    weight_quant=Int8WeightPerTensorFixedPoint,
                    bias_quant=Int16Bias,
                    return_quant_tensor=True)
                rewriters.append(rewriter)
    for rewriter in rewriters:
        model = rewriter.apply(model)
    return model


def preprocess_flexml(model, equalization_iters = 0, **model_kwargs):
    training_state = model.training
    model.eval()
    model = value_trace(model, model_kwargs)
    model = TorchFunctionalToModule().apply(model)
    model = DuplicateSharedStatelessModule().apply(model)
    model = ModuleToModuleByClass(nn.ReLU6, nn.ReLU).apply(model)
    model = MeanMethodToAdaptiveAvgPool2d().apply(model)
    model = AdaptiveAvgPoolToAvgPool().apply(model, *model_kwargs.values())
    model = AvgPoolToDepthwiseConv().apply(model, *model_kwargs.values())
    model = CollapseConsecutiveConcats().apply(model)
    model = MoveSplitBatchNormBeforeCat().apply(model)
    model = MergeBatchNorm().apply(model)
    model = EqualizeGraph(equalization_iters).apply(model)
    model.train(training_state)
    return model


def quantize_flexml(graph_model):
    ignore_missing_keys_state = config.IGNORE_MISSING_KEYS
    config.IGNORE_MISSING_KEYS = True
    training_state = graph_model.training
    graph_model.eval()
    graph_model = flexml_inp_placeholder_handler(graph_model)
    graph_model = flexml_act_handler(graph_model)
    graph_model = flexml_add_output_quant_handler(graph_model)
    graph_model = flexml_residual_handler(graph_model)
    graph_model = flexml_wbiol_handler(graph_model)
    graph_model = DisableLastReturnQuantTensor().apply(graph_model)
    graph_model.train(training_state)
    config.IGNORE_MISSING_KEYS = ignore_missing_keys_state
    return graph_model