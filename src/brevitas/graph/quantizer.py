import copy
from typing import ClassVar, Optional, List
from inspect import isclass
from enum import auto
import operator
from packaging import version

import torch
from torch import nn

from brevitas import torch_version
from brevitas import nn as qnn
from brevitas.inject.defaults import *
from brevitas.fx import brevitas_value_trace
from .rewriter import FnToModuleRewriter
from .rewriter import MethodToModuleRewriter
from .rewriter import ModuleToModuleRewriter
from .rewriter import MergeBatchNorm2d
from .rewriter import DuplicateSharedStatelessModule
from .rewriter import MeanMethodToAdaptiveAvgPool2d
from .rewriter import DisableBreakingReturnQuantTensor
from brevitas.utils.python_utils import AutoName


_io_map = {
    operator.add: qnn.QuantEltwiseAdd,  # todo this should be restricted to tensor types only
    operator.iadd: qnn.QuantEltwiseAdd,
    'add': qnn.QuantEltwiseAdd,
    'add_': qnn.QuantEltwiseAdd,
    torch.add: qnn.QuantEltwiseAdd,
    torch.cat: qnn.QuantCat}


_wbiol_map = {
    nn.Conv1d: qnn.QuantConv1d,
    nn.Conv2d: qnn.QuantConv2d,
    nn.ConvTranspose1d: qnn.QuantConvTranspose1d,
    nn.ConvTranspose2d: qnn.QuantConvTranspose2d,
    nn.Linear: qnn.QuantLinear}


_act_map = {
    nn.ReLU: qnn.QuantReLU,
    nn.ReLU6: qnn.QuantReLU,
    nn.Hardtanh: qnn.QuantHardTanh,
    nn.Sigmoid: qnn.QuantSigmoid,
    nn.Tanh: qnn.QuantTanh,
    torch.relu: qnn.QuantReLU,
    torch.relu_: qnn.QuantReLU,
    torch.nn.functional.relu: qnn.QuantReLU,
    torch.nn.functional.relu_: qnn.QuantReLU,
    torch.nn.functional.relu6: qnn.QuantReLU,
    'relu': qnn.QuantReLU,
    'relu_': qnn.QuantReLU}


_trunc_map = {
    nn.AvgPool2d: qnn.QuantAvgPool2d,
    nn.AdaptiveAvgPool2d: qnn.QuantAdaptiveAvgPool2d,
    torch.nn.functional.avg_pool2d: qnn.QuantAvgPool2d,
    torch.nn.functional.adaptive_avg_pool2d: qnn.QuantAdaptiveAvgPool2d}


_wrapper_map = {
    nn.MaxPool1d: qnn.QuantMaxPool1d,
    nn.MaxPool2d: qnn.QuantMaxPool2d,
    nn.Dropout: qnn.QuantDropout}


_upsample_map = {
    nn.Upsample: qnn.QuantUpsample,
    nn.UpsamplingBilinear2d: qnn.QuantUpsamplingBilinear2d,
    nn.UpsamplingNearest2d: qnn.QuantUpsamplingNearest2d}


def _map_to_rewriter_impl(original):
    if isinstance(original, str):
        return MethodToModuleRewriter
    elif isclass(original) and issubclass(original, nn.Module):
        return ModuleToModuleRewriter
    elif callable(original):
        return FnToModuleRewriter
    else:
        raise RuntimeError(f"Original {original} type not recognized")


class BatchNormHandling(AutoName):
    PRESERVE = auto()
    QUANTIZE = auto()
    MERGE_AND_QUANTIZE = auto()
    MERGE_AND_PRESERVE = auto()


def wbiol_rewriter_list(
        input_quant, weight_quant, bias_quant, output_quant, return_quant_tensor, excluded, **kwargs):
    iq, wq, bq, oq = input_quant, weight_quant, bias_quant, output_quant
    rewriter_list = [_map_to_rewriter_impl(original)(
        original, quant, input_quant=iq, weight_quant=wq, bias_quant=bq, output_quant=oq,
        return_quant_tensor=return_quant_tensor, **kwargs)
        for original, quant in _wbiol_map.items() if original not in excluded]
    return rewriter_list


def act_rewriter_list(act_quant, return_quant_tensor, excluded, **kwargs):
    aq = act_quant
    rewriter_list = [_map_to_rewriter_impl(original)(
        original, quant, act_quant=aq, return_quant_tensor=return_quant_tensor, **kwargs)
        for original, quant in _act_map.items() if original not in excluded]
    return rewriter_list


def trunc_rewriter_list(trunc_quant, return_quant_tensor, excluded, **kwargs):
    rewriter_list = [_map_to_rewriter_impl(original)(
        original, quant, trunc_quant=trunc_quant,
        return_quant_tensor=return_quant_tensor, **kwargs)
        for original, quant in _trunc_map.items() if original not in excluded]
    return rewriter_list


def upsample_rewriter_list(return_quant_tensor, excluded):
    upsample_list = [_map_to_rewriter_impl(original)(
        original, quant, return_quant_tensor=return_quant_tensor)
        for original, quant in _upsample_map.items() if original not in excluded]
    return upsample_list


def legacy_rewriter_list(return_quant_tensor, excluded):
    wrapper_list = [_map_to_rewriter_impl(original)(
        original, quant, return_quant_tensor=return_quant_tensor)
        for original, quant in _wrapper_map.items() if original not in excluded]
    return wrapper_list


def io_rewriter_list(input_quant, output_quant, return_quant_tensor, excluded, **kwargs):
    rewriter_list = [_map_to_rewriter_impl(original)(
        original, quant, input_quant=input_quant, output_quant=output_quant,
        return_quant_tensor=return_quant_tensor, **kwargs)
        for original, quant in _io_map.items() if original not in excluded]
    return rewriter_list


def quantize(
        model: nn.Module,
        input: torch.Tensor,
        weight_quant = Int8WeightPerTensorFloat,
        input_quant = Int8ActPerTensorFloat,
        output_quant = Int8ActPerTensorFloat,
        trunc_quant = TruncTo8bit,
        bias_quant = Int8Bias,
        act_quant = None,
        bn_handling: BatchNormHandling = BatchNormHandling.MERGE_AND_QUANTIZE,
        excluded: Optional[List] = (),
        inplace=False,
        **kwargs):
    if not inplace:
        try:
            model = copy.deepcopy(model)
        except:
            raise RuntimeError("Model cannot be deepcopied, enable inplace quantization.")
    iq, oq, bq = input_quant, output_quant, bias_quant
    wq, aq, tq = weight_quant, act_quant, trunc_quant
    with torch.no_grad():
        graph_model = brevitas_value_trace(model, {'x': input})
    if (bn_handling == BatchNormHandling.MERGE_AND_QUANTIZE
            or bn_handling == BatchNormHandling.MERGE_AND_PRESERVE):
        graph_model = MergeBatchNorm2d().apply(graph_model)
    state_dict = graph_model.state_dict()
    graph_model = DuplicateSharedStatelessModule().apply(graph_model)
    graph_model = MeanMethodToAdaptiveAvgPool2d().apply(graph_model)
    for rewriter in wbiol_rewriter_list(iq, wq, bq, oq, True, excluded, **kwargs):
        graph_model = rewriter.apply(graph_model)
    for rewriter in act_rewriter_list(aq, True, excluded, **kwargs):
        graph_model = rewriter.apply(graph_model)
    for rewriter in trunc_rewriter_list(tq, True, excluded, **kwargs):
        graph_model = rewriter.apply(graph_model)
    for rewriter in io_rewriter_list(iq, oq, True, excluded, **kwargs):
        graph_model = rewriter.apply(graph_model)
    for rewriter in upsample_rewriter_list(True, excluded):
        graph_model = rewriter.apply(graph_model)
    graph_model.load_state_dict(state_dict)
    if (bn_handling == BatchNormHandling.QUANTIZE
            or bn_handling == BatchNormHandling.MERGE_AND_QUANTIZE):
        rewriter = ModuleToModuleRewriter(
            nn.BatchNorm2d,
            qnn.BatchNorm2dToQuantScaleBias,
            input_quant=input_quant,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            output_quant=output_quant,
            return_quant_tensor=True,
            **kwargs)
        graph_model = rewriter.apply(graph_model)
    if torch_version < version.parse('1.5.0'):
        for rewriter in legacy_rewriter_list(True, excluded):
            graph_model = rewriter.apply(graph_model)
    graph_model = DisableBreakingReturnQuantTensor().apply(graph_model)
    return graph_model