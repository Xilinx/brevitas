from typing import ClassVar, Optional, List
from inspect import isclass
from enum import auto

import torch
from torch import nn

from brevitas import nn as qnn
from brevitas.inject.defaults import *
from .tracer import Tracer
from .generator import ModuleGenerator
from .rewriter import TorchFnToModuleRewriter, TensorMethodToModuleRewriter
from .rewriter import ModuleToModuleRewriter
from .rewriter import MergeBatchNorm2d, DisableBreakingReturnQuantTensor
from .rewriter import DuplicateSharedStatelessModule
from .rewriter import MeanToAdaptiveAvgPool2d, InplaceMeanToAdaptiveAvgPool2d
from brevitas.utils.python_utils import AutoName


_io_map = {
    '__add__': qnn.QuantEltwiseAdd,  # todo this should be restricted to tensor types only
    '__iadd__': qnn.QuantEltwiseAdd,
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
    'relu_': qnn.QuantReLU}  # todo add more functional apis


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
        return TensorMethodToModuleRewriter
    elif isclass(original) and issubclass(original, nn.Module):
        return ModuleToModuleRewriter
    elif callable(original):
        return TorchFnToModuleRewriter
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
    if 'mean' not in excluded:
        rewriter_list.append(MeanToAdaptiveAvgPool2d())
    if 'mean_' not in excluded:
        rewriter_list.append(InplaceMeanToAdaptiveAvgPool2d())
    return rewriter_list


def no_args_rewriter_list(return_quant_tensor, excluded):
    wrapper_list = [_map_to_rewriter_impl(original)(
        original, quant, return_quant_tensor=return_quant_tensor)
        for original, quant in _wrapper_map.items() if original not in excluded]
    upsample_list = [_map_to_rewriter_impl(original)(
        original, quant, return_quant_tensor=return_quant_tensor)
        for original, quant in _upsample_map.items() if original not in excluded]
    return wrapper_list + upsample_list


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
        **kwargs):
    iq, oq, bq = input_quant, output_quant, bias_quant
    wq, aq, tq = weight_quant, act_quant, trunc_quant
    with torch.no_grad():
        trace = Tracer(input).trace_model(model)
        gen_model = ModuleGenerator().gen_model(trace)
    state_dict = model.state_dict()
    gen_model = DuplicateSharedStatelessModule().apply(gen_model)
    for rewriter in wbiol_rewriter_list(iq, wq, bq, oq, True, excluded, **kwargs):
        gen_model = rewriter.apply(gen_model)
    for rewriter in act_rewriter_list(aq, True, excluded, **kwargs):
        gen_model = rewriter.apply(gen_model)
    for rewriter in trunc_rewriter_list(tq, True, excluded, **kwargs):
        gen_model = rewriter.apply(gen_model)
    for rewriter in io_rewriter_list(iq, oq, True, excluded, **kwargs):
        gen_model = rewriter.apply(gen_model)
    for rewriter in no_args_rewriter_list(True, excluded):
        gen_model = rewriter.apply(gen_model)
    gen_model.load_state_dict(state_dict)
    if (bn_handling == BatchNormHandling.MERGE_AND_QUANTIZE
            or bn_handling == BatchNormHandling.MERGE_AND_PRESERVE):
        gen_model = MergeBatchNorm2d().apply(gen_model)
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
        gen_model = rewriter.apply(gen_model)
    gen_model = DisableBreakingReturnQuantTensor().apply(gen_model)
    return gen_model

