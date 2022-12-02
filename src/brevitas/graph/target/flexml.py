# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch import nn

from brevitas.fx.brevitas_tracer import symbolic_trace
from brevitas.graph.base import ModuleToModuleByClass
from brevitas.graph.equalize import EqualizeGraph
from brevitas.graph.fixed_point import CollapseConsecutiveConcats
from brevitas.graph.fixed_point import MergeBatchNorm
from brevitas.graph.fixed_point import MoveSplitBatchNormBeforeCat
from brevitas.graph.per_input import AdaptiveAvgPoolToAvgPool
from brevitas.graph.quantize import quantize
from brevitas.graph.standardize import DuplicateSharedStatelessModule
from brevitas.graph.standardize import MeanMethodToAdaptiveAvgPool2d
from brevitas.graph.standardize import RemoveStochasticModules
from brevitas.graph.standardize import TorchFunctionalToModule
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFixedPoint
from brevitas.quant import Int8WeightPerTensorFixedPoint
from brevitas.quant import Int16Bias
from brevitas.quant import Uint8ActPerTensorFixedPoint
from brevitas.quant import Uint8ActPerTensorFixedPointMaxInit
from brevitas.quant.fixed_point import Int8ActPerTensorFixedPointMinMaxInit

FLEXML_QUANT_AVGPOOL_MAP = {nn.AvgPool2d: qnn.flexml.FlexMLQuantAvgPool2d}

FLEXML_QUANT_WBIOL_MAP = {
    nn.Conv1d: (
        qnn.QuantConv1d,
        {
            'weight_quant': Int8WeightPerTensorFixedPoint,
            'bias_quant': Int16Bias,
            'return_quant_tensor': True}),
    nn.Conv2d: (
        qnn.QuantConv2d,
        {
            'weight_quant': Int8WeightPerTensorFixedPoint,
            'bias_quant': Int16Bias,
            'return_quant_tensor': True}),
    nn.ConvTranspose1d: (
        qnn.QuantConvTranspose1d,
        {
            'weight_quant': Int8WeightPerTensorFixedPoint,
            'bias_quant': Int16Bias,
            'return_quant_tensor': True}),
    nn.ConvTranspose2d: (
        qnn.QuantConvTranspose2d,
        {
            'weight_quant': Int8WeightPerTensorFixedPoint,
            'bias_quant': Int16Bias,
            'return_quant_tensor': True}),
    nn.BatchNorm1d: (
        qnn.BatchNorm1dToQuantScaleBias,
        {
            'weight_quant': Int8WeightPerTensorFixedPoint,
            'bias_quant': Int16Bias,
            'return_quant_tensor': True}),
    nn.BatchNorm2d: (
        qnn.BatchNorm2dToQuantScaleBias, {
            'weight_quant': Int8WeightPerTensorFixedPoint, 'return_quant_tensor': True}),
    nn.Linear: (
        qnn.QuantLinear, {
            'weight_quant': Int8WeightPerTensorFixedPoint, 'return_quant_tensor': True})}

FLEXML_UNSIGNED_ACT_TUPLE = (nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Hardsigmoid)

FLEXML_QUANT_ACT_MAP = {
    nn.ReLU:
        (qnn.QuantReLU, {
            'act_quant': Uint8ActPerTensorFixedPoint, 'return_quant_tensor': True}),
    nn.ReLU6: (
        qnn.QuantReLU, {
            'act_quant': Uint8ActPerTensorFixedPointMaxInit,
            'max_val': 6.,
            'return_quant_tensor': True}),
    nn.LeakyReLU: (
        qnn.flexml.FlexMLQuantLeakyReLU,
        {
            'alpha_quant':
                qnn.QuantIdentity(Uint8ActPerTensorFixedPoint, bit_width=16),
            'input_quant':
                qnn.QuantIdentity(
                    Int8ActPerTensorFixedPoint, bit_width=16, scaling_stats_momentum=None),
            'output_quant':
                qnn.QuantIdentity(Int8ActPerTensorFixedPoint, return_quant_tensor=True)}),
    nn.Hardtanh: (
        qnn.QuantHardTanh,
        {
            'act_quant': Int8ActPerTensorFixedPointMinMaxInit,
            'max_val': lambda module: module.max_val,
            'min_val': lambda module: module.min_val}),
    nn.Sigmoid: (
        qnn.QuantSigmoid, {
            'act_quant': Uint8ActPerTensorFixedPoint,
            'return_quant_tensor': True,}),
    nn.SiLU: (
        qnn.flexml.FlexMLQuantSwish, {
            'act_quant': Int8ActPerTensorFixedPoint,
            'return_quant_tensor': True,}),
    nn.Hardswish: (
        qnn.flexml.FlexMLQuantHardswish, {
            'act_quant': Int8ActPerTensorFixedPoint,
            'return_quant_tensor': True,}),
    nn.Hardsigmoid: (
        qnn.flexml.FlexMLQuantHardsigmoid, {
            'act_quant': Uint8ActPerTensorFixedPoint,
            'return_quant_tensor': True,})}

FLEXML_QUANT_IDENTITY_MAP = {
    'signed':
        (qnn.QuantIdentity, {
            'act_quant': Int8ActPerTensorFixedPoint, 'return_quant_tensor': True}),
    'unsigned': (
        qnn.QuantIdentity, {
            'act_quant': Uint8ActPerTensorFixedPoint, 'return_quant_tensor': True}),}


def preprocess_flexml(
        model,
        *model_args,
        trace_model=True,
        equalization_iters=0,
        merge_bias=True,
        merge_bn=True,
        bias_shrinkage='vaiq',
        scale_computation='maxabs',
        **model_kwargs):
    training_state = model.training
    model.eval()
    if trace_model:
        model = symbolic_trace(model)  # TODO model args should contribute to value tracing
    model = TorchFunctionalToModule().apply(model)
    model = DuplicateSharedStatelessModule().apply(model)
    model = ModuleToModuleByClass(nn.ReLU6, nn.ReLU).apply(model)
    model = MeanMethodToAdaptiveAvgPool2d().apply(model)
    model = AdaptiveAvgPoolToAvgPool().apply(model, *model_args, **model_kwargs)
    model = CollapseConsecutiveConcats().apply(model)
    model = MoveSplitBatchNormBeforeCat().apply(model)
    if merge_bn:
        model = MergeBatchNorm().apply(model)
    model = RemoveStochasticModules().apply(model)
    model = EqualizeGraph(
        equalization_iters,
        merge_bias=merge_bias,
        bias_shrinkage=bias_shrinkage,
        scale_computation=scale_computation).apply(model)
    model.train(training_state)
    return model


def quantize_flexml(
        graph_model,
        *model_args,
        avgpool_to_depthwise_conv=False,
        quant_identity_map=FLEXML_QUANT_IDENTITY_MAP,
        quant_wbiol_map=FLEXML_QUANT_WBIOL_MAP,
        quant_act_map=FLEXML_QUANT_ACT_MAP,
        quant_avg_pool_map=FLEXML_QUANT_AVGPOOL_MAP,
        unsigned_act_tuple=FLEXML_UNSIGNED_ACT_TUPLE,
        **model_kwargs):
    return quantize(
        graph_model,
        model_args,
        avgpool_to_depthwise_conv=avgpool_to_depthwise_conv,
        quant_identity_map=quant_identity_map,
        quant_wbiol_map=quant_wbiol_map,
        quant_act_map=quant_act_map,
        quant_avg_pool_map=quant_avg_pool_map,
        unsigned_act_tuple=unsigned_act_tuple,
        model_kwargs=model_kwargs)
