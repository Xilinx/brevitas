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
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.quantize import quantize
from brevitas.graph.quantize import UNSIGNED_ACT_TUPLE
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

FLEXML_COMPUTE_LAYER_MAP = {
    nn.AvgPool2d: (qnn.flexml.FlexMLQuantAvgPool2d, {
        'return_quant_tensor': True}),
    nn.MultiheadAttention: (
        qnn.QuantMultiheadAttention,
        {
            'in_proj_weight_quant': Int8WeightPerTensorFixedPoint,
            'in_proj_bias_quant': Int16Bias,
            'attn_output_weights_quant': Uint8ActPerTensorFixedPoint,
            'q_scaled_quant': Int8ActPerTensorFixedPoint,
            'k_transposed_quant': Int8ActPerTensorFixedPoint,
            'v_quant': Int8ActPerTensorFixedPoint,
            'out_proj_input_quant': Int8ActPerTensorFixedPoint,
            'out_proj_weight_quant': Int8WeightPerTensorFixedPoint,
            'out_proj_bias_quant': Int16Bias,
            'return_quant_tensor': True}),
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
        qnn.BatchNorm2dToQuantScaleBias,
        {
            'weight_quant': Int8WeightPerTensorFixedPoint,
            'bias_quant': Int16Bias,
            'return_quant_tensor': True}),
    nn.Linear: (
        qnn.QuantLinear,
        {
            'weight_quant': Int8WeightPerTensorFixedPoint,
            'bias_quant': Int16Bias,
            'return_quant_tensor': True})}

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
            'min_val': lambda module: module.min_val,
            'return_quant_tensor': True}),
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


def preprocess_for_flexml_quantize(
        model,
        *model_args,
        trace_model=True,
        relu6_to_relu=True,
        equalize_iters=0,
        equalize_merge_bias=True,
        merge_bn=True,
        equalize_bias_shrinkage: str = 'vaiq',
        equalize_scale_computation: str = 'maxabs',
        **model_kwargs):
    training_state = model.training
    model.eval()

    if trace_model:
        model = symbolic_trace(model)
    model = MeanMethodToAdaptiveAvgPool2d().apply(model)
    model = AdaptiveAvgPoolToAvgPool().apply(model, *model_args, **model_kwargs)
    model = preprocess_for_quantize(
        model,
        False,
        relu6_to_relu,
        equalize_iters,
        equalize_merge_bias,
        merge_bn,
        equalize_bias_shrinkage,
        equalize_scale_computation)
    model.train(training_state)
    return model


def quantize_flexml(
        graph_model,
        quant_identity_map=FLEXML_QUANT_IDENTITY_MAP,
        compute_layer_map=FLEXML_COMPUTE_LAYER_MAP,
        quant_act_map=FLEXML_QUANT_ACT_MAP,
        unsigned_act_tuple=UNSIGNED_ACT_TUPLE,
        requantize_layer_handler_output=True):
    return quantize(
        graph_model,
        quant_identity_map=quant_identity_map,
        compute_layer_map=compute_layer_map,
        quant_act_map=quant_act_map,
        unsigned_act_tuple=unsigned_act_tuple,
        requantize_layer_handler_output=requantize_layer_handler_output)
