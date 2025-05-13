
from torch.fx import symbolic_trace
import torch.nn as nn
import torch.nn.functional as F

from brevitas.graph.per_input import AdaptiveAvgPoolToAvgPool
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.quantize import quantize
from brevitas.graph.quantize import UNSIGNED_ACT_TUPLE
from brevitas.graph.standardize import MeanMethodToAdaptiveAvgPool2d
from brevitas.graph.standardize import TorchFunctionalToModule
from brevitas.quant import Int8WeightPerChannelFloatMSE, Int8ActPerTensorFloat, Uint8ActPerTensorFloat, Uint8ActPerTensorFloatMaxInit, Int32Bias
import brevitas.nn as qnn

SHARED_WEIGHT_QUANT = Int8WeightPerChannelFloatMSE
SHARED_BIAS_QUANT = Int32Bias
SHARED_UNSIGNED_ACT_QUANT = Uint8ActPerTensorFloat
SHARED_SIGNED_ACT_QUANT = Int8ActPerTensorFloat
SHARED_RELU6_QUANT = Uint8ActPerTensorFloatMaxInit

FINN_COMPUTE_LAYER_MAP = {
    nn.Conv2d: (
        qnn.QuantConv2d,
        {
            'weight_quant': SHARED_WEIGHT_QUANT,
            'return_quant_tensor': True}),
    nn.Linear: (
        qnn.QuantLinear,
        {
            'weight_quant': SHARED_WEIGHT_QUANT,
            'bias_quant': SHARED_BIAS_QUANT,
            'return_quant_tensor': True}),}

FINN_QUANT_ACT_MAP = {
    nn.ReLU:
        (qnn.QuantReLU, {
            'act_quant': SHARED_UNSIGNED_ACT_QUANT, 'return_quant_tensor': True}),
    nn.ReLU6: (
        qnn.QuantReLU, {
            'act_quant': SHARED_RELU6_QUANT,
            'max_val': 6.,
            'return_quant_tensor': True}),}

FINN_QUANT_IDENTITY_MAP = {
    'signed':
        (qnn.QuantIdentity, {
            'act_quant': SHARED_SIGNED_ACT_QUANT, 'return_quant_tensor': True}),
    'unsigned': (
        qnn.QuantIdentity, {
            'act_quant': SHARED_UNSIGNED_ACT_QUANT, 'return_quant_tensor': True}),}


def default_quantize_maps_finn():
    return {
        "quant_identity_map": FINN_QUANT_IDENTITY_MAP,
        "compute_layer_map": FINN_COMPUTE_LAYER_MAP,
        "quant_act_map": FINN_QUANT_ACT_MAP,
        "unsigned_act_tuple": UNSIGNED_ACT_TUPLE,
    }


def preprocess_for_finn_quantize(
        model,
        *model_args,
        trace_model=True,
        relu6_to_relu=False,
        equalize_iters=0,
        equalize_merge_bias=False,
        merge_bn=False,
        equalize_bias_shrinkage='vaiq',
        equalize_scale_computation='maxabs',
        **model_kwargs):
    training_state = model.training
    model.eval()

    if trace_model:
        model = symbolic_trace(model)
    model = MeanMethodToAdaptiveAvgPool2d().apply(model)
    model = TorchFunctionalToModule(fn_to_module_map=((F.adaptive_avg_pool2d, nn.AdaptiveAvgPool2d),)).apply(model)
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


def quantize_finn(
        graph_model,
        quant_identity_map=FINN_QUANT_IDENTITY_MAP,
        compute_layer_map=FINN_COMPUTE_LAYER_MAP,
        quant_act_map=FINN_QUANT_ACT_MAP,
        unsigned_act_tuple=UNSIGNED_ACT_TUPLE,
        requantize_layer_handler_output=True):
    return quantize(
        graph_model,
        quant_identity_map=quant_identity_map,
        compute_layer_map=compute_layer_map,
        quant_act_map=quant_act_map,
        unsigned_act_tuple=unsigned_act_tuple,
        requantize_layer_handler_output=requantize_layer_handler_output)
