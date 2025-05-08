
from tqdm import tqdm

import torch
from torch.fx import symbolic_trace
import torch.nn as nn

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.calibrate import norm_correction_mode
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.per_input import AdaptiveAvgPoolToAvgPool
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.graph.quantize import quantize
from brevitas.graph.quantize import UNSIGNED_ACT_TUPLE
import brevitas.nn as qnn
import brevitas.onnx as bo
from brevitas.quant import Int8WeightPerChannelFloat, Int8ActPerTensorFloat, Uint8ActPerTensorFloat, Uint8ActPerTensorFloatMaxInit, Int32Bias
from brevitas.quant_tensor import QuantTensor

SEED = 123456

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SHARED_WEIGHT_QUANT = Int8WeightPerChannelFloat
SHARED_BIAS_QUANT = Int32Bias
SHARED_UNSIGNED_ACT_QUANT = Uint8ActPerTensorFloat
SHARED_SIGNED_ACT_QUANT = Int8ActPerTensorFloat
SHARED_RELU6_QUANT = Uint8ActPerTensorFloatMaxInit

FINN_COMPUTE_LAYER_MAP = {
    nn.AvgPool2d: (qnn.TruncAvgPool2d, {
        'return_quant_tensor': True}),
    nn.Conv1d: (
        qnn.QuantConv1d,
        {
            'weight_quant': SHARED_WEIGHT_QUANT,
            'bias_quant': SHARED_BIAS_QUANT,
            'return_quant_tensor': True}),
    nn.Conv2d: (
        qnn.QuantConv2d,
        {
            'weight_quant': SHARED_WEIGHT_QUANT,
            'bias_quant': SHARED_BIAS_QUANT,
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


def get_dataloader(src_dir, split, num_workers=8, batch_size=100, subset_size=150):
    dataset = datasets.ImageFolder(
        f"{src_dir}/{split}",
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]))
    if subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return loader


@torch.no_grad
def test(model, loader):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    total_correct = 0
    total_examples = 0
    for i, (images, target) in enumerate(tqdm(loader)):
        target = target.to(device=device, dtype=dtype)
        images = images.to(device=device, dtype=dtype)
        batch_size = images.shape[0]

        output = model(images)
        if isinstance(output, QuantTensor):
            output = output.value

        # measure accuracy
        pred = torch.argmax(output, axis=1)
        correct = (target == pred).sum()
        total_correct += int(correct)
        total_examples += int(batch_size)

    accuracy = 100 * (total_correct / total_examples)
    print(f"Accuracy: {accuracy:.2f}%, Total Correct: {total_correct}, Total Examples: {total_examples}")
    return accuracy, total_correct, total_examples


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


def calibrate(calib_loader, model):
    """
    Perform calibration and bias correction, if enabled
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with calibration_mode(model):
            for i, (images, target) in enumerate(tqdm(calib_loader)):
                images = images.to(device)
                images = images.to(dtype)
                model(images)


def calibrate_bn(calib_loader, model):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with norm_correction_mode(model):
            for i, (images, target) in enumerate(tqdm(calib_loader)):
                images = images.to(device)
                images = images.to(dtype)
                model(images)


def apply_bias_correction(calib_loader, model):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with bias_correction_mode(model):
            for i, (images, target) in enumerate(tqdm(calib_loader)):
                images = images.to(device)
                images = images.to(dtype)
                model(images)


def apply_act_equalization(model, calib_loader, layerwise):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    add_mul_node = layerwise
    with torch.no_grad():
        with activation_equalization_mode(model,
                                          alpha=0.5,
                                          layerwise=layerwise,
                                          add_mul_node=add_mul_node):
            for i, (images, target) in enumerate(tqdm(calib_loader)):
                images = images.to(device)
                images = images.to(dtype)
                model(images)


# Configure datasets
imagenet_datadir = "imagenet_symlink"
calib_loader = get_dataloader(f"{imagenet_datadir}", "calibration")
valid_loader = get_dataloader(f"{imagenet_datadir}", "val")

# Load model
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.to(device="cuda:0")
model.eval()

# Test float model
print("Float Validation Accuracy")
results = test(model, valid_loader)

# Modify network to assist the quantization process
x = calib_loader.dataset[0][0].unsqueeze(0).to(device="cuda:0") # Resolve the shape of the AveragePool
model = preprocess_for_finn_quantize(model, x=x)

# Test preprocessed model
print("Preprocessed Validation Accuracy")
results = test(model, valid_loader)

# Pre-quantization transformations
act_eq = True
if act_eq:
    print("Starting Activation Equalization:")
    apply_bias_correction(calib_loader, model)
    print("Equalized Model Validation Accuracy")
    results = test(model, valid_loader)

# Quantize Model
model = quantize_finn(model)
model.to(device="cuda:0")

# Post-quantization transformations
cal_bn = False
bias_corr = True
print("Starting activation calibration:")
calibrate(calib_loader, model)
print("Quantized Model Validation Accuracy")
results = test(model, valid_loader)

if cal_bn:
    print("Starting BatchNorm Calibration:")
    calibrate_bn(calib_loader, model)
    print("Quantized Model Validation Accuracy")
    results = test(model, valid_loader)

if bias_corr:
    print("Starting Apply Bias Correction:")
    apply_bias_correction(calib_loader, model)
    print("Quantized Model Validation Accuracy")
    results = test(model, valid_loader)

# Test Quantized model
print("Quantized Model Validation Accuracy")
results = test(model, valid_loader)

# Export model to QONNX
with torch.no_grad():
    bo.export_qonnx(
        model,
        (x),
        "quant_mobilenet_v2.onnx",
        do_constant_folding=True,
        input_names=['x'],
        opset_version=17,
    )
