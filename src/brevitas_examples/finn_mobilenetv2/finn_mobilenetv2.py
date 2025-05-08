
import torch
import torch.nn as nn

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

import brevitas.onnx as bo

from graph.target.finn import default_quantize_maps_finn
from graph.target.finn import preprocess_for_finn_quantize
from graph.target.finn import quantize_finn

from utils import get_dataloader
from utils import test
from quant_utils import calibrate
from quant_utils import calibrate_bn
from quant_utils import apply_bias_correction
from quant_utils import apply_act_equalization

# Global settings
SEED = 123456
batch_size=200
subset_size=400 # Use 'None' if you want to use the entire dataset
device="cuda:0"

verbose=False # Validate model after every step

act_eq = True # Apply act equalization
cal_bn = False # Recalibrate batchnorm
bias_corr = True # Apply bias correction

# Configure datasets
imagenet_datadir = "imagenet_symlink"
calib_loader = get_dataloader(f"{imagenet_datadir}", "calibration", batch_size=batch_size, subset_size=subset_size)
valid_loader = get_dataloader(f"{imagenet_datadir}", "val", batch_size=batch_size, subset_size=subset_size)

# Load model
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.to(device=device)
model.eval()
# Test float model
print("Float Validation Accuracy")
results = test(model, valid_loader)

# Modify network to assist the quantization process
x = next(iter(calib_loader))[0].to(device=device) # Resolve the shape of the AveragePool
model = preprocess_for_finn_quantize(model, x=x)
# Test preprocessed model
if verbose:
    print("Preprocessed Validation Accuracy")
    results = test(model, valid_loader)

# Pre-quantization transformations
if act_eq:
    print("Applying Activation Equalization:")
    apply_bias_correction(calib_loader, model)
    if verbose:
        print("Equalized Model Validation Accuracy")
        results = test(model, valid_loader)

# Quantize Model
finn_quant_maps = default_quantize_maps_finn()
model = quantize_finn(model)
model.to(device=device) # TODO: fix this

# Post-quantization transformations
print("Applying activation calibration:")
calibrate(calib_loader, model)
if verbose:
    print("Quantized Model Validation Accuracy")
    results = test(model, valid_loader)

if cal_bn:
    print("Applying BatchNorm Calibration:")
    calibrate_bn(calib_loader, model)
    if verbose:
        print("Quantized Model Validation Accuracy")
        results = test(model, valid_loader)

if bias_corr:
    print("Applying Apply Bias Correction:")
    apply_bias_correction(calib_loader, model)
    if verbose:
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
