
import torch
import torch.nn as nn

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

import brevitas.onnx as bo

from graph.target.finn import default_quantize_maps_finn
from graph.target.finn import preprocess_for_finn_quantize
from graph.target.finn import quantize_finn

from utils import get_dataloader
from utils import test
from quant_utils import apply_act_equalization
from quant_utils import apply_bias_correction
from quant_utils import apply_gpfq
from quant_utils import apply_gptq
from quant_utils import calibrate

# Global settings
batch_size=200
subset_size=None # Use 'None' if you want to use the entire dataset
device="cuda:0"
verbose=False # Validate model after every step

act_eq = False # Apply act equalization - not very useful for MNv2
act_eq_alpha = 0.5 # [0.0 -> 1.0] Intuition: higher makes weights easier to quantize, lower makes the activations easier to quantize
bias_corr = False # Apply bias correction
gptq = False # Apply GPTQ
gpfq = False # Apply GPFQ

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
    print(f"Applying Activation Equalization (alpha={act_eq_alpha}):")
    apply_act_equalization(calib_loader, model, alpha=act_eq_alpha)
    if verbose:
        print("Equalized Model Validation Accuracy")
        results = test(model, valid_loader)

# Quantize Model
finn_quant_maps = default_quantize_maps_finn()
#finn_quant_maps["compute_layer_map"][nn.Conv2d][1]['weight_bit_width'] = 4 # 1. Override Conv2d weights to have 4-bits
#finn_quant_maps["compute_layer_map"][nn.Conv2d][1]['weight_bit_width'] = lambda module: 8 if module.groups != 1 else 4 # 2. Groupwise Conv2ds @ 8-bits, the rest @ 4-bits
#finn_quant_maps["compute_layer_map"][nn.Conv2d][1]['weight_bit_width'] = lambda module, name: 8 if module.groups != 1 or name == "features.0.0" else 4 # 3. Keep first conv in 8-bits otherwise same as above
model = quantize_finn(model, **finn_quant_maps)
model.to(device=device) # TODO: fix this

# Post-quantization transformations
print("Applying activation calibration:")
calibrate(calib_loader, model)
if verbose:
    print("Quantized Model Validation Accuracy")
    results = test(model, valid_loader)

if gptq:
    print("Applying GPTQ:")
    apply_gptq(calib_loader, model)
    if verbose:
        print("Quantized Model Validation Accuracy")
        results = test(model, valid_loader)

if gpfq:
    print("Applying GPFQ:")
    apply_gpfq(calib_loader, model)
    if verbose:
        print("Quantized Model Validation Accuracy")
        results = test(model, valid_loader)

if bias_corr:
    print("Applying Bias Correction:")
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
