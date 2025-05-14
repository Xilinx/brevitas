
from tqdm import tqdm

import torch

from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gptq import gptq_mode

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


def apply_act_equalization(calib_loader, model, alpha=0.5, add_mul_node=False):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with activation_equalization_mode(model,
                                          alpha=alpha,
                                          layerwise=False,
                                          add_mul_node=add_mul_node):
            for i, (images, target) in enumerate(tqdm(calib_loader)):
                images = images.to(device)
                images = images.to(dtype)
                model(images)


def apply_gptq(
        calib_loader,
        model,
        act_order=True,
        use_quant_activations=True,
        create_weight_orig=True):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        model(next(iter(calib_loader))[0].to(device=device,dtype=dtype)) # Harden weight scale factors if not done already
        with gptq_mode(model,
                       act_order=act_order,
                       use_quant_activations=use_quant_activations,
                       create_weight_orig=create_weight_orig) as gptq:
            gptq_model = gptq.model
            for i in tqdm(range(gptq.num_layers)):
                for i, (images, target) in enumerate(calib_loader):
                    images = images.to(device)
                    images = images.to(dtype)
                    gptq_model(images)
                gptq.update()


def apply_gpfq(
        calib_loader,
        model,
        act_order=True,
        create_weight_orig=True):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        model(next(iter(calib_loader))[0].to(device=device,dtype=dtype)) # Harden weight scale factors if not done already
        with gpfq_mode(model,
                       create_weight_orig=create_weight_orig,
                       use_quant_activations=True,
                       act_order=act_order) as gpfq:
            gpfq_model = gpfq.model
            for i in tqdm(range(gpfq.num_layers)):
                for i, (images, target) in enumerate(calib_loader):
                    images = images.to(device)
                    images = images.to(dtype)
                    gpfq_model(images)
                gpfq.update()
