# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from brevitas.core.scaling import AccumulatorAwareParameterPreScaling
from brevitas.function.shape import over_output_channels
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL

__all__ = ["apply_bias_correction", "apply_ep_init"]


def get_a2q_module(module: nn.Module):
    for submod in module.modules():
        if isinstance(submod, AccumulatorAwareParameterPreScaling):
            return submod
    return None


def _euclidean_projection_onto_positive_simplex(vec: Tensor, radius: float = 1.):
    assert radius > 0, "Error: radius needs to be strictly positive."
    assert vec.ndim == 1, "Error: projection assumes a vector, not a matrix."
    assert vec.min() >= 0, "Error: assuming a vector of non-negative numbers."
    n_elems = vec.shape[0]
    # if we are already within the simplex, then the best projection is itself
    if vec.sum() <= radius:
        return vec
    # using algorithm derived in `Efficient Projections onto the L1-Ball for
    # Learning in High Dimensions`
    v = vec.cpu().detach().numpy()
    u = np.sort(v)[::-1]
    cumsum_u = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n_elems + 1) > (cumsum_u - radius))[0][-1]
    theta = float(cumsum_u[rho] - radius) / (rho + 1)
    w = np.clip(v - theta, 0, np.inf)
    vec.data = torch.tensor(w, dtype=vec.dtype, device=vec.device)
    return vec


def euclidean_projection_onto_l1_ball(vec: Tensor, radius: float):
    assert radius > 0, "Error: radius needs to be strictly positive."
    assert vec.ndim == 1, "Error: projection assumes a vector, not a matrix."
    vec_dir = vec.sign()
    vec_mag = _euclidean_projection_onto_positive_simplex(vec.abs(), radius)
    new_vec = vec_dir * vec_mag
    assert vec.shape == new_vec.shape, "Error: shape changed."
    return new_vec


def l1_proj_matrix_per_channel(weights: Tensor, radius: Tensor):
    assert isinstance(weights, Tensor), "Error: weights is assumed to be a Tensor."
    assert isinstance(radius, Tensor), "Error: radius is assumed to be a Tensor."
    assert weights.ndim == 2, "Error: assuming a matrix with ndim=2."
    # if defined per-tensor
    if radius.ndim == 0:
        radius = torch.ones(weights.shape[0]) * radius
    # if defined per-channel
    else:
        radius = radius.flatten()
        assert radius.nelement() == weights.shape[0], "Error: shape mismatch."
    # project each channel independently
    for i in range(weights.shape[0]):
        w = weights[i]
        z = radius[i].item()
        v = euclidean_projection_onto_l1_ball(w, z)
        weights[i] = v
    return weights


def apply_ep_init(model: nn.Module, inp: Tensor):
    """Euclidean projection-based weight initialization (EP-init) for accumulator-aware
    quantization as proposed in `A2Q+: Improving Accumulator-Aware Weight Quantization`"""
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    module_stats = {}
    hook_list = list()

    def register_upper_bound(module: AccumulatorAwareParameterPreScaling, inp, output, name):
        """Accumulate the regularization penalty across constrained layers"""
        nonlocal module_stats

        (weights, input_bit_width, input_is_signed) = inp
        scales: Tensor = module.scaling_impl(weights)
        max_norm: Tensor = module.calc_max_l1_norm(input_bit_width, input_is_signed)

        shape = over_output_channels(weights)
        s = scales.reshape(shape)
        w = weights.reshape(shape)

        z: Tensor = s * max_norm  # radius
        module_stats[name] = (w.detach(), z.detach())  # no gradients

        restrict_value_impl = module.restrict_clamp_scaling.restrict_value_impl
        pre_scaling_init: Tensor = restrict_value_impl.restrict_init_tensor(scales * max_norm)
        assert pre_scaling_init.shape == module.value.shape, "Error: shape mismatch."
        module.value.data = torch.where(
            module.value.data <= pre_scaling_init, module.value.data, pre_scaling_init)

        return output

    # add hooks to each of the A2Q pre-scaling modules
    for name, mod in model.named_modules():
        if isinstance(mod, QuantWBIOL):
            submod = get_a2q_module(mod)
            if submod is not None:
                hook_fn = partial(register_upper_bound, name=name)
                hook = submod.register_forward_hook(hook_fn)
                hook_list.append(hook)

    inp = inp.to(device=device, dtype=dtype)
    model(inp)  # register the scaled upper bounds

    # project weights onto the l1-ball
    for name, mod in model.named_modules():
        if name in module_stats and isinstance(mod, (nn.Conv2d, nn.Linear)):
            (weights, radius) = module_stats[name]
            weights = l1_proj_matrix_per_channel(weights, radius)
            weights = weights.reshape(mod.weight.shape)
            mod.weight.data = weights

    for hook in hook_list:
        hook.remove()

    return model
