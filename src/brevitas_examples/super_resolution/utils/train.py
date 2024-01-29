# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor

from brevitas.core.scaling.pre_scaling import AccumulatorAwareParameterPreScaling
from brevitas.function import abs_binary_sign_grad

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def calc_average_psnr(ref_images: Tensor, gen_images: Tensor, eps: float = 1e-10) -> Tensor:
    assert ref_images.shape == gen_images.shape, "Input tensor shapes need to match."
    dist = (ref_images - gen_images).square().mean(axis=(1, 2, 3))  # assuming NCHW
    psnr = 10. * torch.log10(1. / dist.clamp_min(eps))
    return psnr.mean()


def train_for_epoch(trainloader, model, criterion, optimizer, reg_weight: float = 1e-3):
    model.train()

    tot_loss, reg_penalty = 0., 0.

    def acc_reg_penalty(module: AccumulatorAwareParameterPreScaling, inp, output):
        """Accumulate the regularization penalty across constrained layers"""
        nonlocal reg_penalty
        (weights, input_bit_width, input_is_signed) = inp
        s = module.scaling_impl(weights)  # s
        g = abs_binary_sign_grad(module.restrict_clamp_scaling(module.value))  # g
        T = module.calc_max_l1_norm(input_bit_width, input_is_signed)  # T / s
        cur_penalty = torch.relu(g - (T * s)).sum()
        reg_penalty += cur_penalty
        return output

    # Register a forward hook to accumulate the regularization penalty
    hook_fns = list()
    for mod in model.modules():
        if isinstance(mod, AccumulatorAwareParameterPreScaling):
            hook = mod.register_forward_hook(acc_reg_penalty)
            hook_fns.append(hook)

    for _, (images, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        task_loss: Tensor = criterion(outputs, targets)
        loss = task_loss + (reg_weight * reg_penalty)
        loss.backward()
        optimizer.step()
        reg_penalty = 0.  # reset the accumulated regularization penalty
        tot_loss += task_loss.item() * images.size(0)

    # Remove the registered forward hooks before exiting
    for hook in hook_fns:
        hook.remove()

    avg_loss = tot_loss / len(trainloader.dataset)
    return avg_loss


@torch.no_grad()
def evaluate_avg_psnr(testloader, model):
    model.eval()
    tot_psnr = 0.
    for _, (images, target) in enumerate(testloader):
        images = images.to(device)
        target = target.to(device)
        output = model(images)
        tot_psnr += calc_average_psnr(output, target).item() * images.size(0)
    avg_psnr = tot_psnr / len(testloader.dataset)
    return avg_psnr
