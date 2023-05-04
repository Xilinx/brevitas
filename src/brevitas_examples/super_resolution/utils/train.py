# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_psnr(gen_image: Tensor, ref_image: Tensor, eps: float = 1e-10):
    dist = (gen_image - ref_image).pow(2).mean()
    psnr = 10. * torch.log10(1. / dist.clamp_min(eps))
    return psnr


def train_for_epoch(trainloader, model, criterion, optimizer, args):
    tot_loss = 0.
    for i, (images, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss: Tensor = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * images.size(0)
    avg_loss = tot_loss / len(trainloader.dataset)
    return avg_loss


@torch.no_grad()
def validate(testloader, model, args):
    model.eval()
    tot_psnr = 0.
    for _, (images, target) in enumerate(testloader):
        images = images.to(device)
        target = target.to(device)
        output = model(images)
        tot_psnr += calculate_psnr(output, target).item() * images.size(0)
    avg_psnr = tot_psnr / len(testloader.dataset)
    return avg_psnr
