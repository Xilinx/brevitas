# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
from typing import Tuple, Type

import numpy as np
import torch
from torch import hub
from torch import Tensor
from torch.nn import Module
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from brevitas.core.scaling.pre_scaling import AccumulatorAwareParameterPreScaling
from brevitas.function import abs_binary_sign_grad
from brevitas.graph.calibrate import bias_correction_mode
from brevitas.graph.calibrate import calibration_mode

from .ep_init import apply_ep_init
from .quant import *
from .resnet import float_resnet18
from .resnet import quant_resnet18

__all__ = [
    "apply_ep_init",
    "apply_act_calibrate",
    "apply_bias_correction",
    "get_model_by_name",
    "filter_params",
    "create_calibration_dataloader",
    "get_cifar10_dataloaders",
    "train_for_epoch",
    "evaluate_topk_accuracies"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_impl = {
    "float_resnet18":
        float_resnet18,
    "quant_resnet18_w4a4_a2q_16b":
        partial(
            quant_resnet18,
            act_bit_width=4,
            acc_bit_width=16,
            weight_bit_width=4,
            weight_quant=CommonIntAccumulatorAwareWeightQuant),
    "quant_resnet18_w4a4_a2q_15b":
        partial(
            quant_resnet18,
            act_bit_width=4,
            acc_bit_width=15,
            weight_bit_width=4,
            weight_quant=CommonIntAccumulatorAwareWeightQuant),
    "quant_resnet18_w4a4_a2q_14b":
        partial(
            quant_resnet18,
            act_bit_width=4,
            acc_bit_width=14,
            weight_bit_width=4,
            weight_quant=CommonIntAccumulatorAwareWeightQuant),
    "quant_resnet18_w4a4_a2q_13b":
        partial(
            quant_resnet18,
            act_bit_width=4,
            acc_bit_width=13,
            weight_bit_width=4,
            weight_quant=CommonIntAccumulatorAwareWeightQuant),
    "quant_resnet18_w4a4_a2q_12b":
        partial(
            quant_resnet18,
            act_bit_width=4,
            acc_bit_width=12,
            weight_bit_width=4,
            weight_quant=CommonIntAccumulatorAwareWeightQuant),
    "quant_resnet18_w4a4_a2q_plus_16b":
        partial(
            quant_resnet18,
            act_bit_width=4,
            acc_bit_width=16,
            weight_bit_width=4,
            weight_quant=CommonIntAccumulatorAwareZeroCenterWeightQuant),
    "quant_resnet18_w4a4_a2q_plus_15b":
        partial(
            quant_resnet18,
            act_bit_width=4,
            acc_bit_width=15,
            weight_bit_width=4,
            weight_quant=CommonIntAccumulatorAwareZeroCenterWeightQuant),
    "quant_resnet18_w4a4_a2q_plus_14b":
        partial(
            quant_resnet18,
            act_bit_width=4,
            acc_bit_width=14,
            weight_bit_width=4,
            weight_quant=CommonIntAccumulatorAwareZeroCenterWeightQuant),
    "quant_resnet18_w4a4_a2q_plus_13b":
        partial(
            quant_resnet18,
            act_bit_width=4,
            acc_bit_width=13,
            weight_bit_width=4,
            weight_quant=CommonIntAccumulatorAwareZeroCenterWeightQuant),
    "quant_resnet18_w4a4_a2q_plus_12b":
        partial(
            quant_resnet18,
            act_bit_width=4,
            acc_bit_width=12,
            weight_bit_width=4,
            weight_quant=CommonIntAccumulatorAwareZeroCenterWeightQuant)}

root_url = 'https://github.com/Xilinx/brevitas/releases/download/'

model_url = {
    "float_resnet18":
        f"{root_url}/a2q_cifar10_r1/float_resnet18-1d98d23a.pth",
    "quant_resnet18_w4a4_a2q_12b":
        f"{root_url}/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_12b-8a440436.pth",
    "quant_resnet18_w4a4_a2q_13b":
        f"{root_url}/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_13b-8c31a2b1.pth",
    "quant_resnet18_w4a4_a2q_14b":
        f"{root_url}/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_14b-267f237b.pth",
    "quant_resnet18_w4a4_a2q_15b":
        f"{root_url}/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_15b-0d5bf266.pth",
    "quant_resnet18_w4a4_a2q_16b":
        f"{root_url}/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_16b-d0af41f1.pth",
    "quant_resnet18_w4a4_a2q_plus_12b":
        f"{root_url}/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_plus_12b-d69f003b.pth",
    "quant_resnet18_w4a4_a2q_plus_13b":
        f"{root_url}/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_plus_13b-332aaf81.pth",
    "quant_resnet18_w4a4_a2q_plus_14b":
        f"{root_url}/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_plus_14b-5a2d11aa.pth",
    "quant_resnet18_w4a4_a2q_plus_15b":
        f"{root_url}/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_plus_15b-3c89551a.pth",
    "quant_resnet18_w4a4_a2q_plus_16b":
        f"{root_url}/a2q_cifar10_r1/quant_resnet18_w4a4_a2q_plus_16b-19973380.pth"}


def get_model_by_name(
        model_name: str,
        pretrained: bool = False,
        init_from_float_checkpoint: bool = False) -> nn.Module:

    assert model_name in model_impl, f"Error: {model_name} not implemented."
    assert not (pretrained and init_from_float_checkpoint), "Error: pretrained and init_from_float_checkpoint cannot both be true."
    model: Module = model_impl[model_name]()

    if init_from_float_checkpoint:
        checkpoint = model_url["float_resnet18"]
        state_dict = hub.load_state_dict_from_url(checkpoint, progress=True, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)

    elif pretrained:
        checkpoint = model_url[model_name]
        state_dict = hub.load_state_dict_from_url(checkpoint, progress=True, map_location='cpu')
        if model_name.startswith("quant"):
            # fixes issue when bias keys are missing in the pre-trained state_dict when loading from checkpoint
            _prepare_bias_corrected_quant_model(model)
        model.load_state_dict(state_dict, strict=True)

    return model


def filter_params(named_params, decay):
    decay_params, no_decay_params = [], []
    for name, param in named_params:
        # Do not apply weight decay to the bias or any scaling parameters
        if 'scaling' in name or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [{
        'params': no_decay_params, 'weight_decay': 0.}, {
            'params': decay_params, 'weight_decay': decay}]


def create_calibration_dataloader(
        dataset: Dataset, batch_size: int, num_workers: int, subset_size: int) -> DataLoader:

    all_indices = np.arange(len(dataset))
    cur_indices = np.random.choice(all_indices, size=subset_size)
    subset = Subset(dataset, cur_indices)
    loader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return loader


def get_cifar10_dataloaders(
        data_root: str,
        batch_size_train: int = 128,
        batch_size_test: int = 100,
        num_workers: int = 2,
        pin_memory: bool = True,
        download: bool = False) -> Tuple[Type[DataLoader]]:

    mean, std = [0.491, 0.482, 0.447], [0.247, 0.243, 0.262]

    # create training dataloader
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    trainset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=download,
        transform=transform_train,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # creating the validation dataloader
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),])
    testset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download,
        transform=transform_test,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return trainloader, testloader


def apply_act_calibrate(calib_loader, model):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with calibration_mode(model):
            for images, _ in tqdm(calib_loader):
                images = images.to(device)
                images = images.to(dtype)
                model(images)


def apply_bias_correction(calib_loader, model: nn.Module):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with bias_correction_mode(model):
            for (images, _) in tqdm(calib_loader):
                images = images.to(device)
                images = images.to(dtype)
                model(images)


def _prepare_bias_corrected_quant_model(model: nn.Module):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    images = torch.randn(10, 3, 32, 32)
    images = images.to(device)
    images = images.to(dtype)
    with torch.no_grad():
        with bias_correction_mode(model):
            model(images)


def train_for_epoch(trainloader, model, criterion, optimizer, reg_weight: float = 1e-3):
    model.train()
    model = model.to(device)

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

    progress_bar = tqdm(trainloader)
    for _, (images, targets) in enumerate(progress_bar):
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
def evaluate_topk_accuracies(testloader, model, criterion):
    model.eval()
    model = model.to(device)

    progress_bar = tqdm(testloader)

    top_1, top_5, tot_loss = 0., 0., 0.
    for _, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = targets.to(device)
        outputs: Tensor = model(images)
        loss: Tensor = criterion(outputs, targets)

        # Evaluating Top-1 and Top-5 accuracy
        _, y_pred = outputs.topk(5, 1, True, True)
        y_pred = y_pred.t()
        correct = y_pred.eq(targets.view(1, -1).expand_as(y_pred))
        top_1 += correct[0].float().sum().item()
        top_5 += correct.float().sum().item()
        tot_loss += loss.item() * images.size(0)
    top_1 /= len(testloader.dataset)
    top_5 /= len(testloader.dataset)
    tot_loss /= len(testloader.dataset)
    return top_1, top_5, tot_loss
