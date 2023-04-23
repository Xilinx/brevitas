# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import configparser
import os
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from brevitas_examples.super_resolution.models import get_model_by_name

SEED = 123456

parser = argparse.ArgumentParser(description='PyTorch BSD300 Validation')
parser.add_argument('--data-dir', help='path to folder containing BSD300 val folder')
parser.add_argument('--model', type=str, default='quant_espcn_x3_v1_4b', help='Name of the model')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', default=16, type=int, help='Minibatch size')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--learning-rate', default=1e-3, help='Learning rate')
parser.add_argument('--upscale-factor', default=3, help='Upscaling factor')
parser.add_argument('--total-epochs', default=3, help='Total number of training epochs')
parser.add_argument('--weight-decay', default=1e-5, help='Weight decay')


def main():
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    model = get_model_by_name(args.model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # NOTE - toy dataset to be replaced with bsd300x3
    transform = transforms.Resize(510 // args.upscale_factor)
    y = torch.randn(50, 1, 510, 510)
    x = transform(y)
    val_dataset = TensorDataset(x, y)
    train_dataset = TensorDataset(x, y)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    train(train_loader, model, args)
    validate(val_loader, model, args)
    export(val_loader, model, args)


def train(train_loader, model, args):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)
    for ep in range(args.total_epochs):
        for (images, targets) in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


def validate(val_loader, model, args):
    model.eval()
    tot_loss = 0.
    with torch.no_grad():
        num_batches = len(val_loader)
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            output = model(images)
            tot_loss += (output - target).pow(2).sum().item()
    avg_loss = tot_loss / len(val_loader.dataset)
    psnr = 10. * math.log10(1. / avg_loss)
    print(f"Average peak signal-to-noise ratio = {psnr:.3f}")


def export(val_loader, model, args):
    # TODO - qonnx export
    pass


if __name__ == '__main__':
    main()
