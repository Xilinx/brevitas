# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import configparser
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.dataset import TensorDataset

from brevitas_examples.super_resolution.models import model_with_cfg

SEED = 123456

parser = argparse.ArgumentParser(description='PyTorch BSD300 Validation')
parser.add_argument('--data-dir', help='path to folder containing BSD300 val folder')
parser.add_argument('--model', type=str, default='quant_espcn_v1_4b', help='Name of the model')
parser.add_argument('--pretrained', action='store_true', help='Load pretrained checkpoint')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', default=256, type=int, help='Minibatch size')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--shuffle', action='store_true', help='Shuffle validation data.')


def main():
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    model, cfg = model_with_cfg(args.model, args.pretrained)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        cudnn.benchmark = True

    print(model)

    upsampling_factor = 3
    x = torch.randn(1000, 1, 256, 256)
    y = torch.randn(1000, 1, 256 * upsampling_factor, 256 * upsampling_factor)
    val_dataset = TensorDataset(x, y)

    valdir = os.path.join(args.data_dir, 'val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True)

    validate(val_loader, model, args)
    return


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
    return


if __name__ == '__main__':
    main()
