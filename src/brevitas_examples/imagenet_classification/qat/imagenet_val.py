# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
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

from brevitas_examples.imagenet_classification.models import model_with_cfg
from brevitas_examples.imagenet_classification.utils import accuracy
from brevitas_examples.imagenet_classification.utils import AverageMeter
from brevitas_examples.imagenet_classification.utils import SEED
from brevitas_examples.imagenet_classification.utils import validate

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--imagenet-dir', help='path to folder containing Imagenet val folder')
parser.add_argument('--model', type=str, default='quant_mobilenet_v1_4b', help='Name of the model')
parser.add_argument('--pretrained', action='store_true', help='Load pretrained checkpoint')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--batch-size', default=256, type=int, help='Minibatch size')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--shuffle', action='store_true', help='Shuffle validation data.')


def main():
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    if args.pretrained:
        pretrained = 'quant_weights'
    else:
        pretrained = None
    model, cfg = model_with_cfg(args.model, pretrained)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        cudnn.benchmark = True

    valdir = os.path.join(args.imagenet_dir, 'val')
    mean = [
        float(cfg.get('PREPROCESS', 'MEAN_0')),
        float(cfg.get('PREPROCESS', 'MEAN_1')),
        float(cfg.get('PREPROCESS', 'MEAN_2'))]
    std = [
        float(cfg.get('PREPROCESS', 'STD_0')),
        float(cfg.get('PREPROCESS', 'STD_1')),
        float(cfg.get('PREPROCESS', 'STD_2'))]
    normalize = transforms.Normalize(mean=mean, std=std)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,])),
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.workers,
        pin_memory=True)

    validate(val_loader, model)
    return


if __name__ == '__main__':
    main()
