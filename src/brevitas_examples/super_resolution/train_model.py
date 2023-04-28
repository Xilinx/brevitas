# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import os
import pprint
import random

import torch
import torch.nn as nn
import torch.optim as optim

from brevitas_examples.super_resolution.models import get_model_by_name
from brevitas_examples.super_resolution.utils import device
from brevitas_examples.super_resolution.utils import evaluate_accumulator_bit_widths
from brevitas_examples.super_resolution.utils import export
from brevitas_examples.super_resolution.utils import get_bsd300_dataloaders
from brevitas_examples.super_resolution.utils import train_for_epoch
from brevitas_examples.super_resolution.utils import validate

SEED = 123456

desc = """Training single-image super resolution models on the BSD300 dataset.

Example:
>> python train_model.py --data-dir=data/ --model=quant_espcn_x3_finn_a2q_w4a4_14b
"""

parser = argparse.ArgumentParser(description='PyTorch BSD300 Validation')
parser.add_argument('--data-dir', help='Path to folder containing BSD300 val folder')
parser.add_argument(
    '--save-path', type=str, default='outputs/', help='Save path for exported model')
parser.add_argument(
    '--model', type=str, default='quant_espcn_x3_w8a8', help='Name of the model configuration')
parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
parser.add_argument('--batch-size', type=int, default=16, help='Minibatch size')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--upscale-factor', type=int, default=3, help='Upscaling factor')
parser.add_argument('--total-epochs', type=int, default=30, help='Total number of training epochs')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
parser.add_argument('--save-model-io', action='store_true', default=False)
parser.add_argument('--export-to-qonnx', action='store_true', default=False)
parser.add_argument('--export-to-qcdq-onnx', action='store_true', default=False)
parser.add_argument('--export-to-qcdq-torch', action='store_true', default=False)


def main():
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    # initialize model, dataset, and training environment
    model = get_model_by_name(args.model)
    model = model.to(device)
    trainloader, testloader = get_bsd300_dataloaders(
        args.data_dir,
        num_workers=args.workers,
        batch_size=args.batch_size,
        batch_size_test=1,
        download=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # train model
    for ep in range(args.total_epochs):
        train_loss = train_for_epoch(trainloader, model, criterion, optimizer, args)
        test_psnr = validate(testloader, model, args)
        print(f"[Epoch {ep:03d}] train_loss={train_loss:.4f}, test_psnr={test_psnr:.2f}")

    # save checkpoint
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(model.state_dict(), f"{args.save_path}/checkpoint.pth")
    print(f"Saved model checkpoint to {args.save_path}/checkpoint.pth")

    # evaluate accumulator bit widths
    stats_dict = {
        'acc_bit_widths': evaluate_accumulator_bit_widths(model),
        'performance': {
            'test_psnr': test_psnr, 'train_loss': train_loss}}
    with open(f"{args.save_path}/stats.json", "w") as outfile:
        json.dump(stats_dict, outfile, indent=4)
    pretty_stats_dict = pprint.pformat(stats_dict, sort_dicts=False)
    print(pretty_stats_dict)

    # save and export model
    export(model, testloader, args)


if __name__ == '__main__':
    main()
