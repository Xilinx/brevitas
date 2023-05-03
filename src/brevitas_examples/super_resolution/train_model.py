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
import torch.optim.lr_scheduler as lrs
from hashlib import sha256

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
>> python train_model.py --data-dir=data/ --model=quant_espcn_finn_a2q_w4a4_14b
"""

parser = argparse.ArgumentParser(description='PyTorch BSD300 Validation')
parser.add_argument('--data-dir', help='Path to folder containing BSD300 val folder')
parser.add_argument(
    '--save-path', type=str, default='outputs/', help='Save path for exported model')
parser.add_argument(
    '--model', type=str, default='quant_espcn_w8a8', help='Name of the model configuration')
parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
parser.add_argument('--batch-size', type=int, default=16, help='Minibatch size')
parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--upscale-factor', type=int, default=3, help='Upscaling factor')
parser.add_argument('--total-epochs', type=int, default=100, help='Total number of training epochs')
parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--step-size', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.98)
parser.add_argument('--eval-acc-bw', action='store_true', default=False)
parser.add_argument('--save-pth-ckpt', action='store_true', default=False)
parser.add_argument('--save-model-io', action='store_true', default=False)
parser.add_argument('--export-to-qonnx', action='store_true', default=False)
parser.add_argument('--export-to-qcdq-onnx', action='store_true', default=False)
parser.add_argument('--export-to-qcdq-torch', action='store_true', default=False)


def filter_params(named_params, decay):
    decay_params, no_decay_params = [], []
    for name, param in named_params:
        # Do not apply weight decay to the bias or any scaling parameters
        if 'scaling' in name or name.endswith(".bias"): 
            no_decay_params.append(param)
        else: 
            decay_params.append(param)
    return [
        {'params': no_decay_params, 'weight_decay': 0.},
        {'params': decay_params, 'weight_decay': decay}]


def main():
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    # initialize model, dataset, and training environment
    model = get_model_by_name(args.model, args.upscale_factor)
    model = model.to(device)
    trainloader, testloader = get_bsd300_dataloaders(
        args.data_dir,
        num_workers=args.workers,
        batch_size=args.batch_size,
        batch_size_test=1,
        upscale_factor=args.upscale_factor,
        download=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter_params(model.named_parameters(), args.weight_decay),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)
    scheduler = lrs.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # train model
    for ep in range(args.total_epochs):
        train_loss = train_for_epoch(trainloader, model, criterion, optimizer, args)
        test_psnr = validate(testloader, model, args)
        scheduler.step()
        print(f"[Epoch {ep:03d}] train_loss={train_loss:.4f}, test_psnr={test_psnr:.2f}")

    # save checkpoint
    os.makedirs(args.save_path, exist_ok=True)
    if args.save_pth_ckpt:
        ckpt_path = f"{args.save_path}/{args.model}_x{args.upscale_factor}.pth"
        torch.save(model.state_dict(), ckpt_path)
        with open(ckpt_path, "rb") as _file:
            bytes = _file.read()
            model_tag = sha256(bytes).hexdigest()[:8]
        new_ckpt_path = f"{args.save_path}/{args.model}_x{args.upscale_factor}-{model_tag}.pth"
        os.rename(ckpt_path, new_ckpt_path)
        print(f"Saved model checkpoint to {new_ckpt_path}")

    # evaluate accumulator bit widths
    if args.eval_acc_bw:
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
