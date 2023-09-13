# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import copy
from hashlib import sha256
import json
import os
import pprint
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

from brevitas_examples.super_resolution.models import get_model_by_name
from brevitas_examples.super_resolution.utils import device
from brevitas_examples.super_resolution.utils import evaluate_accumulator_bit_widths
from brevitas_examples.super_resolution.utils import evaluate_avg_psnr
from brevitas_examples.super_resolution.utils import export
from brevitas_examples.super_resolution.utils import get_bsd300_dataloaders
from brevitas_examples.super_resolution.utils import train_for_epoch

random_seed = 123456

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

desc = """Training single-image super resolution models on the BSD300 dataset.

Example:
>> python train_model.py --data_root=data/ --model=quant_espcn_x2_w8a8_a2q_16b
"""

parser = argparse.ArgumentParser(description='PyTorch BSD300 Validation')
parser.add_argument('--data_root', help='Path to folder containing BSD300 val folder')
parser.add_argument(
    '--save_path', type=str, default='outputs/', help='Save path for exported model')
parser.add_argument(
    '--model',
    type=str,
    default='quant_espcn_x2_w8a8_a2q_16b',
    help='Name of the model configuration')
parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
parser.add_argument('--batch_size', type=int, default=8, help='Minibatch size')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--total_epochs', type=int, default=500, help='Total number of training epochs')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
parser.add_argument('--step_size', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.999)
parser.add_argument('--eval_acc_bw', action='store_true', default=False)
parser.add_argument('--save_pth_ckpt', action='store_true', default=False)
parser.add_argument('--save_model_io', action='store_true', default=False)
parser.add_argument('--export_to_qonnx', action='store_true', default=False)
parser.add_argument('--export_to_qcdq_onnx', action='store_true', default=False)
parser.add_argument('--export_to_qcdq_torch', action='store_true', default=False)


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


def main():
    args = parser.parse_args()

    # initialize model, dataset, and training environment
    model = get_model_by_name(args.model)
    model = model.to(device)
    trainloader, testloader = get_bsd300_dataloaders(
        args.data_root,
        num_workers=args.workers,
        batch_size=args.batch_size,
        upscale_factor=model.upscale_factor,
        download=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter_params(model.named_parameters(), args.weight_decay),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)
    scheduler = lrs.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # train model
    best_psnr, best_weights = 0., copy.deepcopy(model.state_dict())
    for ep in range(args.total_epochs):
        train_loss = train_for_epoch(trainloader, model, criterion, optimizer)
        test_psnr = evaluate_avg_psnr(testloader, model)
        scheduler.step()
        print(f"[Epoch {ep:03d}] train_loss={train_loss:.4f}, test_psnr={test_psnr:.2f}")
        if test_psnr >= best_psnr:
            best_weights = copy.deepcopy(model.state_dict())
            best_psnr = test_psnr
    model.load_state_dict(best_weights)
    model = model.to(device)
    test_psnr = evaluate_avg_psnr(testloader, model)
    print(f"Final test_psnr={test_psnr:.2f}")

    # save checkpoint
    os.makedirs(args.save_path, exist_ok=True)
    if args.save_pth_ckpt:
        ckpt_path = f"{args.save_path}/{args.model}.pth"
        torch.save(best_weights, ckpt_path)
        with open(ckpt_path, "rb") as _file:
            bytes = _file.read()
            model_tag = sha256(bytes).hexdigest()[:8]
        new_ckpt_path = f"{args.save_path}/{args.model}-{model_tag}.pth"
        os.rename(ckpt_path, new_ckpt_path)
        print(f"Saved model checkpoint to {new_ckpt_path}")

    # evaluate accumulator bit widths
    if args.eval_acc_bw:
        inp = testloader.dataset[0][0].unsqueeze(0).to(device)
        stats_dict = {
            'acc_bit_widths': evaluate_accumulator_bit_widths(model, inp),
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
