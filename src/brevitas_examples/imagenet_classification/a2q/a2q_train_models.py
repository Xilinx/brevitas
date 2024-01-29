# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import copy
from hashlib import sha256
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

import brevitas.config as config
from brevitas.export import export_qonnx
from brevitas_examples.imagenet_classification.a2q.ep_init import apply_bias_correction
from brevitas_examples.imagenet_classification.a2q.ep_init import apply_ep_init
import brevitas_examples.imagenet_classification.a2q.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=str, required=True)
parser.add_argument("--model-name", type=str, default="quant_resnet18_w4a4_a2q_32b")
parser.add_argument("--save-path", type=str, default='outputs/')
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--pin-memory", action="store_true", default=False)
parser.add_argument("--batch-size-train", type=int, default=256)
parser.add_argument("--batch-size-test", type=int, default=512)
parser.add_argument("--batch-size-calibration", type=int, default=256)
parser.add_argument("--calibration-samples", type=int, default=1000)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--lr-init", type=float, default=1e-3)
parser.add_argument("--lr-step-size", type=int, default=30)
parser.add_argument("--lr-gamma", type=float, default=0.1)
parser.add_argument("--total-epochs", type=int, default=90)
parser.add_argument("--pretrained", action="store_true", default=False)
parser.add_argument("--save-ckpt", action="store_true", default=False)
parser.add_argument("--apply-bias-corr", action="store_true", default=False)
parser.add_argument("--apply-ep-init", action="store_true", default=False)
parser.add_argument("--export-to-qonnx", action="store_true", default=False)

# ignore missing keys when loading pre-trained checkpoint
config.IGNORE_MISSING_KEYS = True

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# create a random input for graph tracing
random_inp = torch.randn(1, 3, 32, 32)

if __name__ == "__main__":

    args = parser.parse_args()

    config.JIT_ENABLED = not args.export_to_qonnx

    # Initialize dataloaders
    print(f"Loading CIFAR10 dataset from {args.data_root}...")
    trainloader, testloader = utils.get_cifar10_dataloaders(
        data_root=args.data_root,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory)
    calibloader = utils.create_calibration_dataloader(
        dataset=trainloader.dataset,
        batch_size=args.batch_size_calibration,
        num_workers=args.num_workers,
        subset_size=args.calibration_samples)

    print(
        f"Initializating {args.model_name} from",
        "checkpoint..." if args.pretrained else "scratch...")
    model = utils.get_model_by_name(args.model_name, args.pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        utils.filter_params(model.named_parameters(), args.weight_decay),
        lr=args.lr_init,
        weight_decay=args.weight_decay)
    scheduler = lrs.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # Calibrate the quant model on the calibration dataset
    if args.apply_ep_init:
        print("Applying EP-init:")
        apply_ep_init(model, random_inp)

    if args.apply_bias_corr:
        print("Applying bias correction:")
        apply_bias_correction(calibloader, model)

    best_top_1, best_weights = 0., copy.deepcopy(model.state_dict())
    for epoch in range(args.total_epochs):

        train_loss = utils.train_for_epoch(trainloader, model, criterion, optimizer)
        test_top_1, test_top_5, test_loss = utils.evaluate_topk_accuracies(testloader, model, criterion)
        scheduler.step()

        print(
            f"[Epoch {epoch:03d}]",
            f"train_loss={train_loss:.3f},",
            f"test_loss={test_loss:.3f},",
            f"test_top_1={test_top_1:.1%},",
            f"test_top_5={test_top_5:.1%}",
            sep=" ")

        if test_top_1 >= best_top_1:
            best_weights = copy.deepcopy(model.state_dict())
            best_top_1 = test_top_1

    model.load_state_dict(best_weights)
    top_1, top_5, loss = utils.evaluate_topk_accuracies(testloader, model, criterion)
    print(f"Final top_1={top_1:.1%}, top_5={top_5:.1%}, loss={loss:.3f}")

    # save checkpoint
    os.makedirs(args.save_path, exist_ok=True)
    if args.save_ckpt:
        ckpt_path = f"{args.save_path}/{args.model_name}.pth"
        torch.save(best_weights, ckpt_path)
        with open(ckpt_path, "rb") as _file:
            bytes = _file.read()
            model_tag = sha256(bytes).hexdigest()[:8]
        new_ckpt_path = f"{args.save_path}/{args.model_name}-{model_tag}.pth"
        os.rename(ckpt_path, new_ckpt_path)
        print(f"Saved model checkpoint to {new_ckpt_path}")

    if args.export_to_qonnx:
        export_qonnx(
            model.cpu(),
            input_t=random_inp.cpu(),
            export_path=f"{args.save_path}/{args.model_name}-{model_tag}.onnx")
