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
import brevitas_examples.imagenet_classification.a2q.utils as utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-root", type=str, required=True, help="Directory where the dataset is stored.")
parser.add_argument(
    "--model-name",
    type=str,
    default="quant_resnet18_w4a4_a2q_32b",
    help="Name of model to train. Default: 'quant_resnet18_w4a4_a2q_32b'",
    choices=utils.model_impl.keys())
parser.add_argument(
    "--save-path",
    type=str,
    default="outputs/",
    help="Directory where to save checkpoints. Default: 'outputs/'")
parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
    help="Number of workers for the dataloader to use. Default: 0")
parser.add_argument(
    "--pin-memory",
    action="store_true",
    default=False,
    help="If true, pin memory for the dataloader.")
parser.add_argument(
    "--batch-size-train",
    type=int,
    default=256,
    help="Batch size for the training dataloader. Default: 256")
parser.add_argument(
    "--batch-size-test",
    type=int,
    default=512,
    help="Batch size for the testing dataloader. Default: 512")
parser.add_argument(
    "--batch-size-calibration",
    type=int,
    default=256,
    help="Batch size for the calibration dataloader. Default: 256")
parser.add_argument(
    "--calibration-samples",
    type=int,
    default=1000,
    help="Number of samples to use for calibration. Default: 1000")
parser.add_argument(
    "--weight-decay",
    type=float,
    default=1e-5,
    help="Weight decay for the Adam optimizer. Default: 0.00001")
parser.add_argument(
    "--lr-init", type=float, default=1e-3, help="Initial learning rate. Default: 0.001")
parser.add_argument(
    "--lr-step-size",
    type=int,
    default=30,
    help="Step size for the learning rate scheduler. Default: 30")
parser.add_argument(
    "--lr-gamma",
    type=float,
    default=0.1,
    help="Default gamma for the learning rate scheduler. Default: 0.1")
parser.add_argument(
    "--total-epochs", type=int, default=90, help="Total epoch to train the model for. Default: 90")
parser.add_argument(
    "--from-float-checkpoint",
    action="store_true",
    default=False,
    help="If true, use a pre-trained floating-point checkpoint.")
parser.add_argument(
    "--save-torch-model",
    action="store_true",
    default=False,
    help="If true, save torch model to specified save path.")
parser.add_argument(
    "--apply-act-calibration",
    action="store_true",
    default=False,
    help="If true, apply activation calibration to the quantized model.")
parser.add_argument(
    "--apply-bias-correction",
    action="store_true",
    default=False,
    help="If true, apply bias correction to the quantized model.")
parser.add_argument(
    "--apply-ep-init",
    action="store_true",
    default=False,
    help="If true, apply EP-init to the quantized model.")
parser.add_argument(
    "--export-to-qonnx", action="store_true", default=False, help="If true, export model to QONNX.")

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

    model = utils.get_model_by_name(
        args.model_name, init_from_float_checkpoint=args.from_float_checkpoint)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        utils.filter_params(model.named_parameters(), args.weight_decay),
        lr=args.lr_init,
        weight_decay=args.weight_decay)
    scheduler = lrs.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # Calibrate the quant model on the calibration dataset
    if args.apply_ep_init:
        print("Applying EP-init:")
        model = utils.apply_ep_init(model, random_inp)

    # Calibrate the quant model on the calibration dataset
    if args.apply_act_calibration:
        print("Applying activation calibration:")
        utils.apply_act_calibrate(calibloader, model)

    if args.apply_bias_correction:
        print("Applying bias correction:")
        utils.apply_bias_correction(calibloader, model)

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
    print(f"Final: top_1={top_1:.1%}, top_5={top_5:.1%}, loss={loss:.3f}")

    # save checkpoint
    os.makedirs(args.save_path, exist_ok=True)
    if args.save_torch_model:
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
