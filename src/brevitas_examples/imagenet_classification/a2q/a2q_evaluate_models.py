# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from hashlib import sha256
import os
import random

import numpy as np
import torch
import torch.nn as nn

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
    "--load-from-path",
    type=str,
    default=None,
    help="Optional local path to load torch checkpoint from. Default: None")
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
    "--batch-size", type=int, default=512, help="Batch size for the dataloader. Default: 512")
parser.add_argument(
    "--save-torch-model",
    action="store_true",
    default=False,
    help="If true, save torch model to specified save path.")
parser.add_argument(
    "--export-to-qonnx", action="store_true", default=False, help="If true, export model to QONNX.")

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
        batch_size_train=args.batch_size,  # does not matter here
        batch_size_test=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory)

    # if load-from-path is not specified, then use the pre-trained checkpoint
    model = utils.get_model_by_name(args.model_name, pretrained=args.load_from_path is None)
    if args.load_from_path is not None:
        # note that if you used bias correction, you may need to prepare the model for the
        # new biases that were introduced. See `utils.get_model_by_name` for more details.
        state_dict = torch.load(args.load_from_path, map_location="cpu")
        model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()

    top_1, top_5, loss = utils.evaluate_topk_accuracies(testloader, model, criterion)
    print(f"Final top_1={top_1:.1%}, top_5={top_5:.1%}, loss={loss:.3f}")

    # save checkpoint
    os.makedirs(args.save_path, exist_ok=True)
    if args.save_torch_model:
        ckpt_path = f"{args.save_path}/{args.model_name}.pth"
        torch.save(model.state_dict(), ckpt_path)
        with open(ckpt_path, "rb") as _file:
            bytes = _file.read()
            model_tag = sha256(bytes).hexdigest()[:8]
        new_ckpt_path = f"{args.save_path}/{args.model_name}-{model_tag}.pth"
        os.rename(ckpt_path, new_ckpt_path)
        print(f"Saved model checkpoint to: {new_ckpt_path}")

    if args.export_to_qonnx:
        export_qonnx(
            model.cpu(),
            input_t=random_inp.cpu(),
            export_path=f"{args.save_path}/{args.model_name}-{model_tag}.onnx")
