# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import pprint
import random
import torch

from brevitas_examples.super_resolution.models import get_model_by_name
from brevitas_examples.super_resolution.utils import device
from brevitas_examples.super_resolution.utils import evaluate_accumulator_bit_widths
from brevitas_examples.super_resolution.utils import export
from brevitas_examples.super_resolution.utils import get_bsd300_dataloaders
from brevitas_examples.super_resolution.utils import validate

SEED = 123456

desc = """Evaluating single-image super resolution models on the BSD300 dataset.

Example:
>> python eval_model.py --data-dir=data --model-path=outputs/model.pth --model=quant_espcn_w8a8 --upscale-factor=2
"""

parser = argparse.ArgumentParser(description='PyTorch BSD300 Validation')
parser.add_argument('--data-dir', help='Path to folder containing BSD300 val folder')
parser.add_argument('--model-path', help='Path to PyTorch checkpoint')
parser.add_argument(
    '--save-path', type=str, default='outputs/', help='Save path for exported model')
parser.add_argument(
    '--model', type=str, default='quant_espcn_w8a8', help='Name of the model configuration')
parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
parser.add_argument('--batch-size', type=int, default=16, help='Minibatch size')
parser.add_argument('--upscale-factor', type=int, default=3, help='Upscaling factor')
parser.add_argument('--eval-acc-bw', action='store_true', default=False)
parser.add_argument('--save-model-io', action='store_true', default=False)
parser.add_argument('--export-to-qonnx', action='store_true', default=False)
parser.add_argument('--export-to-qcdq-onnx', action='store_true', default=False)
parser.add_argument('--export-to-qcdq-torch', action='store_true', default=False)


def main():
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)

    # initialize model, dataset, and training environment
    model = get_model_by_name(args.model, args.upscale_factor)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model = model.to(device)
    _, testloader = get_bsd300_dataloaders(
        args.data_dir,
        num_workers=args.workers,
        batch_size=args.batch_size,
        batch_size_test=1,
        upscale_factor=args.upscale_factor,
        download=True)

    test_psnr = validate(testloader, model, args)
    print(f"[{args.model}_x{args.upscale_factor}] test_psnr={test_psnr:.2f}")

    # evaluate accumulator bit widths
    if args.eval_acc_bw:
        stats_dict = {
            'acc_bit_widths': evaluate_accumulator_bit_widths(model),
            'test_psnr': test_psnr}
        with open(f"{args.save_path}/stats.json", "w") as outfile:
            json.dump(stats_dict, outfile, indent=4)
        pretty_stats_dict = pprint.pformat(stats_dict, sort_dicts=False)
        print(pretty_stats_dict)

    # save and export model
    export(model, testloader, args)


if __name__ == '__main__':
    main()
