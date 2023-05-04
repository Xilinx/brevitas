# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import pprint
import random
import torch
import numpy as np

from brevitas_examples.super_resolution.models import get_model_by_name
from brevitas_examples.super_resolution.utils import device
from brevitas_examples.super_resolution.utils import evaluate_accumulator_bit_widths
from brevitas_examples.super_resolution.utils import export
from brevitas_examples.super_resolution.utils import get_bsd300_dataloaders
from brevitas_examples.super_resolution.utils import validate

random_seed = 123456

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

desc = """Evaluating single-image super resolution models on the BSD300 dataset.

Example:
>> python eval_model.py --data_root=data --model-path=outputs/model.pth --model=quant_espcn_x2_w8a8_base --upscale-factor=2
"""

parser = argparse.ArgumentParser(description='PyTorch BSD300 Validation')
parser.add_argument('--data_root', help='Path to folder containing BSD300 val folder')
parser.add_argument('--model_path', help='Path to PyTorch checkpoint')
parser.add_argument(
    '--save_path', type=str, default='outputs/', help='Save path for exported model')
parser.add_argument(
    '--model', type=str, default='quant_espcn_x2_w8a8_base', help='Name of the model configuration')
parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
parser.add_argument('--batch_size', type=int, default=16, help='Minibatch size')
parser.add_argument('--upscale_factor', type=int, default=3, help='Upscaling factor')
parser.add_argument('--eval_acc_bw', action='store_true', default=False)
parser.add_argument('--save_model_io', action='store_true', default=False)
parser.add_argument('--export_to_qonnx', action='store_true', default=False)
parser.add_argument('--export_to_qcdq_onnx', action='store_true', default=False)
parser.add_argument('--export_to_qcdq_torch', action='store_true', default=False)


def main():
    args = parser.parse_args()

    # initialize model, dataset, and training environment
    model = get_model_by_name(args.model)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model = model.to(device)
    _, testloader = get_bsd300_dataloaders(
        args.data_root,
        num_workers=args.workers,
        batch_size=args.batch_size,
        batch_size_test=1,
        upscale_factor=args.upscale_factor,
        download=True)

    test_psnr = validate(testloader, model, args)
    print(f"[{args.model}] test_psnr={test_psnr:.2f}")

    # evaluate accumulator bit widths
    if args.eval_acc_bw:
        inp = testloader.dataset[0][0].unsqueeze(0).to(device)
        stats_dict = {
            'acc_bit_widths': evaluate_accumulator_bit_widths(model, inp),
            'test_psnr': test_psnr}
        with open(f"{args.save_path}/stats.json", "w") as outfile:
            json.dump(stats_dict, outfile, indent=4)
        pretty_stats_dict = pprint.pformat(stats_dict, sort_dicts=False)
        print(pretty_stats_dict)

    # save and export model
    export(model, testloader, args)


if __name__ == '__main__':
    main()
