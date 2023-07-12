# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
import warnings

import numpy as np
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from brevitas.export import export_onnx_qcdq
from brevitas.export import export_qonnx
from brevitas.export import export_torch_qcdq

from .train import device


def save_model_io(inp: Tensor, out: Tensor, save_path: str):
    with open(f"{save_path}/input.npy", "wb") as f:
        np.save(f, inp.cpu().numpy())
    with open(f"{save_path}/output.npy", "wb") as f:
        np.save(f, out.cpu().numpy())


def export(model: nn.Module, testloader: DataLoader, args: Namespace, opset_version: int = 11):
    save_path = args.save_path
    inp = testloader.dataset[0][0].unsqueeze(0)  # NCHW
    if args.save_model_io:
        inp = inp.to(device)
        model = model.to(device)
        save_model_io(inp, model(inp).detach(), save_path)
        print(f"Saved I/O to {save_path} as numpy arrays")
    if args.export_to_qonnx:
        export_qonnx(
            model.cpu(),
            input_t=inp.cpu(),
            export_path=f"{save_path}/qonnx_model.onnx",
            opset_version=opset_version)
        print(f"Saved QONNX model to {save_path}/qonnx_model.onnx")
    if args.export_to_qcdq_onnx:
        if opset_version < 13:
            warnings.warn("Need opset 13+ to support per-channel quantization.")
        else:
            export_onnx_qcdq(
                model.cpu(),
                input_t=inp.cpu(),
                export_path=f"{save_path}/qcdq_onnx_model.onnx",
                opset_version=opset_version)
            print(f"Saved QCDQ ONNX model to {save_path}/qcdq_onnx_model.onnx")
    if args.export_to_qcdq_torch:
        export_torch_qcdq(
            model.cpu(), input_t=inp.cpu(), export_path=f"{save_path}/qcdq_torch_model.pt")
        print(f"Saved QCDQ TorchScript model to {save_path}/qcdq_torch_model.pt")
