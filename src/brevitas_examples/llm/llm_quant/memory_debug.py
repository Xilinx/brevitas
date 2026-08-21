# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path
from typing import Any
from typing import Optional

import torch

_GIB = 2 ** 30


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    try:
        from torch.distributed.tensor import DTensor
        if isinstance(tensor, DTensor):
            return tensor.to_local()
    except ImportError:
        pass
    return tensor


def _tensor_bytes(tensor: torch.Tensor) -> int:
    tensor = _local_tensor(tensor)
    return 0 if tensor.is_meta else tensor.numel() * tensor.element_size()


def _nested_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return _tensor_bytes(value)
    if isinstance(value, dict):
        return sum(_nested_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nested_tensor_bytes(item) for item in value)
    return 0


def model_memory_bytes(model: Optional[torch.nn.Module]) -> tuple[int, int, int]:
    if model is None:
        return 0, 0, 0
    parameter_bytes = sum(_tensor_bytes(parameter) for parameter in model.parameters())
    gradient_bytes = sum(
        _tensor_bytes(parameter.grad)
        for parameter in model.parameters()
        if parameter.grad is not None)
    quantizer_bytes = sum(
        _tensor_bytes(parameter) for name,
        parameter in model.named_parameters() if any(
            token in name for token in ('.weight_quant.', '.input_quant.', '.output_quant.')))
    return parameter_bytes, gradient_bytes, quantizer_bytes


def optimizer_memory_bytes(optimizer: Optional[torch.optim.Optimizer]) -> int:
    if optimizer is None:
        return 0
    optimizer = getattr(optimizer, 'optimizer', optimizer)
    if hasattr(optimizer, 'optimizers'):
        return sum(optimizer_memory_bytes(sub_optimizer) for sub_optimizer in optimizer.optimizers)
    return _nested_tensor_bytes(optimizer.state)


def log_memory(
        label: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        reset_peak: bool = False) -> None:
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    free, total = torch.cuda.mem_get_info()
    device_used = total - free
    external = max(device_used - reserved, 0)
    parameter_bytes, gradient_bytes, quantizer_bytes = model_memory_bytes(model)
    optimizer_bytes = optimizer_memory_bytes(optimizer)
    rank = int(os.environ.get('RANK', '0'))
    device = torch.cuda.current_device()

    print(
        f"[memory][rank={rank}][cuda={device}][{label}] "
        f"allocated={allocated / _GIB:.2f}GiB "
        f"reserved={reserved / _GIB:.2f}GiB "
        f"peak_allocated={peak_allocated / _GIB:.2f}GiB "
        f"peak_reserved={peak_reserved / _GIB:.2f}GiB "
        f"device_used={device_used / _GIB:.2f}GiB "
        f"external={external / _GIB:.2f}GiB "
        f"parameters={parameter_bytes / _GIB:.2f}GiB "
        f"gradients={gradient_bytes / _GIB:.2f}GiB "
        f"optimizer={optimizer_bytes / _GIB:.2f}GiB "
        f"quantizer_parameters={quantizer_bytes / _GIB:.2f}GiB",
        flush=True)


def log_quantization_configuration(model: torch.nn.Module) -> None:
    from brevitas.core.quant import IntQuant
    from brevitas.nn import QuantLinear

    quant_linears = [module for module in model.modules() if isinstance(module, QuantLinear)]
    int_quants = [
        quant_module for linear in quant_linears for quant_module in linear.weight_quant.modules()
        if isinstance(quant_module, IntQuant)]
    rank = int(os.environ.get('RANK', '0'))
    print(
        f"[memory][rank={rank}][configuration] "
        f"quant_linears={len(quant_linears)} "
        f"linear_checkpointing={sum(module.quant_checkpointing for module in quant_linears)} "
        f"linear_recompute={sum(module.quant_recompute for module in quant_linears)} "
        f"int_quants={len(int_quants)} "
        f"memory_efficient_int_quants={sum(module.memory_efficient for module in int_quants)}",
        flush=True)


def register_first_quant_linear_memory_hooks(model: torch.nn.Module) -> None:
    from brevitas.nn import QuantLinear

    quant_linear = next((module for module in model.modules() if isinstance(module, QuantLinear)),
                        None)
    if quant_linear is None:
        return
    handles = []

    def pre_forward(module, args):
        log_memory('first_quant_linear_pre_forward', model)

    def post_forward(module, args, output):
        log_memory('first_quant_linear_post_forward', model)
        for handle in handles:
            handle.remove()

    handles.append(quant_linear.register_forward_pre_hook(pre_forward))
    handles.append(quant_linear.register_forward_hook(post_forward))


def start_memory_history() -> None:
    torch.cuda.memory._record_memory_history(max_entries=100000)


def dump_memory_snapshot(path: str) -> None:
    rank = int(os.environ.get('RANK', '0'))
    snapshot_path = Path(path)
    if snapshot_path.suffix:
        snapshot_path = snapshot_path.with_name(
            f"{snapshot_path.stem}_rank{rank}{snapshot_path.suffix}")
    else:
        snapshot_path = snapshot_path / f"memory_rank{rank}.pickle"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._dump_snapshot(str(snapshot_path))
