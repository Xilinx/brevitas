# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from contextlib import nullcontext
from io import BytesIO
from typing import Optional, Tuple, Union
import warnings

from packaging import version

from brevitas.export.onnx.qonnx.handler import BrevitasFloatQuantProxyHandler
from brevitas.export.onnx.standard.qcdq.handler import StdFloatQCDQCastONNXMixin

try:
    import onnx
    import onnxoptimizer as opt
except ModuleNotFoundError:
    onnx = None
    opt = None

import torch
from torch import Tensor
from torch.nn import Module
import torch.onnx

from brevitas import torch_version
from brevitas.quant_tensor import QuantTensor

from ..manager import _override_act_caching_mode
from ..manager import _restore_act_caching_mode
from ..manager import BaseManager
from ..manager import ExportContext


# workaround for fp8 not having many operators implemented
class PatchFp8Ops():

    def __init__(self):
        self.lib = None

    def __enter__(self):
        import numpy as np

        if torch_version >= version.parse('2.1.0') and torch_version < version.parse('2.5'):
            self.lib = torch.library.Library("aten", "IMPL")

            def equal_cpu(self, other):

                if (isinstance(self, Tensor) and
                        self.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)) or (
                            isinstance(other, Tensor) and
                            other.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)):
                    self = self.to(torch.float32)
                    other = other.to(torch.float32)
                    return torch.equal(self, other)
                else:
                    if isinstance(self, Tensor):
                        self = self.cpu().numpy()
                    else:
                        self = np.array(self)
                    if isinstance(other, Tensor):
                        other = other.cpu().numpy
                    else:
                        other = np.array(other)
                    res = bool(np.equal(self, other).all())
                    return torch.tensor([res])

            self.lib.impl("equal", equal_cpu, "CPU")

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.lib = None


class ONNXBaseManager(BaseManager, ABC):

    model_transforms = []
    onnx_passes = []
    custom_fns = []
    dequantize_tracing_input = True
    custom_opset = 1

    @classmethod
    def apply_model_transforms(cls, model):
        for tranform in cls.model_transforms:
            model = tranform(model)
        return model

    @classmethod
    def solve_keep_initializers_as_inputs(cls, export_kwargs):
        # See https://github.com/pytorch/pytorch/commit/7583519b870e33ee3182f330c1bb8663559697b6
        ka = 'keep_initializers_as_inputs'
        if torch_version >= version.parse('1.3.0') and ka not in export_kwargs:
            export_kwargs[ka] = True

    @classmethod
    def solve_enable_onnx_checker(cls, export_kwargs):
        ka = 'enable_onnx_checker'
        if (torch_version >= version.parse('1.5.0') and torch_version <= version.parse('1.10.0') and
                ka not in export_kwargs):
            export_kwargs[ka] = False

    @classmethod
    def register_custom_fns(cls):
        for fn in cls.custom_fns:
            torch.onnx.register_custom_op_symbolic(
                f'{cls.target_name}::{fn.__name__}', fn.symbolic, cls.custom_opset)

    @classmethod
    def export_onnx(
            cls,
            module: Module,
            args: Optional[Union[Tensor, QuantTensor, Tuple]],
            export_path: Optional[str],
            input_shape: Optional[Tuple[int, ...]],
            input_t: Optional[Union[Tensor, QuantTensor]],
            disable_warnings,
            **onnx_export_kwargs):

        if onnx is None or opt is None:
            raise ModuleNotFoundError("Installation of onnx and onnxoptimizer is required.")
        if not (((input_shape is not None) + (input_t is not None) + (args is not None)) == 1):
            raise RuntimeError("Export requires one of input_shape, args, or input_t")

        cls.solve_keep_initializers_as_inputs(onnx_export_kwargs)
        cls.solve_enable_onnx_checker(onnx_export_kwargs)
        cls.register_custom_fns()

        with torch.no_grad():
            with ExportContext(cls):
                with warnings.catch_warnings():
                    if disable_warnings:
                        warnings.simplefilter("ignore")
                    training_state = module.training
                    module = module.eval()
                    module.apply(cls.set_export_handler)
                    if input_shape is not None:
                        args = torch.empty(input_shape, dtype=torch.float)
                    elif input_t is not None:
                        args = input_t
                    # do a forward pass with the dummy input to e.g. store input/output shapes
                    if isinstance(args, tuple) and not isinstance(args, QuantTensor):
                        # https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
                        if isinstance(args[-1], dict) and isinstance(args[-2], dict):
                            model_args = args[:-2] + (args[-1],)
                            model_kwargs = args[-2]
                        elif isinstance(args[-1], dict) and not isinstance(args[-2], dict):
                            model_args = args[:-1]
                            model_kwargs = args[-1]
                        else:
                            model_args = args
                            model_kwargs = {}
                        cls._cache_inp_out(module, *model_args, **model_kwargs)
                    else:
                        cls._cache_inp_out(module, args)
                    # Dequantize QuantTensor, if any and enabled
                    if isinstance(args, QuantTensor):
                        if cls.dequantize_tracing_input:
                            args = args.value
                        else:
                            args = (args,)
                    # enable export mode, this triggers collecting export values into handlers
                    cls.set_export_mode(module, enabled=True)
                    # temporarily disable input caching to avoid collectives empty debug values
                    module.apply(lambda m: _override_act_caching_mode(m, enabled=False))
                    # perform export pass
                    if export_path is not None:
                        export_target = export_path
                    else:
                        model_bytes = BytesIO()
                        export_target = model_bytes

                    # Check if we attached Float-related handlers, then we need to patch export
                    fp8_export_patch = False
                    for m in module.modules():
                        if isinstance(m,
                                      (StdFloatQCDQCastONNXMixin, BrevitasFloatQuantProxyHandler)):
                            fp8_export_patch = True

                    patch_export = PatchFp8Ops if fp8_export_patch else nullcontext
                    with patch_export():
                        torch.onnx.export(module, args, export_target, **onnx_export_kwargs)

                    # restore the model to previous properties
                    module.apply(lambda m: _restore_act_caching_mode(m))
                    cls.set_export_mode(module, enabled=False)
                    module.train(training_state)

                    # do some cleanup on the exported ONNX model
                    if export_path is not None:
                        model = onnx.load(export_path)
                    else:
                        model = onnx.ModelProto.FromString(model_bytes.getvalue())
                    model = opt.optimize(model, cls.onnx_passes)
                    model = cls.apply_model_transforms(model)
                    if export_path is not None:
                        onnx.save(model, export_path)
                    return model

    @classmethod
    def export(
            cls,
            module: Module,
            args: Optional[Union[Tensor, QuantTensor, Tuple]] = None,
            export_path: Optional[str] = None,
            input_shape: Optional[Tuple[int, ...]] = None,  # legacy syntax, alternative to args
            input_t: Optional[Union[Tensor,
                                    QuantTensor]] = None,  # legacy syntax, alternative to args
            disable_warnings=True,
            **onnx_export_kwargs):
        return cls.export_onnx(
            module, args, export_path, input_shape, input_t, disable_warnings, **onnx_export_kwargs)
