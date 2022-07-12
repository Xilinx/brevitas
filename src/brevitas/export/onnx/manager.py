from typing import Tuple, Union, Optional
from abc import ABC
from packaging import version
from contextlib import ExitStack
from io import BytesIO
import warnings

try:
    import onnx
    import onnxoptimizer as opt
except ModuleNotFoundError:
    onnx = None
    opt = None

import torch
import torch.onnx
from torch import Tensor
from torch.nn import Module

from brevitas import torch_version
from brevitas.quant_tensor import QuantTensor
from ..manager import BaseManager, ExportContext
from ..manager import _override_inp_caching_mode, _restore_inp_caching_mode


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
        if torch_version >= version.parse('1.5.0') \
            and torch_version <= version.parse('1.10.0') \
            and ka not in export_kwargs:
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
            input_shape: Optional[Tuple[int, ...]] = None,
            export_path: Optional[str] = None,
            input_t: Optional[Union[Tensor, QuantTensor]] = None,
            disable_warnings=True,
            **kwargs):

        if onnx is None or opt is None:
            raise ModuleNotFoundError("Installation of onnx and onnxoptimizer is required.")
        if input_shape is None and input_t is None:
            raise RuntimeError("Export requires to pass in either input_shape or input_t")
        if input_shape is not None and input_t is not None:
            raise RuntimeError("Export accepts either an input shape or an input tensor, not both")

        cls.solve_keep_initializers_as_inputs(kwargs)
        cls.solve_enable_onnx_checker(kwargs)
        cls.register_custom_fns()

        with torch.no_grad():
            with ExportContext(cls):
                with warnings.catch_warnings():
                    if disable_warnings:
                        warnings.simplefilter("ignore")
                    training_state = module.training
                    module = module.eval()
                    module.apply(cls.set_export_handler)
                    if input_t is None:
                        input_t = torch.empty(input_shape, dtype=torch.float)
                    # do a forward pass with the dummy input to e.g. store input/output shapes
                    cls._cache_inp_out(module, input_t)
                    # Dequantize QuantTensor, if any and enabled
                    if isinstance(input_t, QuantTensor):
                        if cls.dequantize_tracing_input:
                            input_t = input_t.value
                        else:
                            input_t = (input_t,)
                    # enable export mode, this triggers collecting export values into handlers
                    module.apply(lambda m: cls.set_export_mode(m, enabled=True))
                    # temporarily disable input caching to avoid collectives empty debug values
                    module.apply(lambda m: _override_inp_caching_mode(m, enabled=False))
                    # perform export pass
                    with ExitStack() as stack:
                        for mgr in cls._trace_patches():
                            stack.enter_context(mgr)
                        if export_path is not None:
                            export_target = export_path
                        else:
                            model_bytes = BytesIO()
                            export_target = model_bytes
                        torch.onnx.export(module, input_t, export_target, **kwargs)

                    # restore the model to previous properties
                    module.apply(lambda m: _restore_inp_caching_mode(m))
                    module.apply(lambda m: cls.set_export_mode(m, enabled=False))
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
            input_shape: Optional[Tuple[int, ...]] = None,
            export_path: Optional[str] = None,
            input_t: Optional[Union[Tensor, QuantTensor]] = None,
            **kwargs):
        return cls.export_onnx(module, input_shape, export_path, input_t, **kwargs)