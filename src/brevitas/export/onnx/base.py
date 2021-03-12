from typing import Tuple, Union, Optional
from abc import ABC
from packaging import version
from contextlib import ExitStack
from io import BytesIO

try:
    import onnx
    import onnx.optimizer as opt
except ModuleNotFoundError:
    onnx = None
    opt = None

import torch
import torch.onnx
from torch import Tensor
from torch.nn import Module

from brevitas import torch_version
from brevitas.quant_tensor import QuantTensor
from ..base import BaseManager, _set_export_mode, ExportContext
from ..base import _override_inp_caching_mode, _restore_inp_caching_mode


class ONNXBaseManager(BaseManager, ABC):

    model_transforms = []
    onnx_passes = []

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
        if torch_version >= version.parse('1.5.0') and ka not in export_kwargs:
            export_kwargs[ka] = False

    @classmethod
    def export(
            cls,
            module: Module,
            input_shape: Optional[Tuple[int, ...]] = None,
            export_path: Optional[str] = None,
            input_t: Optional[Union[Tensor, QuantTensor]] = None,
            **kwargs):
        return cls.export_onnx(module, input_shape, export_path, input_t, **kwargs)

    @classmethod
    def export_onnx(
            cls,
            module: Module,
            input_shape: Optional[Tuple[int, ...]] = None,
            export_path: Optional[str] = None,
            input_t: Optional[Union[Tensor, QuantTensor]] = None,
            **kwargs):
        """
        * input_shape : tuple describing the shape of network input e.g. (1, 1, 28, 28)
        * export_path : ONNX filename to export to
        * input_t : if specified, do an initial forward pass with this value. this
                    may be necessary for QuantTensor caching.
        * torch_onnx_kwargs : will be passed as kwargs to torch.onnx.export
        """

        if onnx is None or opt is None:
            raise ModuleNotFoundError("Installation of ONNX is required.")
        if input_shape is None and input_t is None:
            raise RuntimeError("Export requires to pass in either input_shape or input_t")
        if input_shape is not None and input_t is not None:
            raise RuntimeError("Export accepts either an input shape or an input tensor, not both")

        cls.solve_keep_initializers_as_inputs(kwargs)
        cls.solve_enable_onnx_checker(kwargs)

        with torch.no_grad():
            with ExportContext(cls):
                training_state = module.training
                module = module.eval()
                module.apply(cls.set_export_handler)
                if input_t is None:
                    input_t = torch.empty(input_shape, dtype=torch.float)
                # do a forward pass with the dummy input to e.g. store input/output shapes
                cls._cache_inp_out(module, input_t)
                # override any given input_t to make sure it's a standard PyTorch tensor
                input_t = torch.empty(input_t.shape, dtype=torch.float)
                # enable export mode, this triggers collecting export values into handlers
                module.apply(lambda m: _set_export_mode(m, enabled=True))
                # temporarily disable input caching to avoid collectives empty debug values
                module.apply(lambda m: _override_inp_caching_mode(m, enabled=False))
                # perform export pass
                with ExitStack() as stack:
                    for mgr in cls._trace_patches():
                        stack.enter_context(mgr)
                    if export_path is not None:
                        torch.onnx.export(module, input_t, export_path, **kwargs)
                    else:
                        model_bytes = BytesIO()
                        torch.onnx.export(module, input_t, model_bytes, **kwargs)
                # restore the model to previous properties
                module.apply(lambda m: _restore_inp_caching_mode(m))
                module.apply(lambda m: _set_export_mode(m, enabled=False))
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