from typing import Tuple, Union, Optional
from abc import ABC
from packaging import version

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
from brevitas.utils.jit_utils import onnx_export_patched
from ..base import BaseManager, _set_export_mode
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
        if torch_version >= version.parse('1.3.0'):
            export_kwargs['keep_initializers_as_inputs'] = True

    @classmethod
    def solve_enable_onnx_checker(cls, export_kwargs):
        if torch_version >= version.parse('1.5.0'):
            export_kwargs['enable_onnx_checker'] = False

    @classmethod
    def export_onnx(
            cls,
            module: Module,
            input_shape: Tuple[int, ...],
            export_path: str,
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

        cls.solve_keep_initializers_as_inputs(kwargs)
        cls.solve_enable_onnx_checker(kwargs)

        with torch.no_grad():
            training_state = module.training
            module = module.eval()
            module.apply(cls.set_export_handler)
            if input_t is None:
                input_t = torch.empty(input_shape, dtype=torch.float)
            # do a forward pass with the dummy input to e.g. store input/output shapes
            cls.cache_inp_out(module, input_t)
            # override any given input_t to make sure it's a standard PyTorch tensor
            input_t = torch.empty(input_shape, dtype=torch.float)
            # enable export mode, this triggers collecting export values into handlers
            module.apply(lambda m: _set_export_mode(m, enabled=True))
            # temporarily disable input caching to avoid collectives empty debug values
            module.apply(lambda m: _override_inp_caching_mode(m, enabled=False))
            # perform export pass
            onnx_export_patched(module, input_t, export_path, **kwargs)
            # restore the model to previous properties
            module.apply(lambda m: _restore_inp_caching_mode(m))
            module.apply(lambda m: _set_export_mode(m, enabled=False))
            module.train(training_state)
            # do some cleanup on the exported ONNX model
            model = onnx.load(export_path)
            model = opt.optimize(model, cls.onnx_passes)
            model = cls.apply_model_transforms(model)
            onnx.save(model, export_path)