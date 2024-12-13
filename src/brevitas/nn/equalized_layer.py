import functools
from inspect import signature
from typing import Callable, Optional

import torch

from brevitas.graph.hadamard import get_hadK
from brevitas.graph.hadamard import matmul_hadU
from brevitas.graph.hadamard import matmul_hadU_cuda
from brevitas.nn.quant_mha import QuantMultiheadAttention

try:
    import fast_hadamard_transform
except:
    fast_hadamard_transform = None

INPUT_NAMES = ['input', 'inp', 'query', 'x', 'hidden_states']


class EqualizedModule(torch.nn.Module):

    def __init__(self, scale_module, layer) -> None:
        super().__init__()
        self.scale = scale_module
        self.layer = layer

    def forward(self, *args, **kwargs):
        # Convert args + kwargs + defaults into kwargs
        bound_arguments = signature(self.layer.forward).bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        kwargs = bound_arguments.arguments

        possible_input_kwargs = INPUT_NAMES
        input_kwarg = [x for x in kwargs.keys() if x in possible_input_kwargs][0]
        x = kwargs[input_kwarg]
        out = x
        if 'key' in kwargs:
            if kwargs['key'].data_ptr() != out.data_ptr():
                raise ValueError(
                    "Cross MHA is not supported for activation equalization."
                    "Replace kwargs with positional args to avoid this exception.")
        out = self.scale(out)

        kwargs[input_kwarg] = out
        # QuantMultiheadAttention is not a subclass of MultiheadAttention
        # We need to preserve the correctness of the forward even after
        # quantization has been applied
        if isinstance(self.layer, (torch.nn.MultiheadAttention, QuantMultiheadAttention)):
            kwargs['key'] = out
            kwargs['value'] = out
        # We convert everything to args so that hooks can work correctly
        out = self.layer(*kwargs.values())
        return out


def _apply_ort_device(tensor, ort, *args):
    ort = ort.type_as(tensor)
    return torch.matmul(tensor, ort)


class RotatedModule(torch.nn.Module):

    def __init__(self, layer, had_mat=None, k=None) -> None:
        super().__init__()
        if had_mat is not None:
            self.had_mat = torch.nn.Parameter(had_mat).cpu()
        else:
            self.had_mat = None
        self.layer = layer
        self.k = k

    @property
    def weight(self) -> Optional[torch.Tensor]:
        return getattr(self.layer, 'weight', None)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return getattr(self.layer, 'bias', None)

    def forward(self, inp, **kwargs):
        is_cuda = 'cuda' in str(inp.device) and torch.version.cuda is not None
        # If k is None, we assume that an orthogonal matrix is used
        if self.k is None:
            inp = _apply_ort_device(inp, self.had_mat)
        else:
            if is_cuda and fast_hadamard_transform is not None:
                if self.had_mat is None or self.k is None:
                    had_K, K = get_hadK(inp.shape[-1])
                else:
                    had_K = self.had_mat
                    K = self.k
                inp = matmul_hadU_cuda(inp, had_K, K)
            else:
                inp = matmul_hadU(inp)
        o = self.layer(inp)

        return o


def rot_func_wrapper(weight: torch.Tensor, rot_mat: torch.Tensor, rotation_function: Callable):
    weight_shape = weight.shape
    rot_mat_dim = rot_mat.shape[0]
    return rotation_function(weight.view(-1, weight_shape.shape[1] // rot_mat_dim,
                                         rot_mat_dim)).view(weight_shape)


class RotationWeightParametrization(torch.nn.Module):

    def __init__(
        self,
        rot_mat: torch.nn.Parameter,
        rot_func: Callable,
        input_axis: Optional[int] = None,
        output_axis: Optional[int] = None,
        is_source: bool = False,
        is_sink: bool = False,
        is_orphan: bool = False,
    ) -> None:
        super().__init__()
        self.rot_mat = rot_mat
        self.rot_func = rot_func
        self.input_axis = input_axis
        self.output_axis = output_axis
        self.is_source = is_source
        self.is_sink = is_sink
        self.is_orphan = is_orphan
        self.K = None

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.is_sink or self.is_orphan:
            if self.input_axis == 1:
                weight = self.rot_func(weight, self.rot_mat, self.K)
            elif self.input_axis == 0:
                weight = self.rot_func(weight.t(), self.rot_mat, self.K).t()
            else:
                raise RuntimeError("Not supported yet")

        if self.is_source:
            if self.output_axis == 0:
                weight = self.rot_func(weight.t(), self.rot_mat, self.K).t()
            elif self.output_axis == 1:
                weight = self.rot_func(weight, self.rot_mat, self.K)
            else:
                raise RuntimeError("Not supported yet")

        return weight


class RotationBiasParametrization(torch.nn.Module):

    def __init__(
        self,
        rot_mat: torch.nn.Parameter,
        rot_func: Callable,
        input_axis: Optional[int] = None,
        output_axis: Optional[int] = None,
        is_source: bool = False,
        is_sink: bool = False,
        is_orphan: bool = False,
    ) -> None:
        super().__init__()
        self.rot_mat = rot_mat
        self.rot_func = rot_func
        self.input_axis = input_axis
        self.output_axis = output_axis
        self.is_source = is_source
        self.is_sink = is_sink
        self.is_orphan = is_orphan
        self.K = None

    def forward(self, bias: torch.Tensor) -> torch.Tensor:
        if self.is_source:
            bias = self.rot_func(bias, self.rot_mat, self.K)

        return bias


def functional_rotate_input(inp, transpose=False):
    is_cuda = 'cuda' in str(inp.device) and torch.version.cuda is not None
    if transpose:
        inp = inp.t()
    if is_cuda and fast_hadamard_transform is not None:
        had_K, K = get_hadK(inp.shape[-1])
        inp = matmul_hadU_cuda(inp, had_K, K)
    else:
        inp = matmul_hadU(inp)

    if transpose:
        inp = inp.t()
    return inp
