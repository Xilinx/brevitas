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


class UnfusedRotatedModule(torch.nn.Module):

    def __init__(
        self,
        module: torch.nn.Module,
        rot_func: Callable,
        rot_mat: torch.nn.Parameter,
        _get_input_axis: Callable,
        _get_output_axis: Callable,
        is_source: bool = False,
        is_sink: bool = False,
        is_orphan: bool = False,
    ) -> None:
        super().__init__()
        self.module = module
        self.rot_func = rot_func
        self.rot_mat = rot_mat
        self.K = None

        # TODO: This were included to prevent circular imports.
        self._get_input_axis = _get_input_axis
        self._get_output_axis = _get_output_axis

        self.is_source = is_source
        self.is_sink = is_sink
        self.is_orphan = is_orphan

    # TODO: Does it make sense the extra complexity just to prevent the view operation?
    # Probably if no reshaping needs to be done, no change is required
    def _wrap_rot(self) -> bool:
        weight_shape = self.module.weight.shape
        rot_dim = self.rot_mat.shape[0]
        if self.is_sink or self.is_orphan:
            weight_shape_dim = weight_shape[self._get_input_axis(self.module)]
        elif self.is_source:
            weight_shape_dim = weight_shape[self._get_output_axis(self.module)]
        else:
            weight_shape_dim = None

        if weight_shape_dim is not None:
            if rot_dim != weight_shape_dim:
                assert weight_shape_dim % rot_dim == 0, "Sizes need to be divisibile"
                return True
        # No need to incorporate additional view operations
        return False

    # These properties enable propagating the fusing to the module weights
    @property
    def weight(self) -> Optional[torch.Tensor]:
        weight = getattr(self.module, 'weight', None)
        # Add rotation and let these being propagated till the parent
        # unfused rotated module
        if self.is_sink or self.is_orphan:
            axis = self._get_input_axis(self.module)
            if axis == 1:
                weight = self.rot_func(weight, self.rot_mat, self.K)
            elif axis == 0:
                weight = self.rot_func(weight.t(), self.rot_mat, self.K).t()
            else:
                raise RuntimeError("Not supported yet")

        if self.is_source:
            axis = self._get_output_axis(self.module)
            if axis == 0:
                weight = self.rot_func(weight.t(), self.rot_mat, self.K).t()
            elif axis == 1:
                weight = self.rot_func(weight, self.rot_mat, self.K)
            else:
                raise RuntimeError("Not supported yet")

        return weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        bias = getattr(self.module, 'bias', None)
        # Propagate bias adding the rotations incrementally
        if self.is_source:
            if bias is not None:
                bias = self.rot_func(bias, self.rot_mat, self.K)

        return bias

    @property
    def unrotated_module(self) -> torch.nn.Module:
        return self.module.unrotated_module if isinstance(
            self.module, UnfusedRotatedModule) else self.module

    def forward(self, inp, **kwargs):
        # Rotated matrices
        weight = self.weight.data
        bias = self.bias.data if self.bias is not None else None

        # Propagate calls till getting to the original module being rotated
        child_module = self.module
        # Iterate until the original module is reached, keeping the rotations that need to be performed on the input
        while isinstance(child_module, UnfusedRotatedModule):
            child_module = child_module.module
        # child_module contains the original module in the network. Before applying its forward method, we need to
        # rotate the inpute appropiately
        if self.is_orphan:
            # Rotate the input for an orphan sink
            inp = self.rot_func(inp, self.rot_mat, self.K)
        # Modify the weights, and run the original model forward. After that, restore the previous values.
        if weight is not None:
            orig_weight = child_module.weight.data
            child_module.weight.data = weight
        if bias is not None:
            orig_bias = child_module.bias.data
            child_module.bias.data = bias
        # Call forward of the original module
        o = child_module(inp)
        # Restore un-rotated weights
        child_module.weight.data = orig_weight
        if bias is not None:
            child_module.bias.data = orig_bias
        # Return rotated output
        return o


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
