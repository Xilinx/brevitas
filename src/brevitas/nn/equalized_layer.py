from inspect import signature

import torch

from brevitas.graph.hadamard import find_closest_hadamard_number
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


class RotatedModule(torch.nn.Module):

    def __init__(self, layer, had_mat=None, k=None, expand=False) -> None:
        super().__init__()
        if had_mat is not None:
            self.had_mat = torch.nn.Parameter(had_mat).cpu()
        else:
            self.had_mat = None
        self.layer = layer
        self.k = k
        self.expand = expand

    def input_pad(self, inp):
        # TODO: This only works for Linear layers. We have an assert in equalize.py to check for this
        hidden_dim = inp.shape[-1]
        new_hidden = find_closest_hadamard_number(hidden_dim, hidden_dim + 1)
        pad = new_hidden - hidden_dim
        pad_dim = len(inp.shape) * 2
        pad_tensor = [0] * pad_dim
        # Right padding the feature dimension
        pad_tensor[1] = int(pad)
        inp = torch.nn.functional.pad(inp, pad_tensor, mode='constant', value=0)
        return inp

    def forward(self, inp, **kwargs):
        is_cuda = 'cuda' in str(inp.device) and torch.version.cuda is not None
        if self.expand:
            inp = self.input_pad(inp)

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


def functional_rotate_input(inp, transpose=False):
    is_cuda = 'cuda' in str(inp.device) and torch.version.cuda is not None
    if transpose:
        inp = inp.transpose(-2, -1)
    if is_cuda and fast_hadamard_transform is not None:
        had_K, K = get_hadK(inp.shape[-1])
        inp = matmul_hadU_cuda(inp, had_K, K)
    else:
        inp = matmul_hadU(inp)

    if transpose:
        inp = inp.transpose(-2, -1)
    return inp
