from inspect import signature

import torch

from brevitas.nn.quant_mha import QuantMultiheadAttention

INPUT_NAMES = ['input', 'inp', 'query', 'x']


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
