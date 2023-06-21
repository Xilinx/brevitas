import torch

from brevitas.nn.quant_mha import QuantMultiheadAttention


class EqualizedModule(torch.nn.Module):

    def __init__(self, scale_module, layer) -> None:
        super().__init__()
        self.scale = scale_module
        self.layer = layer

    def forward(self, x, *args, **kwargs):
        args = list(args)
        out = x
        if 'key' in kwargs:
            if kwargs['key'].data_ptr() != out.data_ptr():
                raise ValueError(
                    "Cross MHA is not supported for activation equalization."
                    "Replace kwargs with positional args to avoid this exception.")
        out = self.scale(out)

        pos_inputs = [out]
        # QuantMultiheadAttention is not a subclass of MultiheadAttention
        # We need to preserve the correctness of the forward even after
        # quantization has been applied
        if isinstance(self.layer, (torch.nn.MultiheadAttention, QuantMultiheadAttention)):
            if 'key' not in kwargs.items():
                pos_inputs.append(out)
                args.pop(0)
            else:
                kwargs['key'] = out
            if 'value' not in kwargs.items():
                pos_inputs.append(out)
                args.pop(0)
            else:
                kwargs['value'] = out

        out = self.layer(*pos_inputs, *args, **kwargs)
        return out
