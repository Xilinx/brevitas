import torch

from brevitas.nn.quant_mha import QuantMultiheadAttention


class EqualizedModule(torch.nn.Module):

    def __init__(self, scale_module, layer) -> None:
        super().__init__()
        self.scale = scale_module
        self.layer = layer

    def forward(self, *args, **kwargs):
        args = list(args)

        if len(args) > 0:
            x = args[0]
            # We delete it since it will updated and passed as first arg
            args.pop(0)
        elif len(kwargs) > 0 and 'query' in kwargs:
            x = kwargs['query']
            # We delete it since it will updated and passed as first arg
            del kwargs['query']
        else:
            raise ValueError("Unsupported input type")

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
            if 'key' not in kwargs.keys():
                pos_inputs.append(out)
                args.pop(0)
            else:
                kwargs['key'] = out
            if 'value' not in kwargs.keys():
                pos_inputs.append(out)
                args.pop(0)
            else:
                kwargs['value'] = out
        out = self.layer(*(pos_inputs + args), **kwargs)
        return out
