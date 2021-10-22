from torch.autograd import Function

from . import DOMAIN_STRING


class QuantHardTanhFn(Function):

    @staticmethod
    def symbolic(g, input, qnt_type, thres, bias, scale):
        if qnt_type == "BIPOLAR":
            return g.op(
                f'{DOMAIN_STRING}::MultiThreshold', input, thres,
                out_dtype_s=qnt_type,
                out_scale_f=2.0,
                out_bias_f=-1.0)
        else:
            ret = g.op(
                f'{DOMAIN_STRING}::MultiThreshold', input, thres,
                out_dtype_s=qnt_type)
            if bias is not None:
                ret = g.op('Add', ret, bias)
            if scale is not None:
                ret = g.op('Mul', ret, scale)
            return ret

    @staticmethod
    def forward(ctx, input, qnt_type, thres, bias, scale):
        return input.clamp(0)


class QuantReLUFn(Function):

    @staticmethod
    def symbolic(g, input, qnt_type, thres, bias, scale):
        ret = g.op(f'{DOMAIN_STRING}::MultiThreshold', input, thres, out_dtype_s=qnt_type)
        if scale is not None:
            ret = g.op('Mul', ret, scale)
        return ret

    @staticmethod
    def forward(ctx, input, qnt_type, thres, bias, scale):
        return input.clamp(0)
