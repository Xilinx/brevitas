
import torch

try:
    import torch.library.custom_op as _custom_op
except:
    _custom_op = _fake_custom_op

custom_op = _custom_op

def _fake_custom_op(fn):
    def _default(*args, **kwargs):
        def register_fake(*args, **kwargs):
            pass
        fn(*args, **kwargs)
    return _default
