# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch


# A class to mock torch.library.custom_op, to avoid AttributeErrors in older versions of PyTorch (<2.4)
# TODO: Remove when PyTorch<2.4 deprecated
def _fake_custom_op(*dec_args, **dec_kwargs):

    def _decorator(fn):

        class FakeCustomOp(torch.nn.Module):

            @staticmethod
            def __call__(*args, **kwargs):
                return fn(*args, **kwargs)

            @staticmethod
            def register_fake(*args, **kwargs):

                def register_fake_decorator(fake_func):
                    return fake_func(*args, **kwargs)

                return register_fake_decorator

        return FakeCustomOp()

    return _decorator


try:
    from torch.library import custom_op as _custom_op
except:
    _custom_op = _fake_custom_op

custom_op = _custom_op
