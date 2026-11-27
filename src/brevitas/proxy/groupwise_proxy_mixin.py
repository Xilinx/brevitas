# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch import nn

from brevitas.inject import BaseInjector as Injector


class GroupwiseProxyMixin:
    """
    Group settings shared by the groupwise proxies.

    ``group_dim`` and ``group_size`` are read from the injector once, at construction, instead of
    at every call, so that torch.compile does not have to trace the injector. The mixin has to
    precede the proxy class in the bases, since it extends its ``__init__`` and
    ``apply_input_view``.
    """

    def __init__(self, quant_layer: nn.Module, quant_injector: Injector) -> None:
        super().__init__(quant_layer, quant_injector)
        self._group_dim = self.quant_injector.group_dim
        self._group_size = self.quant_injector.group_size
        if self._group_dim is None or self._group_size is None:
            raise RuntimeError(
                "Groupwise quantization requires a resolved group_dim and group_size")

    @property
    def group_dim(self):
        return self._group_dim

    @property
    def group_size(self):
        return self._group_size

    def apply_input_view(self, x):
        x = super().apply_input_view(x)
        start_dim = self.group_dim if self.group_dim >= 0 else self.group_dim - 1
        return x.flatten(start_dim, start_dim + 1)
