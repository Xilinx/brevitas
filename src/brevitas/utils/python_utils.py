# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import contextmanager
from enum import Enum


class AutoName(str, Enum):

    def _generate_next_value_(name, start, count, last_values):
        return name

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return str(self).lower() == str(other).lower()


@contextmanager
def patch(module, attr, new_value):
    """
    Context manager for monkey patching.
    Each patch can be used only once.
    with patch(torch, 'add', custom_add):
        ...
    """
    old_value = getattr(module, attr)
    setattr(module, attr, new_value)
    try:
        yield getattr(module, attr)
    finally:
        setattr(module, attr, old_value)


def islambda(v):
    LAMBDA = lambda: 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__
