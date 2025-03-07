# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import contextmanager
from enum import Enum
import functools
from typing import List


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


def recurse_getattr(obj, attr: str):
    """
    Recursive `getattr`.

    Args:
        obj:
            A class instance holding the attribute.
        attr (`str`):
            The attribute that is to be retrieved, e.g. 'attribute1.attribute2'.
    """

    def _getattr(obj, attr):
        return getattr(obj, attr)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def hooked_on_a_function(function, prefunction):

    @functools.wraps(function)
    def run(*args, **kwargs):
        prefunction(*args, **kwargs)
        return function(*args, **kwargs)

    return run


def longest_common_prefix(strings: List[str]):
    """
    Finds the longest common prefix of a list of strings.

    Args:
        strings (list): A list of strings.

    Returns:
        The longest common prefix of the strings.
    """
    # If the list of strings is empty, return an empty string.
    if len(strings) == 0:
        return ""

    # Find the shortest string in the list.
    shortest_string = min(strings, key=len)

    # Iterate over the characters in the shortest string.
    for i, char in enumerate(shortest_string):
        # If the character is not the same in all the strings, return the prefix up to the current character.
        if any(string[i] != char for string in strings):
            return shortest_string[:i]

    # If all the characters in the shortest string are the same in all the strings, return the shortest string.
    return shortest_string
