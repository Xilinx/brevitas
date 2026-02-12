# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import contextmanager
from enum import Enum
import functools
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union
import warnings


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


T = TypeVar("T")


class Registry(Generic[T]):

    def __init__(self, registry_name: Optional[str] = None) -> None:
        self._registry_name = registry_name
        self._registry: Dict[str, T] = {}

    @staticmethod
    def register(
        registry: "Registry[T]",
        names: Union[str, List[str]],
    ) -> Callable[[T], T]:
        return registry.register(names)

    @property
    def registry_name(self) -> str:
        return "Registry" if self._registry_name is None else self._registry_name

    def register(self, names: Union[str, List[str]]) -> Callable[[T], T]:
        if isinstance(names, str):
            names = [names]

        def decorator(value: T) -> T:
            # Allow registering the same value to multiple keys
            for name in names:
                if name in self._registry:
                    warnings.warn(
                        f"'{name}' is already registered in {self.registry_name}. Overwriting the existing value."
                    )
                self._registry[name] = value
            return value

        return decorator

    def get_registered_keys(self) -> Iterable[str]:
        return self._registry.keys()

    def get(self, name: str) -> T:
        try:
            return self._registry[name]
        except KeyError:
            available = ", ".join(sorted(self._registry)) or "<empty>"
            raise ValueError(
                f"'{name}' not found in {self.registry_name}. The available values are: {available}"
            )
