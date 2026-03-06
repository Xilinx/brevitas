# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import contextmanager
from dataclasses import is_dataclass
from enum import Enum
import functools
import json
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union
import warnings

# Regular expressions for checking if a string can be converted to an int or a float.
INT_RE = re.compile(r"^[+-]?\d+$")
FLOAT_RE = re.compile(
    r"""^[+-]?(
            (?:\d+\.\d*)|   # 1. or 1.23
            (?:\.\d+)|      # .23
            (?:\d+)         # 1
        )
        (?:[eE][+-]?\d+)?$
    """,
    re.VERBOSE,
)


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


def convert_str_dict(passed_value: Dict) -> Dict:
    "Safely checks that a passed value is a dictionary and converts any string values to their appropriate types."
    for key, value in passed_value.items():
        if isinstance(value, dict):
            passed_value[key] = convert_str_dict(value)
        elif isinstance(value, str):
            # First check for bool and convert
            if value.lower() in ("true", "false"):
                passed_value[key] = value.lower() == "true"
            # Check for digit
            elif INT_RE.match(value):
                passed_value[key] = int(value)
            elif FLOAT_RE.match(value):
                passed_value[key] = float(value)

    return passed_value


def parse_dataclass_dicts(data_cls: Any, dict_attributes: List[str]) -> None:
    """
    Parses the strings in `dict_attributes` of dataclass `data_cls` to dictionaries.
    """
    assert is_dataclass(data_cls), f"data_cls must be a dataclass, but got {type(data_cls)}"
    for attr in dict_attributes:
        if not hasattr(data_cls, attr):
            raise ValueError(f"Dataclass {type(data_cls).__name__} has no attribute named {attr}")
        kwargs = getattr(data_cls, attr)

        if kwargs is None:
            kwargs = {}
        elif isinstance(kwargs, str):
            # Parse in args that could be `dict` sent in from the CLI as a string
            kwargs = json.loads(kwargs)
            # Convert str values to types if applicable
            kwargs = convert_str_dict(kwargs)
        elif isinstance(kwargs, dict):
            pass
        else:
            # Raise an error if the attribute cannot be parsed into a dictionary
            raise ValueError(
                f"Value set for attribute {attr} of dataclass {type(data_cls).__name__} cannot be converted into a dictionary."
            )
        # Set the updated value
        setattr(data_cls, attr, kwargs)


T = TypeVar("T")


class Registry(Generic[T]):

    def __init__(self, registry_name: Optional[str] = None) -> None:
        self._registry_name: str = registry_name
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
            # The value is registered under each name in `names`.
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
                f"'{name}' not found in {self.registry_name}. The registered keys are: {available}")
