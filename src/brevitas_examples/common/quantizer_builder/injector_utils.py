"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
from enum import Enum
import inspect
from typing import Any
from typing import Dict
from typing import Tuple

from _dependencies.exceptions import DependencyError


def _short(value: Any) -> str:
    """Compact, single-line representation of a resolved dependency."""
    if inspect.isclass(value):
        return value.__name__
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _this_expression(attribute: Any) -> str:
    """Reconstruct a readable form of a `this` expression, e.g. (this<<1).device."""
    shifts = 0
    parts = []
    for kind, symbol in attribute.__expression__:
        if kind == "." and symbol == "__parent__":
            shifts += 1
        elif kind == ".":
            parts.append(f".{symbol}")
        else:  # "[]"
            parts.append(f"[{symbol!r}]")
    base = f"(this<<{shifts})" if shifts else "this"
    return base + "".join(parts)


def _resolve(injector: type, name: str) -> str:
    """Resolve `name` on the injector, returning a compact value or `<unresolved>`."""
    try:
        return _short(getattr(injector, name))
    except DependencyError as e:
        return f"<unresolved: {e}>"
    except Exception as e:  # resolution can trigger arbitrary construction errors
        return f"<unresolved: {type(e).__name__}: {e}>"


def _format_args(args: list, have_defaults: int) -> str:
    """Render required/optional args, e.g. (a, b, [c, d]) where c, d are optional."""
    required = args[:have_defaults]
    optional = args[have_defaults:]
    rendered = list(required)
    if optional:
        rendered.append("[" + ", ".join(optional) + "]")
    return "(" + ", ".join(rendered) + ")"


def format_contribution(bases: Tuple[type, ...], attrs: Dict[str, Any]) -> str:
    """Render the merged component output of a builder: the ``bases`` (MRO order)
    and the assembled namespace ``attrs`` (compact, sorted)."""
    lines = ["Base classes:"]
    lines.extend(f"  - {base.__name__}" for base in bases)
    lines.append("Attributes:")
    width = max((len(name) for name in attrs), default=0)
    lines.extend(f"  {name:<{width}} = {_short(attrs[name])}" for name in sorted(attrs))
    return "\n".join(lines)


def describe_injector(injector: type, resolve: bool = True, depth: int = 0) -> None:
    """Print each attribute of a Brevitas injector, its dependency kind and value.

    For ``@value`` functions the required (and optional) arguments are shown
    along with what the function resolves to. ``this`` expressions, init-injected
    classes, raw constants and nested injectors are each labelled accordingly.

    Args:
        injector: an injector class (e.g. the output of ``build_quant_injector``).
        resolve: if True, attempt to resolve each attribute and print its value.
        depth: how many levels of nested injectors to recurse into (0 = none).
    """
    deps = injector.__dependencies__
    width = max((len(n) for n in deps), default=0)
    for name in sorted(deps):
        if name.startswith("__"):
            continue
        marker, attribute, args, have_defaults = deps[name]

        if marker == "nested_injector":
            nested = getattr(attribute, "injector", attribute)
            kind = f"<nested> {getattr(nested, '__name__', nested)}"
            print(f"{name:<{width}} = {kind}")
            if depth > 0 and hasattr(nested, "__dependencies__"):
                describe_injector(nested, resolve=resolve, depth=depth - 1)
            continue

        # Raw constants resolve to themselves, so the ` -> resolved` is redundant.
        show_resolved = resolve
        if marker == "this":
            kind = f"<this>  {_this_expression(attribute)}"
        elif inspect.isfunction(attribute):
            kind = f"<value>{_format_args(args, have_defaults)}"
        elif inspect.isclass(attribute):
            kind = f"<class> {attribute.__name__}{_format_args(args, have_defaults)}"
        elif hasattr(attribute, "dependency"):  # _RawSpec: a constant
            kind = f"<raw>   {_short(attribute.dependency)}"
            show_resolved = False
        else:
            kind = f"<{marker}> {_short(attribute)}"

        resolved = f" -> {_resolve(injector, name)}" if show_resolved else ""

        print(f"{name:<{width}} = {kind}{resolved}")
