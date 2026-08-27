#!/usr/bin/env python3

"""Inspect quantizers that (incorrectly) declare ``proxy_class`` twice.

Several quantizers in ``brevitas_examples/common/generative/quantizers.py`` assign
``proxy_class`` twice inside the same class body. In Python the second assignment
silently wins, so the first ``proxy_class`` is dead code and the quantizer ends up
using a different proxy than the one listed first.

This script, for every affected quantizer:
  1. Statically finds the duplicate ``proxy_class`` assignments (via AST) and prints
     both the "shadowed" (first) and the "effective" (last) proxy class.
  2. Confirms at runtime which ``proxy_class`` the class object actually exposes.
  3. Instantiates a layer quantized to 3 bits with the *effective* proxy and another
     with the *shadowed* (intended-first) proxy, feeds the same input through both,
     and reports the numeric difference between the two quantized outputs.

No environment setup is performed here -- run it in an environment where ``brevitas``
and its dependencies are importable:

    PYTHONPATH=src python check_duplicate_proxy_class.py
"""

import ast
import importlib
import os

import torch

QUANTIZERS_REL_PATH = os.path.join(
    "src", "brevitas_examples", "common", "generative", "quantizers.py")
QUANTIZERS_MODULE = "brevitas_examples.common.generative.quantizers"

# Fixed input so the run is reproducible.
torch.manual_seed(0)
GROUP_SIZE = 4
GROUP_DIM = -1
BIT_WIDTH = 3


def find_duplicate_proxy_classes(path):
    """Return {class_name: [(lineno, proxy_expr_src), ...]} for classes with >1 proxy_class."""
    with open(path) as f:
        tree = ast.parse(f.read())

    result = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        assigns = []
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "proxy_class":
                        assigns.append((stmt.lineno, ast.unparse(stmt.value)))
        if len(assigns) > 1:
            result[node.name] = assigns
    return result


def resolve_proxy(module, expr_src):
    """Resolve a proxy_class expression (a bare name) to the actual class object."""
    return getattr(module, expr_src, None)


def build_quant_identity(act_quant_cls, proxy_override=None):
    """Build a QuantIdentity using act_quant_cls, optionally overriding proxy_class.

    These are per-group *float* activation quantizers, so we drive them through a
    QuantIdentity with group_size / group_dim, quantized to BIT_WIDTH bits.
    """
    from brevitas.nn import QuantIdentity

    kwargs = dict(
        act_quant=act_quant_cls,
        bit_width=BIT_WIDTH,
        group_size=GROUP_SIZE,
        group_dim=GROUP_DIM,
        return_quant_tensor=False,
    )
    if proxy_override is not None:
        kwargs["proxy_class"] = proxy_override
    return QuantIdentity(**kwargs)


def dequantized_output(module, x):
    """Run x through module and return a plain dequantized tensor."""
    module.eval()
    out = module(x)
    # Depending on return type, extract the dequantized value.
    value = getattr(out, "value", out)
    return value


def main():
    quant_module = importlib.import_module(QUANTIZERS_MODULE)
    duplicates = find_duplicate_proxy_classes(QUANTIZERS_REL_PATH)

    if not duplicates:
        print("No quantizers with duplicate proxy_class found.")
        return

    print(f"Found {len(duplicates)} quantizer(s) with duplicate proxy_class:\n")

    x = torch.randn(2, 8)

    for name, assigns in duplicates.items():
        shadowed_line, shadowed_expr = assigns[0]
        effective_line, effective_expr = assigns[-1]

        cls = getattr(quant_module, name)
        actual = getattr(cls, "proxy_class", None)
        actual_name = getattr(actual, "__name__", repr(actual))

        print("=" * 72)
        print(f"Quantizer: {name}")
        print("-" * 72)
        print(f"  shadowed (line {shadowed_line}, ignored): {shadowed_expr}")
        print(f"  effective (line {effective_line}, wins)  : {effective_expr}")
        print(f"  runtime proxy_class actually picked up   : {actual_name}")
        if actual_name != effective_expr:
            print("  NOTE: runtime value differs from static last assignment!")

        shadowed_proxy = resolve_proxy(quant_module, shadowed_expr)
        effective_proxy = resolve_proxy(quant_module, effective_expr)

        print(
            "\n  Quantizing two layers to "
            f"{BIT_WIDTH} bits (group_size={GROUP_SIZE}, group_dim={GROUP_DIM})...")
        try:
            eff_mod = build_quant_identity(cls, proxy_override=effective_proxy)
            eff_out = dequantized_output(eff_mod, x)

            shad_mod = build_quant_identity(cls, proxy_override=shadowed_proxy)
            shad_out = dequantized_output(shad_mod, x)

            diff = (eff_out - shad_out).abs()
            denom = eff_out.abs().clamp_min(1e-12)
            rel = (diff / denom)

            print(f"    effective proxy : {getattr(effective_proxy, '__name__', effective_proxy)}")
            print(f"    shadowed proxy  : {getattr(shadowed_proxy, '__name__', shadowed_proxy)}")
            print(f"    max abs diff    : {diff.max().item():.6e}")
            print(f"    mean abs diff   : {diff.mean().item():.6e}")
            print(f"    max rel diff    : {rel.max().item():.6e}")
            if diff.max().item() == 0.0:
                print("    -> Outputs identical: the two proxies behave the same here.")
            else:
                print("    -> Outputs differ: the shadowed proxy_class would change results.")
        except Exception as e:  # noqa: BLE001
            print(f"    Could not run quantization comparison: {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    main()
