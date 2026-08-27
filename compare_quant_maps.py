#!/usr/bin/env python3

"""Compare WEIGHT_QUANT_MAP and INPUT_QUANT_MAP defined in different parts of the codebase.

The two maps are nested dictionaries whose leaves are quantizer classes. This script
imports each map from its module, flattens it into a set of `key.path -> ClassName`
entries, and reports:
  * key paths present in one map but not the other
  * key paths present in both but mapping to a different quantizer class

Usage:
    python compare_quant_maps.py            # compare both maps across the two modules
    python compare_quant_maps.py -v         # also print the fully-shared entries
"""

import argparse
import importlib

# (label, module path, attribute name) for every map we want to compare.
# To compare additional locations, just add more entries here.
SOURCES = {
    "WEIGHT_QUANT_MAP": [
        ("ptq_common", "brevitas_examples.imagenet_classification.ptq.ptq_common"),
        ("generative", "brevitas_examples.common.generative.quantize"),],
    "INPUT_QUANT_MAP": [
        ("ptq_common", "brevitas_examples.imagenet_classification.ptq.ptq_common"),
        ("generative", "brevitas_examples.common.generative.quantize"),],}


def flatten(d, prefix=()):
    """Flatten a nested dict into {('k1', 'k2', ...): leaf_value}."""
    flat = {}
    for key, value in d.items():
        path = prefix + (key,)
        if isinstance(value, dict):
            flat.update(flatten(value, path))
        else:
            flat[path] = value
    return flat


def leaf_name(value):
    """Human-readable name for a leaf (a quantizer class, usually)."""
    return getattr(value, "__name__", repr(value))


def load_map(module_path, attr_name):
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def compare(map_name, sources, verbose=False):
    print("=" * 70)
    print(f"Comparing {map_name}")
    print("=" * 70)

    flattened = {}
    for label, module_path in sources:
        raw = load_map(module_path, map_name)
        flattened[label] = flatten(raw)
        print(f"  [{label}] {module_path}: {len(flattened[label])} leaf entries")
    print()

    labels = list(flattened.keys())
    # Pairwise comparison (handles 2+ sources gracefully).
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            fa, fb = flattened[a], flattened[b]
            keys_a, keys_b = set(fa), set(fb)

            only_a = sorted(keys_a - keys_b)
            only_b = sorted(keys_b - keys_a)
            common = keys_a & keys_b
            differing = sorted(k for k in common if leaf_name(fa[k]) != leaf_name(fb[k]))
            same = sorted(k for k in common if leaf_name(fa[k]) == leaf_name(fb[k]))

            print(f"--- {a}  vs  {b} ---")

            if only_a:
                print(f"\n  Only in [{a}] ({len(only_a)}):")
                for k in only_a:
                    print(f"    {'.'.join(k)}  ->  {leaf_name(fa[k])}")

            if only_b:
                print(f"\n  Only in [{b}] ({len(only_b)}):")
                for k in only_b:
                    print(f"    {'.'.join(k)}  ->  {leaf_name(fb[k])}")

            if differing:
                print(f"\n  Same key, different quantizer ({len(differing)}):")
                for k in differing:
                    print(
                        f"    {'.'.join(k)}\n"
                        f"        [{a}] {leaf_name(fa[k])}\n"
                        f"        [{b}] {leaf_name(fb[k])}")

            if not only_a and not only_b and not differing:
                print("\n  Maps are identical.")

            if verbose and same:
                print(f"\n  Identical in both ({len(same)}):")
                for k in same:
                    print(f"    {'.'.join(k)}  ->  {leaf_name(fa[k])}")

            print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="also list entries that are identical in both maps")
    parser.add_argument(
        "--map",
        choices=sorted(SOURCES),
        action="append",
        help="restrict comparison to this map (repeatable); default: all")
    args = parser.parse_args()

    selected = args.map or sorted(SOURCES)
    for map_name in selected:
        compare(map_name, SOURCES[map_name], verbose=args.verbose)


if __name__ == "__main__":
    main()
