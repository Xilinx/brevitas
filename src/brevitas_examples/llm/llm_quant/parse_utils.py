# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
from pathlib import Path

from brevitas.utils.logging import setup_logger
from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY

logging = setup_logger(__name__)


def _load_plugin(path: str, module_name: str, file_kind: str) -> None:
    """Validate and import a plugin file by path.

    The plugin file is expected to register entries into the relevant
    registry as a side-effect of being imported.
    """
    if not Path(path).expanduser().exists():
        raise FileNotFoundError(f"{file_kind} file path {path} does not exist.")
    if not path.endswith(".py"):
        raise ValueError(f"{path} is not a .py file.")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {file_kind.lower()} path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def parse_custom_quantizer(quant_name: str) -> str:
    """Load a custom quantizer plugin and return the quantizer name.

    The *quant_name* format is ``path/to/plugin.py:quant_name``.
    The plugin file is expected to register entries into the
    ``QUANTIZERS_REGISTRY`` as a side-effect of being imported.
    """
    # Detect "/path/to/plugin.py:quant_name"
    if ":" not in quant_name:
        return quant_name

    quant_path, quant_name = quant_name.rsplit(":", 1)

    # Retrieve previously registered quantizers
    pre_registered_quantizers = set(QUANTIZERS_REGISTRY.get_registered_keys())
    # Load the module with the custom quantizers
    _load_plugin(quant_path, "custom_quant", "Quantizer")
    # Retrieve newly registered quantizers
    post_registered_quantizers = set(QUANTIZERS_REGISTRY.get_registered_keys())

    logging.debug(
        f"The following quantizers were loaded from {quant_path}: {', '.join(post_registered_quantizers - pre_registered_quantizers)}"
    )

    return quant_name


def parse_custom_trainer(plugin_spec: str) -> str:
    """Load a custom rotation-training plugin and return the config name.

    The *plugin_spec* format is ``path/to/plugin.py:config_name``. When no
    plugin path is provided (i.e. the spec contains no ``:``), the spec is
    treated as a bare config name and returned unchanged.
    The plugin file is expected to register a ``TrainerSetup`` into the
    ``TRAINER_SETUP_REGISTRY`` as a side-effect of being imported.

    Returns the *config_name* portion so the caller can look up the
    registered values by name.
    """
    # Detect "/path/to/plugin.py:config_name"
    if ":" not in plugin_spec:
        return plugin_spec

    path, config_name = plugin_spec.rsplit(":", 1)

    _load_plugin(path, "custom_trainer", "Training plugin")

    logging.debug(f"Training plugin loaded from {path} with config name '{config_name}'")

    return config_name
