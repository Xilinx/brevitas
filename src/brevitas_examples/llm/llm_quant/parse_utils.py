# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
from pathlib import Path

from brevitas.utils.logging import setup_logger
from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY
from brevitas_examples.llm.llm_quant.rotation_optimization import OPTIMIZER_CONFIG_REGISTRY
from brevitas_examples.llm.llm_quant.rotation_optimization import TRAINER_REGISTRY
from brevitas_examples.llm.llm_quant.rotation_optimization import TRAINING_ARGS_REGISTRY

logging = setup_logger(__name__)


def parse_custom_quantizer(quant_name: str) -> str:
    """Load a custom quantizer plugin and return the quantizer name.

    The *quant_name* format is ``path/to/plugin.py:quant_name``.
    The plugin file is expected to register entries into the
    ``QUANTIZERS_REGISTRY`` as a side-effect of being imported.
    """
    # Detect "/path/to/plugin.py:quant_name"
    quant_path = None
    if ":" in quant_name:
        path, name = quant_name.rsplit(":", 1)
        # Treat as a file plugin if paths points to an existing .py file
        if not Path(path).expanduser().exists():
            raise FileNotFoundError(f"Quantizer file path {path} does not exist.")
        if not path.endswith(".py"):
            raise ValueError(f"{path} is not a .py file.")
        quant_path = path
        quant_name = name

    if quant_path is not None:
        # Retrieve previously registered quantizers
        pre_registered_quantizers = set(QUANTIZERS_REGISTRY.get_registered_keys())
        # Load the module with the custom quantizers
        spec = importlib.util.spec_from_file_location("custom_quant", quant_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for quantizer path: {quant_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Retrieve newly registered quantizers
        post_registered_quantizers = set(QUANTIZERS_REGISTRY.get_registered_keys())

        logging.debug(
            f"The following quantizers were loaded from {quant_path}: {', '.join(post_registered_quantizers - pre_registered_quantizers)}"
        )

    return quant_name


def parse_custom_trainer(plugin_spec: str) -> str:
    """Load a custom rotation-training plugin and return the config name.

    The *plugin_spec* format is ``path/to/plugin.py:config_name``.
    The plugin file is expected to register entries into one or more of
    ``TRAINER_REGISTRY``, ``TRAINING_ARGS_REGISTRY``, and
    ``OPTIMIZER_CONFIG_REGISTRY`` as a side-effect of being imported.

    Returns the *config_name* portion so the caller can look up the
    registered values by name.
    """
    if ":" not in plugin_spec:
        raise ValueError(
            f"Invalid custom-trainer spec '{plugin_spec}'. "
            "Expected format: 'path/to/plugin.py:config_name'")

    path, config_name = plugin_spec.rsplit(":", 1)

    if not Path(path).expanduser().exists():
        raise FileNotFoundError(f"Training plugin file path {path} does not exist.")
    if not path.endswith(".py"):
        raise ValueError(f"{path} is not a .py file.")

    spec = importlib.util.spec_from_file_location("custom_trainer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for training plugin path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    logging.debug(f"Training plugin loaded from {path} with config name '{config_name}'")

    return config_name
