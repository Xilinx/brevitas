# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
import importlib
from pathlib import Path

from brevitas.utils.logging import setup_logger
from brevitas.utils.python_utils import Registry
from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY
from brevitas_examples.llm.llm_quant.trainer_utils import TRAINER_REGISTRY

logging = setup_logger(__name__)


def _import_plugin_module(path: str, module_name: str) -> None:
    """Validate and import a plugin ``.py`` file by path.

    The plugin file is expected to register entries into the relevant
    registry as a side-effect of being imported.
    """
    if not Path(path).expanduser().exists():
        raise FileNotFoundError(f"Plugin file path {path} does not exist.")
    if not path.endswith(".py"):
        raise ValueError(f"{path} is not a .py file.")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for plugin path: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _load_plugin(name: str, registry: Registry, module_name: str = "custom_plugin") -> str:
    """Load a custom plugin and return its registered name.

    The *name* format is ``path/to/plugin.py:name``. When no plugin path
    is provided (i.e. *name* contains no ``:``), it is treated as a bare
    name and returned unchanged.

    The plugin file is expected to register entries into *registry* as a
    side-effect of being imported. The *name* portion is returned so the
    caller can look up the registered values by name.
    """
    # Detect "/path/to/plugin.py:name"
    if ":" not in name:
        return name

    path, name = name.rsplit(":", 1)

    # Snapshot the registry keys before/after import to report what the
    # plugin registered.
    pre_registered = set(registry.get_registered_keys())
    _import_plugin_module(path, module_name)
    post_registered = set(registry.get_registered_keys())

    logging.debug(
        f"The following entries were registered into {registry.registry_name} "
        f"from {path}: {', '.join(post_registered - pre_registered)}")

    return name


parse_custom_quantizer = partial(
    _load_plugin, registry=QUANTIZERS_REGISTRY, module_name="custom_quant")
parse_custom_trainer = partial(
    _load_plugin, registry=TRAINER_REGISTRY, module_name="custom_trainer")
