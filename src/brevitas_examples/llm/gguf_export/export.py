# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# This code was adapted from https://github.com/intel/auto-round, under the following LICENSE:
# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import re
import shutil
import tempfile
import time

import torch

from brevitas.utils.logging import setup_logger

# Imported for its side effect: registers the GGUF custom quantizers (gguf_q4_0,
# gguf_q4_k, ...) in QUANTIZERS_REGISTRY so they are selectable via --custom-quantizer.
from . import custom_quantizers  # noqa: F401
from .convert import ModelBase
from .proxy import GGUFGroupwiseWeightQuantProxyFromInjector
from .targets import FTYPE_MAP
from .targets import GGUF_EXPORT_TARGETS

logger = setup_logger(__name__)


def _gguf_proxies(model):
    """Yield every GGUF-aware weight-quant proxy reachable from ``model``."""
    for m in model.modules():
        if isinstance(m, GGUFGroupwiseWeightQuantProxyFromInjector):
            yield m


def _enable_gguf_export_caching(model):
    """Enable metadata-only weight caching on each GGUF proxy."""
    prior_settings = {}
    for wq in _gguf_proxies(model):
        prior_settings[wq] = (
            wq.cache_inference_quant_weight, wq.cache_inference_quant_weight_metadata_only)
        # NOTE: quant_inference_mode enables metadata-only weight caching by default.
        # It does not clear this cache on exit; clear against to avoid stale metadata.
        wq._cached_weight = None
        wq.cache_inference_quant_weight_metadata_only = True
        wq.cache_inference_quant_weight = True
    return prior_settings


def _restore_gguf_export_caching(prior_settings):
    """Restore the cache settings for each GGUF proxy."""
    for wq, (prior_enabled, prior_metadata_only) in prior_settings.items():
        wq.cache_inference_quant_weight_metadata_only = prior_metadata_only
        wq.cache_inference_quant_weight = prior_enabled


def _resolve_model_name(name_or_path: str) -> str:
    # For HF cache snapshot dirs, recover the model id from models--<org>--<model>
    # so general.name is human-readable rather than the snapshot SHA.
    m = re.search(r'models--([^/]+)/snapshots/[a-f0-9]{40}', name_or_path)
    if m:
        return m.group(1).split('--')[-1]
    parts = [p for p in name_or_path.split('/') if p]
    return parts[-1] if parts else name_or_path


def save_quantized_as_gguf(
        model,
        tokenizer,
        backend="gguf:q4_0",
        override_model_tensors=None,
        override_qtype=None,
        export_path=None):
    """Export the model to gguf format.

    When ``override_model_tensors``/``override_qtype`` are None, no tensor qtype is
    overridden at export time: every tensor follows the quantization it already
    has (or the file type otherwise).

    ``export_path`` controls where the ``.gguf`` file is written:

    * ``None`` (default): write to the current working directory, using gguf-py's
      auto-derived ``<name>-<size_label>-<ftype>.gguf`` naming.
    * a path ending in ``.gguf``: treated as the exact file to write (parent
      directories are created if needed).
    * any other path: treated as a directory to write into (created if needed),
      using the same auto-derived naming as the ``None`` case.
    """
    st = time.time()
    config = model.config

    # TODO: every tensor now carries its own qtype via
    # GGUFGroupwiseWeightQuantProxyFromInjector.gguf_qtype, so `ftype` (derived
    # from `backend` below) no longer determines how already-quantized tensors are
    # packed. It's still used for the `general.file_type` header, the fallback
    # qtype applied to any untagged tensor, and `{ftype}`-based auto-naming --
    # worth revisiting whether `ftype` can be simplified or dropped for those.
    assert backend in GGUF_EXPORT_TARGETS, f"{backend} is not supported"
    output_type = backend.split(":")[-1].lower()
    output_type = FTYPE_MAP.get(output_type)

    if export_path is None:
        fname_out = Path('.')
    elif str(export_path).endswith('.gguf'):
        fname_out = Path(export_path)
        fname_out.parent.mkdir(parents=True, exist_ok=True)
    else:
        fname_out = Path(export_path)
        fname_out.mkdir(parents=True, exist_ok=True)

    tmp_work_dir = Path(tempfile.mkdtemp(prefix='brevitas_gguf_export_'))
    is_training = model.training
    prior_gguf_cache_settings = dict()
    try:
        tokenizer.save_pretrained(tmp_work_dir)
        config.save_pretrained(tmp_work_dir)
        if getattr(model, 'generation_config', None) is not None:
            model.generation_config.save_pretrained(tmp_work_dir)

        with torch.no_grad():
            hparams = ModelBase.load_hparams(tmp_work_dir)
            model_architecture = hparams["architectures"][0]
            model_class = ModelBase.from_model_architecture(model_architecture)
            model_name = _resolve_model_name(model.name_or_path)

            model_instance = model_class(
                model,
                dir_model=tmp_work_dir,
                ftype=output_type,
                fname_out=fname_out,
                is_big_endian=False,
                model_name=model_name,
                split_max_tensors=False,
                split_max_size=0,
                dry_run=False,
                small_first_shard=False,
                override_model_tensors=override_model_tensors,
                override_qtype=override_qtype)

            model.eval()
            prior_gguf_cache_settings = _enable_gguf_export_caching(model)
            model_instance.write()
    # Restore model state and remove temporary files after success or failure.
    finally:
        _restore_gguf_export_caching(prior_gguf_cache_settings)
        model.train(is_training)
        shutil.rmtree(tmp_work_dir, ignore_errors=True)

    rt = time.time() - st
    logger.info(f"Model successfully exported to {model_instance.fname_out}, running time={rt}")

    return model
