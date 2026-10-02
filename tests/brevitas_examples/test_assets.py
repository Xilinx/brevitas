# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import os
from pathlib import Path
import shutil
import zipfile

from filelock import FileLock

from tests.brevitas_examples._cache import get_hf_cache_dir
from tests.brevitas_examples._cache import get_lock_dir
from tests.brevitas_examples._cache import get_nltk_cache_dir


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def resolve_hf_asset(repo_id: str, revision: str | None = None) -> str:
    """Resolve a Hub asset once and return a local, loadable snapshot path.

    Hugging Face Hub provides the interprocess lock and atomic cache publication.
    Tests consume the resulting local path so model loading itself cannot start
    another network request.
    """
    if os.path.isdir(repo_id):
        return repo_id

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    cache_dir = get_hf_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    common_args = {
        "repo_id": repo_id,
        "revision": revision,
        "cache_dir": str(cache_dir),}

    try:
        return snapshot_download(**common_args, local_files_only=True)
    except LocalEntryNotFoundError:
        if _env_enabled("BREVITAS_TEST_OFFLINE"):
            raise

    lock_id = hashlib.sha256(f"{repo_id}\n{revision}".encode()).hexdigest()
    lock_dir = get_lock_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_dir / f"hf-{lock_id}.lock"):
        try:
            return snapshot_download(**common_args, local_files_only=True)
        except LocalEntryNotFoundError:
            return snapshot_download(**common_args, local_files_only=False)


NLTK_RESOURCES = {
    'punkt': 'tokenizers/punkt',
    'punkt_tab': 'tokenizers/punkt_tab',
    'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
    'stopwords': 'corpora/stopwords',}

LIGHTEVAL_DATASETS = (
    ('allenai/ai2_arc', 'ARC-Challenge'),
    ('allenai/ai2_arc', 'ARC-Easy'),
    ('allenai/winogrande', 'winogrande_xl'),
    ('Rowan/hellaswag', 'default'),
)


def _nltk_resource_files(resource: str):
    resource_path = NLTK_RESOURCES[resource]
    yield get_nltk_cache_dir() / f'{resource_path}.zip'
    yield get_nltk_cache_dir() / resource_path


def _ensure_nltk_resource(nltk, resource: str) -> None:
    resource_path = NLTK_RESOURCES[resource]
    for attempt in range(3):
        try:
            nltk.data.find(resource_path)
            return
        except (LookupError, zipfile.BadZipFile) as exc:
            last_error = exc
            for path in _nltk_resource_files(resource):
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()

        if _env_enabled('BREVITAS_TEST_OFFLINE'):
            raise RuntimeError(f'Missing NLTK resource in offline mode: {resource}') from last_error
        try:
            downloaded = nltk.download(resource, download_dir=str(get_nltk_cache_dir()), quiet=True)
        except Exception as exc:
            last_error = exc
        else:
            if not downloaded:
                last_error = RuntimeError(f'Failed to download NLTK resource: {resource}')

    raise RuntimeError(f'Invalid NLTK resource after download: {resource}') from last_error


def prepare_lighteval_assets() -> None:
    """Prepare LightEval resources before pytest creates parallel workers."""
    import nltk

    nltk_dir = get_nltk_cache_dir()
    nltk_dir.mkdir(parents=True, exist_ok=True)
    os.environ['NLTK_DATA'] = str(nltk_dir)
    if str(nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir))

    lock_dir = get_lock_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_dir / 'lighteval-assets.lock'):
        for resource in ('punkt', 'punkt_tab'):
            _ensure_nltk_resource(nltk, resource)

        from datasets import load_dataset

        for repo_id, config in LIGHTEVAL_DATASETS:
            load_dataset(repo_id, config)
