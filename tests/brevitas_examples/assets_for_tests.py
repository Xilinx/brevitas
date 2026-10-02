# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import os
import shutil
import zipfile

from filelock import FileLock

TEST_CACHE_VERSION = '1'


def get_test_cache_dir() -> str:
    cache_dir = os.environ.get('BREVITAS_TEST_CACHE_DIR')
    if cache_dir:
        return cache_dir
    return os.path.join(os.getcwd(), 'data')


def get_lock_dir() -> str:
    return os.path.join(get_test_cache_dir(), '.locks')


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

    cache_dir = os.path.join(get_test_cache_dir(), 'huggingface', 'hub')
    os.makedirs(cache_dir, exist_ok=True)
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
    os.makedirs(lock_dir, exist_ok=True)
    with FileLock(os.path.join(lock_dir, f"hf-{lock_id}.lock")):
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


def prepare_lighteval_assets() -> None:
    """Prepare LightEval resources before pytest creates parallel workers."""
    import nltk

    nltk_dir = os.path.join(get_test_cache_dir(), 'nltk')
    os.makedirs(nltk_dir, exist_ok=True)
    os.environ['NLTK_DATA'] = nltk_dir
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)

    lock_dir = get_lock_dir()
    os.makedirs(lock_dir, exist_ok=True)
    with FileLock(os.path.join(lock_dir, 'lighteval-assets.lock')):
        for resource in ('punkt', 'punkt_tab'):
            resource_path = NLTK_RESOURCES[resource]
            for attempt in range(3):
                try:
                    nltk.data.find(resource_path)
                    break
                except (LookupError, zipfile.BadZipFile) as exc:
                    last_error = exc
                    for path in (os.path.join(nltk_dir, f'{resource_path}.zip'),
                                 os.path.join(nltk_dir, resource_path)):
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        elif os.path.exists(path):
                            os.remove(path)
                if _env_enabled('BREVITAS_TEST_OFFLINE'):
                    raise RuntimeError(
                        f'Missing NLTK resource in offline mode: {resource}') from last_error
                try:
                    downloaded = nltk.download(resource, download_dir=str(nltk_dir), quiet=True)
                except Exception as exc:
                    last_error = exc
                else:
                    if not downloaded:
                        last_error = RuntimeError(f'Failed to download NLTK resource: {resource}')
            else:
                raise RuntimeError(
                    f'Invalid NLTK resource after download: {resource}') from last_error

        from datasets import load_dataset

        for repo_id, config in LIGHTEVAL_DATASETS:
            load_dataset(repo_id, config)
