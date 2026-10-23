# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
import hashlib
import os
import shutil
import sys
import tempfile
import time
import zipfile

from filelock import FileLock

TEST_CACHE_VERSION = '1'

LLM_MODELS = (
    'hf-internal-testing/tiny-random-LlamaForCausalLM',
    'hf-internal-testing/tiny-random-MistralForCausalLM',
    'hf-internal-testing/tiny-random-OPTForCausalLM',
)

DIFFUSION_MODELS = (('hf-internal-testing/tiny-stable-diffusion-pipe', 'refs/pr/4'),)

NLTK_RESOURCES = {
    'punkt': 'tokenizers/punkt',
    'punkt_tab': 'tokenizers/punkt_tab',
    'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
    'stopwords': 'corpora/stopwords',}

# Few-shot datasets used by LightEval and lm_eval (lm_eval's piqa is baber/piqa).
EVAL_DATASETS = (
    ('allenai/ai2_arc', 'ARC-Challenge'),
    ('allenai/ai2_arc', 'ARC-Easy'),
    ('allenai/winogrande', 'winogrande_xl'),
    ('Rowan/hellaswag', 'default'),
    ('baber/piqa', None),
)


def get_test_cache_dir() -> str:
    cache_dir = os.environ.get('BREVITAS_TEST_CACHE_DIR')
    if cache_dir:
        return cache_dir
    return os.path.join(os.getcwd(), 'data')


def get_lock_dir() -> str:
    # Locks only matter within one machine, so keep them out of the cached directory.
    # The hash keeps different caches (and users on a shared machine) apart.
    cache_id = hashlib.sha256(get_test_cache_dir().encode()).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f'brevitas-test-locks-{cache_id}')


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _retry(operation, attempts=3, delay=5):
    """Run a network operation, retrying failures with a growing delay.

    Offline failures are deterministic, so offline mode never retries.
    """
    if _env_enabled('BREVITAS_TEST_OFFLINE'):
        return operation()
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == attempts:
                raise
            print(f'Attempt {attempt}/{attempts} failed ({exc!r}), retrying in {delay * attempt}s')
            time.sleep(delay * attempt)


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
        "cache_dir": cache_dir,}

    lock_id = hashlib.sha256(f"{repo_id}\n{revision}".encode()).hexdigest()
    lock_dir = get_lock_dir()
    os.makedirs(lock_dir, exist_ok=True)
    # The local probe runs under the lock: a worker must not read a snapshot another worker is still writing.
    with FileLock(os.path.join(lock_dir, f"hf-{lock_id}.lock")):
        try:
            return snapshot_download(**common_args, local_files_only=True)
        except LocalEntryNotFoundError:
            if _env_enabled("BREVITAS_TEST_OFFLINE"):
                raise
            return _retry(partial(snapshot_download, **common_args, local_files_only=False))


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
            try:
                nltk.data.find(NLTK_RESOURCES[resource])
            except (LookupError, zipfile.BadZipFile):
                if _env_enabled('BREVITAS_TEST_OFFLINE'):
                    raise RuntimeError(f'Missing NLTK resource in offline mode: {resource}')
                _retry(partial(_download_nltk_resource, nltk, resource, nltk_dir))

        from datasets import load_dataset

        for repo_id, config in EVAL_DATASETS:
            _retry(partial(load_dataset, repo_id, config))


def _download_nltk_resource(nltk, resource: str, nltk_dir: str) -> None:
    """Replace a missing or corrupt NLTK resource with a freshly downloaded, verified copy."""
    resource_path = NLTK_RESOURCES[resource]
    for path in (os.path.join(nltk_dir, f'{resource_path}.zip'),
                 os.path.join(nltk_dir, resource_path)):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    if not nltk.download(resource, download_dir=nltk_dir, quiet=True):
        raise RuntimeError(f'Failed to download NLTK resource: {resource}')
    nltk.data.find(resource_path)  # raises if the archive is incomplete


def prepare_assets(family: str) -> None:
    """Download everything a test family needs, so the test matrix only restores it."""
    if family == 'llm':
        for model in LLM_MODELS:
            resolve_hf_asset(model)
        prepare_lighteval_assets()
    elif family == 'diffusion':
        for model, revision in DIFFUSION_MODELS:
            resolve_hf_asset(model, revision=revision)
        # Constructing the metric downloads its Inception weights into TORCH_HOME.
        from torchmetrics.image.fid import FrechetInceptionDistance
        FrechetInceptionDistance(normalize=False)
    else:
        raise ValueError(f'Unknown asset family: {family}')


if __name__ == '__main__':
    prepare_assets(sys.argv[1])
