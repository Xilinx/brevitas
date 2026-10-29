# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
import hashlib
import os
import shutil
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.error import URLError
import zipfile

from filelock import FileLock

TEST_CACHE_VERSION = '1'

# Hub models used by the tests, pinned to a commit so that results do not change when the upstream
# repo does. resolve_hf_asset() rejects models that are not listed here, so the cache prepared in
# CI always has what the tests load. To update a pin, replace the commit id: this file is part of
# the cache key, so the cache is rebuilt.
LLM_MODELS = {
    'hf-internal-testing/tiny-random-LlamaForCausalLM':
        '9fb191250dd56d0ba7ec9785a025ed29c03d5998',
    'hf-internal-testing/tiny-random-MistralForCausalLM':
        '7517176934d50d00707900b92ef6a95c782e51a2',
    'hf-internal-testing/tiny-random-OPTForCausalLM':
        '0abea37ca0a786ba455967e799b7b3d67f86541f',}

# Pinned to what used to be the 'refs/pr/4' revision.
DIFFUSION_MODELS = {
    'hf-internal-testing/tiny-stable-diffusion-pipe': '1c1a2170e293f9ebca09bcb49da75d9a23bcdf95',}

NLTK_RESOURCES = {
    'punkt': 'tokenizers/punkt',
    'punkt_tab': 'tokenizers/punkt_tab',
    'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng',
    'stopwords': 'corpora/stopwords',}

# Few-shot datasets used by LightEval and lm_eval (lm_eval's piqa is baber/piqa), as
# (repo, config, revision). Add the dataset of any new few-shot task here: the offline CI matrix
# only has what is prepared.
EVAL_DATASETS = (
    ('allenai/ai2_arc', 'ARC-Challenge', '210d026faf9955653af8916fad021475a3f00453'),
    ('allenai/ai2_arc', 'ARC-Easy', '210d026faf9955653af8916fad021475a3f00453'),
    ('allenai/winogrande', 'winogrande_xl', '01e74176c63542e6b0bcb004dcdea22d94fb67b5'),
    ('Rowan/hellaswag', 'default', '218ec52e09a7e7462a5400043bb9a69a41d06b76'),
    ('baber/piqa', None, '142f6d7367fd9877f0fb3b5734ea6a545f54cdd1'),
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


def _is_transient(exc: Exception, transient: tuple) -> bool:
    """Whether a failed network operation is worth another attempt."""
    # HTTP errors from huggingface_hub/requests carry a response, those from urllib a code.
    status = getattr(getattr(exc, 'response', None), 'status_code', None)
    if status is None and isinstance(exc, HTTPError):
        status = exc.code
    if status is not None:
        return status in (408, 429) or status >= 500
    transient = (ConnectionError, TimeoutError, URLError, zipfile.BadZipFile, *transient)
    return isinstance(exc, transient)


def _retry(operation, transient=(), attempts=3, delay=5):
    """Run a network operation, retrying only transient failures with a growing delay.

    Connection problems, timeouts, HTTP 408/429/5xx and corrupt downloads are retried. Missing or
    gated repos and bad arguments fail immediately, and so does everything in offline mode, where
    failures are deterministic. `transient` adds exception types for APIs with their own errors.
    """
    if _env_enabled('BREVITAS_TEST_OFFLINE'):
        return operation()
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt == attempts or not _is_transient(exc, transient):
                raise
            print(f'Attempt {attempt}/{attempts} failed ({exc!r}), retrying in {delay * attempt}s')
            time.sleep(delay * attempt)


def resolve_hf_asset(repo_id: str) -> str:
    """Resolve a listed Hub model to a local snapshot path, pinned to the revision in this file.

    The lock serializes pytest-xdist workers on one machine. Separate CI jobs do not share it: the
    prepare-assets job downloads everything once and the test matrix only restores the cache.
    """
    if os.path.isdir(repo_id):
        return repo_id

    revision = {**LLM_MODELS, **DIFFUSION_MODELS}.get(repo_id)
    if revision is None:
        raise ValueError(
            f"'{repo_id}' is not listed in assets_for_tests.py. Add it with a pinned revision "
            "so that CI prepares it.")

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    cache_dir = os.path.join(get_test_cache_dir(), 'huggingface', 'hub')
    os.makedirs(cache_dir, exist_ok=True)
    common_args = {"repo_id": repo_id, "revision": revision, "cache_dir": cache_dir}

    lock_id = hashlib.sha256(f"{repo_id}\n{revision}".encode()).hexdigest()
    lock_dir = get_lock_dir()
    os.makedirs(lock_dir, exist_ok=True)
    # The local probe runs under the lock: a worker must not read a snapshot another worker is
    # still writing.
    with FileLock(os.path.join(lock_dir, f"hf-{lock_id}.lock")):
        try:
            return snapshot_download(**common_args, local_files_only=True)
        except LocalEntryNotFoundError:
            if _env_enabled("BREVITAS_TEST_OFFLINE"):
                raise
            # snapshot_download also raises this when the Hub is unreachable and nothing is cached.
            return _retry(
                partial(snapshot_download, **common_args, local_files_only=False),
                transient=(LocalEntryNotFoundError,))


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

        for repo_id, config, revision in EVAL_DATASETS:
            _retry(partial(load_dataset, repo_id, config, revision=revision))


def _download_nltk_resource(nltk, resource: str, nltk_dir: str) -> None:
    """Replace a missing or corrupt NLTK resource with a freshly downloaded, verified copy."""
    resource_path = NLTK_RESOURCES[resource]
    for path in (os.path.join(nltk_dir, f'{resource_path}.zip'),
                 os.path.join(nltk_dir, resource_path)):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
    # nltk.download returns False on any failure instead of raising.
    if not nltk.download(resource, download_dir=nltk_dir, quiet=True):
        raise ConnectionError(f'Failed to download NLTK resource: {resource}')
    nltk.data.find(resource_path)  # raises if the archive is incomplete


def _load_bnn_pynq_datasets(datadir: str, download: bool) -> None:
    """Instantiate the train and test splits of MNIST and CIFAR10, optionally downloading them."""
    from torchvision.datasets import CIFAR10

    from brevitas_examples.bnn_pynq.trainer import MirrorMNIST

    for builder in (MirrorMNIST, CIFAR10):
        for train in (True, False):
            builder(root=datadir, train=train, download=download)


def _bnn_pynq_datasets_available(datadir: str) -> bool:
    try:
        _load_bnn_pynq_datasets(datadir, download=False)
    except (RuntimeError, FileNotFoundError):
        return False
    return True


def prepare_bnn_pynq_datasets() -> None:
    """Make the datasets of the bnn_pynq tests available before any pytest-xdist worker uses them.

    The bnn_pynq trainer builds both splits of MNIST and CIFAR10 with ``download=True``. Workers
    that download the same files at the same time corrupt them, so a lock plus a sentinel file
    guarantee that only the first process downloads while the others wait and then reuse the files.
    In offline mode nothing is downloaded: missing datasets are an error.
    """
    # The default --datadir of brevitas_examples.bnn_pynq.bnn_pynq_train.launch is ./data, which is
    # the cache dir when running from the repo root, as nox does.
    datadir = get_test_cache_dir()
    os.makedirs(datadir, exist_ok=True)
    lock_dir = get_lock_dir()
    os.makedirs(lock_dir, exist_ok=True)
    sentinel = os.path.join(datadir, '.download.done')
    with FileLock(os.path.join(lock_dir, 'bnn-pynq-datasets.lock')):
        cache_valid = False
        if os.path.isfile(sentinel):
            with open(sentinel) as f:
                cache_valid = f.read() == TEST_CACHE_VERSION
        if cache_valid and _bnn_pynq_datasets_available(datadir):
            return
        if _env_enabled('BREVITAS_TEST_OFFLINE'):
            raise RuntimeError(f'The bnn_pynq datasets are missing from the cache at {datadir}')
        # torchvision raises RuntimeError when all mirrors fail or a checksum does not match.
        _retry(partial(_load_bnn_pynq_datasets, datadir, download=True), transient=(RuntimeError,))
        if not _bnn_pynq_datasets_available(datadir):
            raise RuntimeError(f'Incomplete dataset cache at {datadir}')
        sentinel_tmp = sentinel + '.tmp'
        with open(sentinel_tmp, 'w') as f:
            f.write(TEST_CACHE_VERSION)
        os.replace(sentinel_tmp, sentinel)


def _pin_default_revision(repo_id: str) -> None:
    """Point the cache's `main` ref at the pinned commit, so bare repo ids resolve offline."""
    refs_dir = os.path.join(
        get_test_cache_dir(), 'huggingface', 'hub', f"models--{repo_id.replace('/', '--')}", 'refs')
    os.makedirs(refs_dir, exist_ok=True)
    with open(os.path.join(refs_dir, 'main'), 'w') as f:
        f.write({**LLM_MODELS, **DIFFUSION_MODELS}[repo_id])


def prepare_assets(family: str) -> None:
    """Download everything a test family needs, so the test matrix only restores it."""
    if family == 'llm':
        for model in LLM_MODELS:
            resolve_hf_asset(model)
            _pin_default_revision(model)
        prepare_lighteval_assets()
    elif family == 'diffusion':
        for model in DIFFUSION_MODELS:
            resolve_hf_asset(model)
            _pin_default_revision(model)
        # Constructing the metric downloads its Inception weights into TORCH_HOME.
        from torchmetrics.image.fid import FrechetInceptionDistance
        _retry(partial(FrechetInceptionDistance, normalize=False))
    elif family == 'datasets':
        prepare_bnn_pynq_datasets()
    else:
        raise ValueError(f'Unknown asset family: {family}')


if __name__ == '__main__':
    prepare_assets(sys.argv[1])
