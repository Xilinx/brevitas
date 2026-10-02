# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import os
from pathlib import Path

from filelock import FileLock

from tests.brevitas_examples._cache import get_hf_cache_dir
from tests.brevitas_examples._cache import get_lock_dir


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
