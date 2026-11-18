# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared version selection for the documentation build and publication artifact."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable


DEVELOPMENT_BRANCH = 'master'
SITE_URL = 'https://xilinx.github.io/brevitas'
# Set by the docs workflow to the comma-separated tags of the published GitHub Releases.
PUBLISHED_TAGS_ENV = 'BREVITAS_DOC_RELEASE_TAGS'
# Oldest release published on the documentation site; earlier tags are never built.
MINIMUM_STABLE_VERSION = (0, 9, 1)
STABLE_TAG_PATTERN = re.compile(r'v(\d+)\.(\d+)\.(\d+)')


def parse_stable_tag(tag: str) -> tuple[int, int, int] | None:
    """Return ``tag`` as a ``(major, minor, patch)`` tuple, or None if it is not a documented release."""
    match = STABLE_TAG_PATTERN.fullmatch(tag)
    if match is None:
        return None
    version = tuple(int(component) for component in match.groups())
    return version if version >= MINIMUM_STABLE_VERSION else None


def stable_tags(tags: Iterable[str]) -> list[str]:
    """Keep only documented release tags, sorted from oldest to newest."""
    documented_tags = [tag for tag in tags if parse_stable_tag(tag) is not None]
    # Sort numerically, so that v0.10.0 comes after v0.9.1.
    return sorted(documented_tags, key=parse_stable_tag)


def tags_from_environment() -> list[str] | None:
    """Return the release tags provided by the docs workflow, or None when building outside CI."""
    value = os.environ.get(PUBLISHED_TAGS_ENV)
    if value is None:
        return None
    return stable_tags(value.split(','))


def git_stable_tags() -> list[str]:
    """Return the release tags of the local repository, used for local builds."""
    result = subprocess.run(
        ['git', 'tag', '--list'], check=True, capture_output=True, text=True)
    return stable_tags(result.stdout.splitlines())


def selected_release_tags() -> list[str]:
    """Return the release tags to document: the ones from CI if provided, otherwise the local Git tags."""
    environment_tags = tags_from_environment()
    if environment_tags is not None:
        return environment_tags
    return git_stable_tags()


def tag_whitelist(tags: Iterable[str]) -> str:
    """Build the sphinx-multiversion regex that matches exactly ``tags``."""
    # With no tags this is `^(?:)$`, which matches no tag name, so only master is built.
    return r'^(?:' + '|'.join(re.escape(tag) for tag in tags) + r')$'


def published_release_tags(release_pages: list[list[dict]]) -> list[str]:
    """Extract the documented release tags from the GitHub Releases API output, skipping drafts and pre-releases."""
    # `gh api --paginate --slurp` returns one list of releases per page.
    return stable_tags(
        release['tag_name']
        for page in release_pages
        for release in page
        if not release['draft'] and not release['prerelease'])


def main() -> None:
    """Print the documented release tags, comma-separated, from a GitHub Releases API JSON file."""
    parser = argparse.ArgumentParser(description='Select published Brevitas documentation tags.')
    parser.add_argument('--published-releases', type=Path, required=True)
    args = parser.parse_args()

    with args.published_releases.open(encoding='utf-8') as releases_file:
        releases = json.load(releases_file)
    print(','.join(published_release_tags(releases)))


if __name__ == '__main__':
    main()
