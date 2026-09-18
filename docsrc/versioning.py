"""Shared version selection for the documentation build and publication artifact."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable
from typing import Mapping


DEVELOPMENT_BRANCH = 'master'
SITE_URL = 'https://xilinx.github.io/brevitas'
PUBLISHED_TAGS_ENV = 'BREVITAS_DOC_RELEASE_TAGS'
MINIMUM_STABLE_VERSION = (0, 9, 1)
STABLE_TAG_PATTERN = re.compile(r'^v(\d+)\.(\d+)\.(\d+)$')


def parse_stable_tag(tag: str) -> tuple[int, int, int] | None:
    match = STABLE_TAG_PATTERN.fullmatch(tag)
    if match is None:
        return None
    version = tuple(int(component) for component in match.groups())
    return version if version >= MINIMUM_STABLE_VERSION else None


def stable_tags(tags: Iterable[str]) -> list[str]:
    """Return supported stable tags in semantic-version order."""
    unique_tags = {tag for tag in tags if parse_stable_tag(tag) is not None}
    return sorted(unique_tags, key=lambda tag: parse_stable_tag(tag))


def tags_from_environment() -> list[str] | None:
    """Read the release tags selected by the Pages workflow, when present."""
    value = os.environ.get(PUBLISHED_TAGS_ENV)
    if value is None:
        return None
    return stable_tags(tag for tag in value.split(',') if tag)


def git_stable_tags() -> list[str]:
    """Discover supported tags for local builds without GitHub Release metadata."""
    result = subprocess.run(
        ['git', 'tag', '--list'], check=True, capture_output=True, text=True)
    return stable_tags(result.stdout.splitlines())


def selected_release_tags() -> list[str]:
    """Prefer explicitly published releases; retain Git discovery for local use."""
    environment_tags = tags_from_environment()
    if environment_tags is not None:
        return environment_tags
    return git_stable_tags()


def tag_whitelist(tags: Iterable[str]) -> str:
    """Return a sphinx-multiversion regex matching exactly ``tags``."""
    escaped_tags = [re.escape(tag) for tag in stable_tags(tags)]
    if not escaped_tags:
        return r'$.^'
    return r'^(?:' + '|'.join(escaped_tags) + r')$'


def published_release_tags(releases: object) -> list[str]:
    """Extract non-draft, non-prerelease stable tags from GitHub API output."""
    if isinstance(releases, list) and releases and all(isinstance(item, list) for item in releases):
        releases = [release for page in releases for release in page]
    if not isinstance(releases, list):
        raise ValueError('GitHub releases JSON must be a list of releases')

    tags = []
    for release in releases:
        if not isinstance(release, Mapping):
            raise ValueError('GitHub releases JSON contains an invalid release')
        if release.get('draft') or release.get('prerelease'):
            continue
        tag = release.get('tag_name')
        if isinstance(tag, str):
            tags.append(tag)
    return stable_tags(tags)


def main() -> None:
    parser = argparse.ArgumentParser(description='Select published Brevitas documentation tags.')
    parser.add_argument('--published-releases', type=Path, required=True)
    args = parser.parse_args()

    with args.published_releases.open(encoding='utf-8') as releases_file:
        releases = json.load(releases_file)
    print(','.join(published_release_tags(releases)))


if __name__ == '__main__':
    main()
