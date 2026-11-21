# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Validate a multiversion Sphinx build and prepare it for GitHub Pages."""

import argparse
import json
from pathlib import Path
import shutil

from versioning import DEVELOPMENT_BRANCH
from versioning import SITE_URL
from versioning import selected_release_tags


def remove_doctrees(site_directory: Path) -> None:
    """Delete the Sphinx `.doctrees` build caches, which are not needed to serve the site."""
    for path in site_directory.rglob('.doctrees'):
        shutil.rmtree(path)


def validate_versions(site_directory: Path, release_tags: list[str]) -> None:
    """Fail if a selected version has no folder in the site."""
    # sphinx-multiversion only logs an error and moves on when it cannot load a tag's config,
    # so a version can be missing even though the build succeeded.
    missing_versions = [
        version for version in (DEVELOPMENT_BRANCH, *release_tags)
        if not (site_directory / version).is_dir()]
    if missing_versions:
        raise RuntimeError(f'sphinx-multiversion did not build {missing_versions}')


def write_versions_manifest(site_directory: Path, release_tags: list[str]) -> None:
    """Write the version-switcher list, marking the newest release as the stable one."""
    manifest = [{
        'name': 'master (development)',
        'version': DEVELOPMENT_BRANCH,
        'url': f'{SITE_URL}/{DEVELOPMENT_BRANCH}/',
    }]
    manifest.extend({
        'name': tag,
        'version': tag,
        'url': f'{SITE_URL}/{tag}/',
    } for tag in release_tags)
    if release_tags:
        # release_tags is sorted by version, so the last entry is the latest stable release.
        # `preferred` is the PyData theme flag that identifies the latest stable version.
        manifest[-1]['name'] = f'{release_tags[-1]} (stable)'
        manifest[-1]['preferred'] = True

    # Every version reads the switcher from master (see `json_url` in conf.py), so older
    # releases also list versions published after them.
    manifest_path = site_directory / DEVELOPMENT_BRANCH / '_static' / 'versions.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def write_root_redirect(site_directory: Path, release_tags: list[str]) -> None:
    """Write the site's root index.html, which redirects to the latest stable release."""
    destination = release_tags[-1] if release_tags else DEVELOPMENT_BRANCH
    target = f'./{destination}/'
    site_directory.joinpath('index.html').write_text(
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '  <meta charset="utf-8">\n'
        f'  <meta http-equiv="refresh" content="0; url={target}">\n'
        f'  <link rel="canonical" href="{target}">\n'
        '  <title>Brevitas documentation</title>\n'
        '</head>\n'
        '<body>\n'
        f'  <p>Redirecting to <a href="{target}">the latest stable Brevitas documentation</a>.</p>\n'
        '</body>\n'
        '</html>\n',
        encoding='utf-8')


def main() -> None:
    """Turn the sphinx-multiversion output into a validated GitHub Pages site."""
    parser = argparse.ArgumentParser(description='Prepare a complete GitHub Pages documentation artifact.')
    parser.add_argument('site_directory', type=Path)
    args = parser.parse_args()

    site_directory = args.site_directory
    release_tags = selected_release_tags()
    remove_doctrees(site_directory)
    validate_versions(site_directory, release_tags)
    write_versions_manifest(site_directory, release_tags)
    write_root_redirect(site_directory, release_tags)
    # Serve the files as they are: without it, Jekyll would hide Sphinx folders such as `_static`.
    (site_directory / '.nojekyll').touch()

    print(f'Prepared Pages artifact for {DEVELOPMENT_BRANCH} and {", ".join(release_tags) or "no releases"}.')


if __name__ == '__main__':
    main()
