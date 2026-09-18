"""Validate a multiversion Sphinx build and prepare it for GitHub Pages."""

import argparse
import html
import json
from pathlib import Path
import shutil
import stat

from versioning import DEVELOPMENT_BRANCH
from versioning import SITE_URL
from versioning import selected_release_tags


def assert_regular_files(site_directory: Path) -> None:
    for path in site_directory.rglob('*'):
        path_stat = path.lstat()
        if stat.S_ISLNK(path_stat.st_mode):
            raise RuntimeError(f'Pages artifact contains a symbolic link: {path}')
        if path.is_dir():
            continue
        if not stat.S_ISREG(path_stat.st_mode):
            raise RuntimeError(f'Pages artifact contains a non-regular file: {path}')
        if path_stat.st_nlink != 1:
            raise RuntimeError(f'Pages artifact contains a hard link: {path}')


def remove_doctrees(site_directory: Path) -> None:
    for path in site_directory.rglob('.doctrees'):
        if path.is_dir():
            shutil.rmtree(path)


def validate_versions(site_directory: Path, release_tags: list[str]) -> None:
    expected_versions = {DEVELOPMENT_BRANCH, *release_tags}
    actual_versions = {
        path.name for path in site_directory.iterdir()
        if path.is_dir()
    }
    unexpected_versions = actual_versions - expected_versions
    missing_versions = expected_versions - actual_versions
    if missing_versions or unexpected_versions:
        raise RuntimeError(
            f'Expected versions {sorted(expected_versions)}, found {sorted(actual_versions)}; '
            f'missing {sorted(missing_versions)}, unexpected {sorted(unexpected_versions)}')

    for version in expected_versions:
        if not (site_directory / version / 'index.html').is_file():
            raise RuntimeError(f'Missing documentation index for {version}')


def write_versions_manifest(site_directory: Path, release_tags: list[str]) -> None:
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

    manifest_path = site_directory / DEVELOPMENT_BRANCH / '_static' / 'versions.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def write_root_redirect(site_directory: Path, release_tags: list[str]) -> None:
    destination = release_tags[-1] if release_tags else DEVELOPMENT_BRANCH
    target = f'./{destination}/'
    escaped_target = html.escape(target, quote=True)
    site_directory.joinpath('index.html').write_text(
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '  <meta charset="utf-8">\n'
        f'  <meta http-equiv="refresh" content="0; url={escaped_target}">\n'
        f'  <link rel="canonical" href="{escaped_target}">\n'
        '  <title>Brevitas documentation</title>\n'
        '</head>\n'
        '<body>\n'
        f'  <p>Redirecting to <a href="{escaped_target}">the latest stable Brevitas documentation</a>.</p>\n'
        '</body>\n'
        '</html>\n',
        encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare a complete GitHub Pages documentation artifact.')
    parser.add_argument('site_directory', type=Path)
    args = parser.parse_args()

    site_directory = args.site_directory.resolve()
    if not site_directory.is_dir():
        raise RuntimeError(f'Documentation output directory does not exist: {site_directory}')

    release_tags = selected_release_tags()
    remove_doctrees(site_directory)
    validate_versions(site_directory, release_tags)
    assert_regular_files(site_directory)
    write_versions_manifest(site_directory, release_tags)
    write_root_redirect(site_directory, release_tags)
    (site_directory / '.nojekyll').touch()

    print(f'Prepared Pages artifact for {DEVELOPMENT_BRANCH} and {", ".join(release_tags) or "no releases"}.')


if __name__ == '__main__':
    main()
