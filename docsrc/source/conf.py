# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
from pathlib import Path
import subprocess
import sys


# Resolve sources from the revision copied by sphinx-multiversion. Otherwise autodoc imports the
# installed development package for every historical tag.
multiversion_source = os.environ.get('SPHINX_MULTIVERSION_SOURCEDIR')
if multiversion_source:
    docs_source_dir = Path(multiversion_source).resolve()
else:
    docs_source_dir = Path(__file__).resolve().parent
repository_root = docs_source_dir.parents[1]
version_src_dir = repository_root / 'src'
sys.path.insert(0, str(version_src_dir))


def get_current_branch_name():
    try:
        # Get the symbolic reference for the current branch
        result = subprocess.run(
            ['git', 'symbolic-ref', '--short', 'HEAD'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            # Fallback: try 'git branch --show-current' (Git >= 2.22)
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return None
    except Exception as e:
        print(f"Error: {e}")
        return None


import brevitas

brevitas_path = Path(brevitas.__file__).resolve()
if not brevitas_path.is_relative_to(version_src_dir):
    raise RuntimeError(
        f"Expected to import brevitas from {version_src_dir}, imported {brevitas_path} instead")
# -- Project information -----------------------------------------------------

project = 'Brevitas'
copyright = '2025 - Advanced Micro Devices, Inc.'
author = 'AMD Research and Advanced Development'

current_version = os.environ.get('SPHINX_MULTIVERSION_NAME')
# importlib.metadata reports the installed package version, so use the tag name for releases.
if current_version and current_version.startswith('v'):
    release = current_version.removeprefix('v')
else:
    release = brevitas.__version__
local_branch = get_current_branch_name()


def set_version_from_multiversion_name(app, config):
    current_version = os.environ.get('SPHINX_MULTIVERSION_NAME')
    if current_version and current_version.startswith('v'):
        tag_version = current_version.removeprefix('v')
        config.version = tag_version
        config.release = tag_version


def setup(app):
    # sphinx-multiversion restores metadata read from the tag's conf.py during config-inited.
    app.connect('config-inited', set_version_from_multiversion_name, priority=1000)

# It is possible to invoke the documentation command by specifing:
# - A specific version to build
# - 'local', which will build the current branch as if it were the master branch
# - Nothing, which will build all documentations for a bunch of different versions specified below and the actual master branch
version_to_build = os.environ.get('VERSION', '')
if version_to_build == 'local':
    current_version = 'master'
    smv_outputdir_format = 'master'
    branch_to_build = local_branch
elif version_to_build == '':
    # This select all versions above v0.9
    # ^v: Starts with v
    # ([1-9][0-9]*\.\d+\.\d+): Matches v1.0.0, v2.3.4, etc. (major version ≥ 1)
    # |: OR
    # 0\.(1[0-9]|\d{2,})\.\d+: Matches v0.10.0, v0.11.0, ..., v0.99.0, etc. (minor version ≥ 10)
    # |: OR
    # 0\.9\.(?!0+$)\d+: Matches v0.9.1, v0.9.2, ..., but not v0.9.0
    # $: End of string
    version_to_build = r'^v([1-9][0-9]*\.\d+\.\d+|0\.(1[0-9]|\d{2,})\.\d+|0\.9\.(?!0+$)\d+)$'
    branch_to_build = 'master'
else:
    branch_to_build = 'master'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'm2r2',
    'sphinx.ext.autodoc',
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinxemoji.sphinxemoji',
    'nbsphinx',
    'nbsphinx_link',
    'sphinx_gallery.load_style',
    "sphinx.ext.githubpages",
    "sphinx_multiversion",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Napoleon config
napoleon_use_param = True


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
autodoc_mock_imports = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'pydata_sphinx_theme'
html_title = f'Brevitas Documentation - {current_version}'
# Dictionary of theme options
# html_logo is broken on sphinx 6
# https://github.com/pydata/pydata-sphinx-theme/issues/1094
html_theme_options = {
   "header_links_before_dropdown": 6,
   "logo": {
      "image_light": "brevitas_logo_black.svg",
      "image_dark": "brevitas_logo_white.svg",
   },
    "switcher": {
        "json_url": "https://xilinx.github.io/brevitas/master/_static/versions.json",
        "version_match": current_version,
    },
    "footer_end": ["version-switcher"]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Ensure env.metadata[env.docname]['nbsphinx-link-target']
# points relative to repo root:
nbsphinx_link_target_root = str(repository_root)


intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/1.26/", None),
    "python": ("https://docs.python.org/3.10", None),
    "torch": ("https://pytorch.org/docs/2.7/", None),
}

smv_tag_whitelist = version_to_build            # Select which tag to build

smv_branch_whitelist = branch_to_build

smv_remote_whitelist = None                     # Only use local branches
