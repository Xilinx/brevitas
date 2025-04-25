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
import sys

import brevitas

sys.path.insert(0, os.path.abspath(brevitas.__file__))
# -- Project information -----------------------------------------------------

project = 'Brevitas'
copyright = '2023 - Advanced Micro Devices, Inc.'
author = 'Alessandro Pappalardo'

# The full version, including alpha/beta/rc tags
release = brevitas.__version__


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

# Dictionary of theme options
# html_logo is broken on sphinx 6
# https://github.com/pydata/pydata-sphinx-theme/issues/1094
html_theme_options = {
   "header_links_before_dropdown": 8,
   "logo": {
      "image_light": "brevitas_logo_black.svg",
      "image_dark": "brevitas_logo_white.svg",
   }
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Ensure env.metadata[env.docname]['nbsphinx-link-target']
# points relative to repo root:
nbsphinx_link_target_root =  os.path.join(os.path.dirname(__file__), '..', '..')


intersphinx_mapping = {
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "python": ("https://docs.python.org/", None),
    "torch": ("https://pytorch.org/docs/main/", None),
}
