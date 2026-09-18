# Minimal Makefile for Sphinx documentation with sphinx-multiversion

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
MULTIVERSION  ?= sphinx-multiversion
PYTHON        ?= python
SOURCEDIR     = source
BUILDDIR      = build
VERSION       ?=
BUILD_VERSION ?= $(VERSION)

export VERSION=$(BUILD_VERSION)

.PHONY: help clean html github pages livehtml

help:
	@echo "Available targets:"
	@echo "  html       Build documentation for all versions using sphinx-multiversion"
	@echo "  pages      Build and validate the complete GitHub Pages artifact"
	@echo "  github     Deprecated alias for pages"
	@echo "  livehtml   Live-reload documentation (single version, not multiversion)"
	@echo "  clean      Remove generated HTML documentation"
	@echo "  help       Show this message"

clean:
	@rm -rf "$(BUILDDIR)/html"

html:
	@$(MULTIVERSION) "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

pages: clean html
	@$(PYTHON) prepare_pages.py "$(BUILDDIR)/html"

github: pages

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/livehtml" $(SPHINXOPTS)
