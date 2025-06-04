# Minimal Makefile for Sphinx documentation with sphinx-multiversion

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
MULTIVERSION  ?= sphinx-multiversion
SOURCEDIR     = source
BUILDDIR      = build
BUILD_VERSION ?=

.PHONY: help html github livehtml

help:
	@echo "Available targets:"
	@echo "  html       Build documentation for all versions using sphinx-multiversion"
	@echo "  github     Build and copy HTML docs to ../docs"
	@echo "  livehtml   Live-reload documentation (single version, not multiversion)"
	@echo "  help       Show this message"

html:
	@export VERSION=$(BUILD_VERSION) && $(MULTIVERSION) "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

github:
	@$(MAKE) html
	@cp -a "$(BUILDDIR)/html/." ../docs
	@cp -a "$(SOURCEDIR)/index.html" ../docs
	@touch .nojekyll ../docs

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/livehtml" $(SPHINXOPTS)