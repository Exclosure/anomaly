# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build


# Autodoc options
SPHINX_APIDOC = sphinx-apidoc
APIDOC_DIR    = $(SOURCEDIR)/source
MODULE_PATH   = $(SOURCEDIR)/../anomaly
EXCLUDE       = setup.py test/

.PHONY: apidoc clean help Makefile

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


# Used for building autodoc files
apidoc: Makefile
	@$(SPHINX_APIDOC) \
		--force \
		--module-first \
		--output-dir $(APIDOC_DIR) \
		$(MODULE_PATH) \
		$(EXCLUDE)


clean: Makefile
	-rm -r $(APIDOC_DIR)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile apidoc
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
