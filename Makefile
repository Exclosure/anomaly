PYTHON_VERSION ?= $(shell python -c 'import sys; v=sys.version_info; print(f"{v.major}.{v.minor}")')

# Leave as the first command to make it the default command.
.PHONY: default
defualt: help

# Other commands are in alphabetical order
.PHONY: all
all: docs  ## Build all


.PHONY: black
black:  ## Format code using black
	@python -m black --version
	@python -m black .


.PHONY: ci-lint
ci-lint:  ## Run CI linting
	@pre-commit --version
	@pre-commit run --all-files --verbose


.PHONY: ci
ci: ci-lint test  ## Run all tests in CI


.PHONY: clean
clean: clean-docs  ## Clean all


.PHONY: clean-docs
clean-docs:  ## Clean docs build
	@$(MAKE) -C docs/ clean


.PHONY: docs
docs:  ## Make documents in HTML format.
	@$(MAKE) -C docs/ html


.PHONY: help
help:  ## Self-documenting help command.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'


.PHONY: lint
lint:  ## Lint code using pylint
	@python -m pylint --version
	@python -m pylint --verbose anomaly/


.PHONY: test
test:  ## Run tests
	@python -m pytest --version
	@python -m pytest \
		--cov=anomaly \
		--doctest-modules \
		--junitxml=reports/junit/test-results-${PYTHON_VERSION}.xml \
		--cov-report=xml:reports/coverage-${PYTHON_VERSION}.xml \
		--cov-report=html:reports/coverage-html-${PYTHON_VERSION} \
		--cov-fail-under=95 \
		tests
