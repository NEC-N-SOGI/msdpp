project_name = msdpp
image_name = $(USER)-$(project_name)
container_name = $(USER)-$(project_name)
gpu = all
home = /home/$(USER)
data_dir = /data/$(USER)
options = --mount type=bind,source=$(home)/.cache/uv,target=$(home)/.cache/uv --mount type=bind,source=$(data_dir),target=/$(project_name)/share_datasets

up: .build ## Launch a container
	docker run -it --rm --name=$(container_name) --ipc=host \
		--gpus='"device=$(gpu)"' \
		-v $$(pwd):/$(project_name) \
		-v /${project_name}/.venv \
		$(options) \
		$(image_name)


build: .build ;

.PRECIOUS: .build
.build: environments/Dockerfile pyproject.toml uv.lock
	DOCKER_BUILDKIT=1 docker build -t $(image_name) \
		--progress=plain \
		--target=$${BUILD_TARGET} \
		--build-arg PROJECT_NAME=$(project_name) \
		--build-arg PROJECT_DIRECTORY=/$(project_name) \
		--build-arg PYTHON_VERSION=3.10 \
		--build-arg LOCAL_UID=$$(id -u) \
		--build-arg LOCAL_GID=$$(id -g) \
		--build-arg LOCAL_USER_NAME=$$(id -un) \
		-f environments/Dockerfile \
		.
	touch $@

venv: uv.lock pyproject.toml ## Create venv and install the project
	uv sync --all-extras --native-tls

.PRECIOUS: uv.lock
uv.lock:
	uv lock --native-tls

.PHONY: ruff-format
ruff-format:
	uv run ruff check --fix src
	uv run ruff format src

.PHONY: ruff-lint
ruff-lint:
	uv run ruff check src
	uv run ruff check src --exit-zero

.PHONY: mdformat
mdformat:
	uv run mdformat *.md

.PHONY: mdformat-check
mdformat-check:
	uv run mdformat --check *.md

.PHONY: mypy
mypy:
	uv run mypy src

.PHONY: ty
ty:
	uv run ty check src

.PHONY: pyright
pyright:
	uv run pyright src

.PHONY: validate-project
validate-project:
	uv run validate-pyproject pyproject.toml

.PHONY: format
format: ## Apply formatters to the project
	$(MAKE) ruff-format
	$(MAKE) mdformat

.PHONY: lint
lint: ## Apply formatter checks and linters to the project
	$(MAKE) ruff-lint
	$(MAKE) ty
	$(MAKE) mdformat-check
	$(MAKE) mypy
	$(MAKE) pyright
	$(MAKE) validate-project

.PHONY: precommit
precommit: ## Run pre-commit hooks
	uv run pre-commit run --all-files

.PHONY: prepush
prepush: ## Run pre-push hooks
	uv run pre-commit run --all-files --hook-stage push

.PHONY: precommit-lint
precommit-lint: ## Apply formatter checks and linters to the project
	$(MAKE) ruff-lint
	$(MAKE) ty
	$(MAKE) mdformat-check
	$(MAKE) pyright
	$(MAKE) validate-project

.PHONY: help
.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z%._-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
