# ==============================================================================
# Makefile — SO-ARM101 MuJoCo Simulation (Zero Host Dependencies)
# ==============================================================================
.DEFAULT_GOAL := help
.PHONY: help build install tests test verify run dev start stop logs shell clean

DOCKER      ?= $(shell command -v docker 2>/dev/null || echo /Applications/Docker.app/Contents/Resources/bin/docker)
COMPOSE     := $(DOCKER) compose -f docker-compose.yml
COMPOSE_DEV := $(DOCKER) compose -f docker-compose.yml -f docker-compose.dev.yml
DOCKER_RUN  := $(COMPOSE_DEV) run --rm --no-deps sim

SEED        ?= 42
TRIALS      ?= 5
STUDIO_PORT ?= 8080
JSON_OUT    ?= tests/sim_suite_report.json

BLUE  := \033[0;34m
GREEN := \033[0;32m
NC    := \033[0m

help: ## Displays this help menu
	@printf "\n$(BLUE)SO-ARM101 MuJoCo Simulation — Containerized Commands:$(NC)\n\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-16s$(NC) %s\n", $$1, $$2}'
	@printf "\n"

build: ## Builds dev and production Docker images
	$(COMPOSE_DEV) build
	$(COMPOSE) build

install: ## Builds dev container and installs Python dependencies
	$(COMPOSE_DEV) build

tests: ## Runs the MuJoCo simulation test suite and base verification
	$(DOCKER_RUN) python tests/sim_suite.py --seed $(SEED) --trials $(TRIALS) --json-out $(JSON_OUT)
	$(DOCKER_RUN) python pick_and_place_verify.py

test: tests ## Alias for 'tests'

verify: ## Runs base pick-and-place verification and renders 2x2 summary
	$(DOCKER_RUN) python pick_and_place_verify.py

run: ## Starts MuJoCo Studio WebViewer server in the foreground
	STUDIO_PORT=$(STUDIO_PORT) $(COMPOSE_DEV) up

dev: run ## Alias for 'run'

start: ## Starts MuJoCo Studio WebViewer server in the background
	STUDIO_PORT=$(STUDIO_PORT) $(COMPOSE_DEV) up -d

stop: ## Stops background containers cleanly
	$(COMPOSE_DEV) down

logs: ## Streams live logs from the simulation container
	$(COMPOSE_DEV) logs -f sim

shell: ## Opens an interactive bash shell inside the container
	$(DOCKER_RUN) /bin/bash

clean: ## Stops containers and removes networks and volumes
	$(COMPOSE_DEV) down --remove-orphans -v
