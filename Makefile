SHELL := /bin/bash

PROJECT_NAME := mlops-demand-forecasting
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
UVICORN := $(VENV)/bin/uvicorn
COMPOSE := docker compose

.DEFAULT_GOAL := help

.PHONY: help
.PHONY: setup venv install env directories
.PHONY: generate-data validate-data build-features train evaluate pipeline
.PHONY: api test test-coverage lint format quality
.PHONY: docker-build docker-up docker-down docker-restart docker-ps docker-logs
.PHONY: airflow-logs mlflow-logs api-logs
.PHONY: airflow-dags airflow-trigger
.PHONY: clean clean-data clean-artifacts clean-docker reset

help:
	@echo ""
	@echo "$(PROJECT_NAME)"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@echo "  make setup              Create the virtual environment and install dependencies"
	@echo "  make generate-data      Generate synthetic demand data"
	@echo "  make validate-data      Validate the raw dataset"
	@echo "  make build-features     Create the processed feature dataset"
	@echo "  make train              Train models and log experiments in MLflow"
	@echo "  make evaluate           Evaluate the trained model"
	@echo "  make pipeline           Run the complete local ML pipeline"
	@echo "  make api                Run the FastAPI application locally"
	@echo "  make test               Run automated tests"
	@echo "  make test-coverage      Run tests with coverage"
	@echo "  make lint               Run Ruff code checks"
	@echo "  make format             Format Python files"
	@echo "  make quality            Run lint, formatting check, and tests"
	@echo "  make docker-build       Build all Docker images"
	@echo "  make docker-up          Start all Docker services"
	@echo "  make docker-down        Stop all Docker services"
	@echo "  make docker-restart     Restart the Docker environment"
	@echo "  make docker-ps          Show running services"
	@echo "  make docker-logs        Follow all service logs"
	@echo "  make airflow-logs       Follow Airflow logs"
	@echo "  make mlflow-logs        Follow MLflow logs"
	@echo "  make api-logs           Follow FastAPI logs"
	@echo "  make airflow-dags       List Airflow DAGs"
	@echo "  make airflow-trigger    Trigger the training DAG"
	@echo "  make clean              Remove Python cache files"
	@echo "  make clean-data         Remove generated datasets"
	@echo "  make clean-artifacts    Remove generated model artifacts"
	@echo "  make clean-docker       Remove containers and volumes"
	@echo "  make reset              Reset the local development environment"
	@echo ""

setup: venv install env directories
	@echo "Project environment configured successfully."

venv:
	@if [ ! -d "$(VENV)" ]; then \
		python3 -m venv $(VENV); \
		echo "Virtual environment created."; \
	else \
		echo "Virtual environment already exists."; \
	fi

install: venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt

env:
	@if [ ! -f ".env" ]; then \
		cp .env.example .env; \
		sed -i "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$$(id -u)/" .env; \
		echo ".env file created."; \
	else \
		echo ".env file already exists."; \
	fi

directories:
	mkdir -p \
		data/raw \
		data/processed \
		artifacts \
		airflow/logs \
		airflow/plugins

generate-data:
	$(PYTHON) -m src.data.generate_data

validate-data:
	$(PYTHON) -m src.data.validation

build-features:
	$(PYTHON) -m src.features.build_features

train:
	$(PYTHON) -m src.models.train

evaluate:
	$(PYTHON) -m src.models.evaluate

pipeline: generate-data validate-data build-features train evaluate
	@echo "Complete Machine Learning pipeline executed successfully."

api:
	$(UVICORN) api.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload

test:
	$(PYTEST) tests -v

test-coverage:
	$(PYTEST) tests \
		-v \
		--cov=src \
		--cov=api \
		--cov-report=term-missing \
		--cov-report=html

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

quality:
	$(RUFF) check .
	$(RUFF) format . --check
	$(PYTEST) tests -v

docker-build: env directories
	$(COMPOSE) build

docker-up: env directories
	$(COMPOSE) up -d --build

docker-down:
	$(COMPOSE) down

docker-restart:
	$(COMPOSE) down
	$(COMPOSE) up -d --build

docker-ps:
	$(COMPOSE) ps

docker-logs:
	$(COMPOSE) logs -f --tail=200

airflow-logs:
	$(COMPOSE) logs -f --tail=200 \
		airflow-api-server \
		airflow-scheduler \
		airflow-dag-processor

mlflow-logs:
	$(COMPOSE) logs -f --tail=200 mlflow

api-logs:
	$(COMPOSE) logs -f --tail=200 api

airflow-dags:
	$(COMPOSE) exec airflow-scheduler airflow dags list

airflow-trigger:
	$(COMPOSE) exec airflow-scheduler \
		airflow dags trigger ml_training_pipeline

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov coverage.xml

clean-data:
	rm -f data/raw/*.csv
	rm -f data/processed/*.csv

clean-artifacts:
	rm -rf artifacts/*
	mkdir -p artifacts

clean-docker:
	$(COMPOSE) down -v --remove-orphans

reset: clean-docker clean clean-data clean-artifacts
	@echo "Project environment reset successfully."