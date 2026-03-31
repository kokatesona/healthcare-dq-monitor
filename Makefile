.PHONY: help setup up down lint test ingest dbt-run dbt-test

help:
	@echo ""
	@echo "  Healthcare DQ Monitor — dev commands"
	@echo ""
	@echo "  make setup       Install Python dependencies"
	@echo "  make up          Start Docker services"
	@echo "  make down        Stop Docker services"
	@echo "  make lint        Run ruff linter"
	@echo "  make test        Run pytest with coverage"
	@echo "  make ingest      Generate synthetic data"
	@echo "  make dbt-run     Run dbt models"
	@echo "  make dbt-test    Run dbt schema tests"
	@echo ""

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	cp -n .env.example .env || true

up:
	docker compose up -d
	@echo "Airflow  → http://localhost:8080  (admin / admin)"
	@echo "MLflow   → http://localhost:5001"

down:
	docker compose down

lint:
	.venv/bin/ruff check src/ tests/

test:
	.venv/bin/pytest tests/ --cov=src --cov-report=term-missing -v

ingest:
	.venv/bin/python src/ingest/generate_synthetic.py

dbt-run:
	cd models/dbt/healthcare_dq && ../../../.venv/bin/dbt run

dbt-test:
	cd models/dbt/healthcare_dq && ../../../.venv/bin/dbt test
