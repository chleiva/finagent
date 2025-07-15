.PHONY: help install test lint format clean run-demo run-launcher setup

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in development mode
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

install-all: ## Install all dependencies including notebooks
	pip install -e ".[dev,notebooks]"

test: ## Run tests
	python -m pytest tests/ -v

test-cov: ## Run tests with coverage
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

lint: ## Run linting
	flake8 src/ tests/
	mypy src/

format: ## Format code with black
	black src/ tests/

format-check: ## Check if code is formatted correctly
	black --check src/ tests/

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

run-demo: ## Run the demo launcher
	python src/launchers/demo_launcher.py

run-launcher: ## Run the trading model launcher
	python src/launchers/trading_model_launcher.py

setup: ## Initial setup of the project
	pip install -r config/requirements.txt
	pip install -e ".[dev]"

data-prep: ## Prepare training data
	python src/data_processing/concatenate_all.py

train: ## Train a model (optimized 1M - main training script)
	python src/model_training/model_training_15Jul_optimized1M.py

evaluate: ## Evaluate models
	python src/evaluation/evaluate.py

jupyter: ## Start Jupyter notebook server
	jupyter notebook --notebook-dir=notebooks/

docs: ## Generate documentation
	cd docs && make html

docker-build: ## Build Docker image
	docker build -t trading-model-launcher .

docker-run: ## Run Docker container
	docker run -it trading-model-launcher

# Development workflow
dev-setup: install-dev format lint test ## Complete development setup

pre-commit: format lint test ## Run pre-commit checks

# Data processing
process-data: ## Process raw data
	python src/data_processing/prepare_data.py

# Model management
list-models: ## List available models
	ls -la models/saved/

clean-models: ## Clean old model files
	find models/saved/ -name "*.pkl" -mtime +30 -delete

# Logging
view-logs: ## View application logs
	tail -f logs/app.log

clear-logs: ## Clear log files
	rm -f logs/*.log
