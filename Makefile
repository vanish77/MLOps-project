.PHONY: help install test lint format clean train

help:
	@echo "Available commands:"
	@echo "  make install    - Install all dependencies"
	@echo "  make test       - Run all tests with coverage"
	@echo "  make lint       - Run code quality checks"
	@echo "  make format     - Format code with black and isort"
	@echo "  make clean      - Clean build artifacts and cache"
	@echo "  make train      - Run training with baseline config"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -x

lint:
	flake8 src/ scripts/ --count --statistics
	black --check src/ scripts/ tests/
	isort --check-only src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml

train:
	python scripts/train.py --config configs/baseline.yaml

validate:
	python scripts/validate.py --model-path artefacts/distilbert-imdb

