# Testing Guide

## Overview

This project includes comprehensive testing to ensure code quality, correctness, and reliability. Tests are automatically run on every commit via GitHub Actions CI/CD pipeline.

## Test Structure

```
tests/
??? __init__.py
??? conftest.py              # Pytest fixtures and configuration
??? test_validation.py       # Data validation tests
??? test_inference.py        # Inference and postprocessing tests
??? test_data.py             # Data preprocessing tests
??? test_config.py           # Configuration tests
```

## Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_validation.py

# Run specific test class
pytest tests/test_validation.py::TestDatasetStructure

# Run specific test function
pytest tests/test_validation.py::TestDatasetStructure::test_valid_structure
```

### With Coverage

```bash
# Run tests with coverage report
pytest --cov=src --cov-report=html --cov-report=term

# View HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Test Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Categories

### 1. Data Validation Tests

**File:** `tests/test_validation.py`

Tests for data quality and correctness:
- Dataset structure validation (required splits)
- Schema validation (required columns)
- Data type validation (text strings, numeric labels)
- Label range validation (0-1 for binary classification)
- Text quality checks (no empty texts, minimum length)
- Dataset balance checks (class distribution)
- Tokenized data validation

**Example:**
```python
def test_valid_structure(self, sample_dataset):
    """Test validation passes for valid dataset structure."""
    assert validate_dataset_structure(sample_dataset)
```

### 2. Inference and Postprocessing Tests

**File:** `tests/test_inference.py`

Tests for model output processing:
- Logits to probabilities conversion
- Probability to label conversion
- Label to text mapping (negative/positive)
- Complete prediction processing pipeline
- Batch prediction processing
- Edge cases (equal logits, very confident predictions)
- Input validation (NaN, Inf values)

**Example:**
```python
def test_process_prediction_positive(self, postprocessor):
    """Test full prediction processing for positive sentiment."""
    logits = np.array([[0.1, 2.5]])  # Strong positive
    result = postprocessor.process_prediction(logits)
    
    assert result["sentiment"] == "positive"
    assert result["label"] == 1
    assert result["confidence"] > 0.7
```

### 3. Data Preprocessing Tests

**File:** `tests/test_data.py`

Tests for data cleaning and tokenization:
- HTML tag removal (`<br>`, `<div>`, etc.)
- Whitespace normalization
- Tokenization pipeline
- Edge cases (empty strings, special characters, unicode)

**Example:**
```python
def test_remove_html_tags(self):
    """Test removal of various HTML tags."""
    text = "<div>Hello</div><span>World</span>"
    cleaned = _basic_clean(text)
    assert "<" not in cleaned
    assert "Hello" in cleaned and "World" in cleaned
```

### 4. Configuration Tests

**File:** `tests/test_config.py`

Tests for configuration management:
- Config file loading
- Parameter overrides
- Type casting (string to int/float/bool)
- Nested config handling

## Continuous Integration

### GitHub Actions Workflow

The CI pipeline (`.github/workflows/ci.yml`) includes:

1. **Lint Job**
   - Code style checks with flake8
   - Formatting checks with black
   - Import sorting checks with isort

2. **Test Job**
   - Runs on Python 3.9, 3.10, 3.11
   - Executes all unit tests
   - Generates coverage reports
   - Uploads coverage to Codecov

3. **Integration Test Job**
   - Tests module imports
   - Validates script syntax
   - Checks config loading

4. **Security Job**
   - Runs Bandit security scanner
   - Checks for common security issues

### Triggering CI

CI automatically runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

## Writing New Tests

### Test Fixtures

Use fixtures from `conftest.py`:

```python
def test_my_function(sample_dataset):
    """Test with sample dataset fixture."""
    result = my_function(sample_dataset)
    assert result is not None
```

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Best Practices

1. **One assertion per test** (when possible)
2. **Clear test names** describing what is tested
3. **Use fixtures** for common test data
4. **Test edge cases** and error conditions
5. **Mock external dependencies** (API calls, file I/O)

### Example Test Structure

```python
class TestMyFeature:
    """Tests for my feature."""
    
    def test_normal_case(self):
        """Test normal operation."""
        result = my_function("valid input")
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case handling."""
        result = my_function("")
        assert result is None
    
    def test_error_case(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            my_function("invalid input")
```

## Code Coverage Goals

- **Target:** > 80% coverage for core modules
- **Critical paths:** 100% coverage (data validation, inference)
- **Exclude:** Configuration files, __init__.py files

## Troubleshooting

### Tests Fail Locally But Pass in CI

- Check Python version (CI uses 3.9-3.11)
- Check environment variables
- Ensure clean virtual environment

### Import Errors

```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest
```

### Slow Tests

```bash
# Run tests in parallel
pytest -n auto

# Skip slow tests
pytest -m "not slow"
```

## Integration with Development Workflow

### Pre-commit Hook (Optional)

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
make lint && make test-fast
```

### Before Committing

```bash
# Format code
make format

# Run tests
make test

# Check coverage
pytest --cov=src --cov-report=term-missing
```

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [Coverage.py documentation](https://coverage.readthedocs.io/)

