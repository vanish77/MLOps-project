# Sentiment Analysis of IMDb Reviews

## Project Goal

Develop a machine learning service capable of automatically classifying movie reviews from IMDb as positive or negative, enabling automated content analysis for businesses (e.g., media companies, recommendation services, monitoring brand sentiment).

## Target Metrics for Production

- **Average service response time:** <= 200 ms
- **Share of failed requests:** <= 1%
- **Memory/CPU usage:** within specified SLA limits
- **Model quality:** Accuracy >= 90%

## Dataset

**IMDb reviews dataset**
- 50,000 labeled reviews (train/test split)
- Balanced: 25,000 positive, 25,000 negative examples
- Public domain, commonly used as a benchmark for sentiment analysis

## Experiment Plan

1. **Baseline:**  
   - Fine-tune the pretrained `distilbert-base-uncased` model from Hugging Face Transformers on the IMDb dataset using default parameters.

2. **Data Handling:**  
   - Explore text preprocessing strategies (e.g., basic cleaning, tokenization).

3. **Hyperparameter Tuning:**  
   - Experiment with batch size, learning rate, number of epochs to improve results.

4. **Metrics Measurement:**  
   - Evaluate accuracy, precision, recall, and F1-score.  
   - Measure response time and resource usage during inference (target average <= 200 ms).

5. **Robustness Tests:**  
   - Check model performance on edge cases (sarcasm, very short/long reviews).

6. **Deployment Preparation:**  
   - Containerize the application, expose a simple API.  
   - Monitor performance, failed requests, and log predictions.

---

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Training

```bash
# Basic training with default config
python scripts/train.py --config configs/baseline.yaml

# With verbose logging
python scripts/train.py --config configs/baseline.yaml --verbose

# Override config parameters
python scripts/train.py --config configs/baseline.yaml \
  -o training.learning_rate=3e-5 \
  -o training.num_train_epochs=3
```

### Validation

```bash
# Test model on examples
python scripts/validate.py --model-path artefacts/distilbert-imdb

# Custom examples
python scripts/validate.py --model-path artefacts/distilbert-imdb \
  --examples "Amazing movie!" "Terrible film."
```

### Inference Example

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load trained model
model = AutoModelForSequenceClassification.from_pretrained("./artefacts/distilbert-imdb")
tokenizer = AutoTokenizer.from_pretrained("./artefacts/distilbert-imdb")

# Make prediction
text = "This movie was amazing!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()
print(f"Sentiment: {'Positive' if prediction == 1 else 'Negative'}")
```

---

## Project Structure

```
MLOps/
|
+-- README.md                      # Project documentation
+-- requirements.txt               # Python dependencies
+-- Makefile                       # Convenience commands
+-- pytest.ini                     # Pytest configuration
+-- pyproject.toml                 # Tool configurations (black, isort)
+-- .flake8                        # Flake8 linting rules
+-- .gitignore                     # Git ignore rules
+-- example_inference.py           # Inference usage example
+-- TESTING.md                     # Testing guide
+-- TEST_RESULTS.md                # Test results summary
|
+-- .github/
|   +-- workflows/
|       +-- ci.yml                 # CI/CD pipeline (GitHub Actions)
|
+-- configs/
|   +-- baseline.yaml              # Training configuration
|
+-- scripts/
|   +-- train.py                   # Training script
|   +-- validate.py                # Model validation script
|   +-- upload_to_hub.py           # HF Hub upload script
|
+-- src/
|   +-- mlops_imdb/
|       +-- __init__.py
|       +-- config.py              # Configuration management
|       +-- data.py                # Data loading and preprocessing
|       +-- model.py               # Model creation
|       +-- train.py               # Training logic
|       +-- validation.py          # Data validation functions
|       +-- inference.py           # Prediction postprocessing for API
|
+-- tests/                         # Test suite
|   +-- __init__.py
|   +-- conftest.py                # Pytest fixtures and configuration
|   +-- test_config.py             # Configuration tests
|   +-- test_data.py               # Data preprocessing tests
|   +-- test_inference.py          # Inference and postprocessing tests
|   +-- test_validation.py         # Data validation tests
|
+-- artefacts/                     # Created during training (not in git)
    +-- distilbert-imdb/           # Trained model (HF compatible)
    |   +-- model.safetensors      # Model weights
    |   +-- config.json            # Model config
    |   +-- tokenizer.json         # Tokenizer
    |   +-- vocab.txt              # Vocabulary
    |   +-- metrics_test.json      # Test metrics
    |   +-- training_config.yaml   # Training config copy
    |
    +-- logs/
        +-- train.log              # Training logs
```

---

## Implementation Details

### Requirements Implemented

- **Data loading and preprocessing:** `src/mlops_imdb/data.py`
  - IMDb dataset loading via Hugging Face datasets
  - HTML tag removal
  - Train/validation/test split
  - Tokenization

- **Data validation:** `src/mlops_imdb/validation.py`
  - Dataset structure and schema validation
  - Data type and range checks
  - Text quality verification
  - Dataset balance analysis
  - Integrated into training pipeline

- **Neural network architecture:** `src/mlops_imdb/model.py`
  - DistilBERT-based model for binary classification
  - Pretrained weights from Hugging Face
  - Hugging Face compatible format

- **Training execution:** `scripts/train.py`
  - CLI interface with argument parsing
  - Config-based training
  - Parameter overrides support
  - Automatic data validation

- **Model saving:** `src/mlops_imdb/train.py`
  - Saves using `save_pretrained()` method
  - Hugging Face compatible format
  - Stored in `artefacts/distilbert-imdb/`

- **Inference and postprocessing:** `src/mlops_imdb/inference.py`
  - Prediction postprocessing for API responses
  - Logits to probabilities conversion
  - Label to text mapping
  - Confidence scoring
  - Batch processing support

- **Validation and metrics:** Test set evaluation
  - Accuracy, Precision, Recall, F1-score
  - Metrics output to console and JSON file
  - Example predictions via `scripts/validate.py`

- **Configuration file:** `configs/baseline.yaml`
  - All training parameters (LR, batch size, epochs, etc.)
  - Model architecture settings
  - Data preprocessing options
  - Random seed for reproducibility

- **CLI with arguments:** `scripts/train.py`
  - `--config` to specify config file
  - `--override` / `-o` for parameter overrides
  - `--verbose` for detailed logging

- **Logging:** Python `logging` module
  - Console output (stdout)
  - File logging (`artefacts/logs/train.log`)
  - Detailed step-by-step logs

- **Testing:** Comprehensive test suite (`tests/`)
  - 73+ unit tests covering all modules
  - Data validation tests
  - Inference pipeline tests
  - Preprocessing tests
  - Configuration tests
  - Automated via GitHub Actions CI/CD

- **Code quality:** Automated checks
  - flake8 for linting
  - black for code formatting
  - isort for import sorting
  - Bandit for security scanning

### Model Performance

Baseline results on test set:
- **Accuracy:** 90.88% (target: >= 90%)
- **Precision:** 89.98%
- **Recall:** 92.00%
- **F1-Score:** 90.98%

### Reproducibility

- Fixed random seed (42)
- All dependencies with versions in `requirements.txt`
- Training configuration saved with model
- Complete training logs

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_validation.py -v

# Run only unit tests
pytest -m unit

# Run tests in parallel (faster)
pytest -n auto
```

### Test Coverage

The project includes comprehensive tests for:

- **Data validation** (`test_validation.py`)
  - Dataset structure and schema validation
  - Data types and label ranges
  - Text quality checks
  - Dataset balance verification

- **Data preprocessing** (`test_data.py`)
  - HTML tag removal
  - Text cleaning functions
  - Tokenization pipeline

- **Inference and postprocessing** (`test_inference.py`)
  - Logits to probabilities conversion
  - Label prediction and confidence scores
  - API response formatting
  - Edge cases handling

- **Configuration** (`test_config.py`)
  - Config loading and parsing
  - Parameter overrides
  - Type casting

### Code Quality

```bash
# Format code with black
black src/ scripts/ tests/

# Sort imports with isort
isort src/ scripts/ tests/

# Check code quality with flake8
flake8 src/ scripts/
```

### Continuous Integration

GitHub Actions automatically runs on every commit:
- **Linting**: flake8, black, isort checks
- **Testing**: Unit tests (73+ tests) on Python 3.10
- **Code Coverage**: Coverage reports uploaded to Codecov
- **Security Scan**: Bandit security analysis
- **Integration Tests**: Module imports and script syntax validation

**CI Workflow:** `.github/workflows/ci.yml`

**Status:** All tests passing (73/73) ?

**How to check:** https://github.com/vanish77/MLOps-project/actions
