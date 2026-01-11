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

### Baseline Experiment (Completed)

1. **Baseline:**  
   - Fine-tune the pretrained `distilbert-base-uncased` model from Hugging Face Transformers on the IMDb dataset
   - Hyperparameters: LR=2e-5, batch_size=16, epochs=1
   - **Result:** Accuracy 90.88% (target >= 90% achieved)

2. **Data Handling:**  
   - Text preprocessing: HTML tag removal, tokenization
   - Dataset split: 80% train, 10% validation, 10% test
   - Data validation: Structure, types, quality checks

3. **Metrics Measurement:**  
   - Evaluated accuracy, precision, recall, and F1-score
   - Test set metrics: Accuracy 90.88%, Precision 89.98%, Recall 92.00%, F1 90.98%
   - Metrics saved in `models/distilbert-imdb/metrics_test.json`

### Future Experiments (Planned)

4. **Hyperparameter Tuning:**  
   - Experiment with batch size (8, 16, 32)
   - Learning rate tuning (1e-5, 2e-5, 3e-5, 5e-5)
   - Number of epochs (1, 2, 3)
   - Each experiment versioned via DVC for comparison

5. **Model Architecture Variations:**  
   - Try different models: BERT-base, RoBERTa
   - Compare with DistilBERT baseline
   - Version each model via DVC

6. **Data Augmentation:**  
   - Back-translation
   - Synonym replacement
   - Test impact on model performance

7. **Robustness Tests:**  
   - Edge cases: sarcasm, very short/long reviews
   - Adversarial examples
   - Cross-domain evaluation

8. **Deployment Preparation:**  
   - Containerize the application (Docker)
   - Expose simple API (FastAPI/Flask)
   - Monitor performance, failed requests, and log predictions

**All experiments are versioned via DVC for easy comparison and rollback.**

---

## Quick Start

### Installation and Setup

```bash
# Clone repository
git clone https://github.com/vanish77/MLOps-project.git
cd MLOps-project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Setup DVC and pull data/models
dvc init
dvc pull  # Pull data and models from remote storage
```

### Full Reproduction

**Complete reproduction from scratch (single command):**

```bash
# Clone repository
git clone https://github.com/vanish77/MLOps-project.git
cd MLOps-project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или: .venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Setup DVC and pull data/models
dvc init
dvc remote add -d storage ./dvc_storage  # Local storage (or use Google Drive/S3)
dvc pull  # Download data and models from DVC remote

# Reproduce entire pipeline
dvc repro  # Runs: prepare -> train -> evaluate
```

**Or use single command (if you have Python and DVC installed):**

```bash
git clone https://github.com/vanish77/MLOps-project.git && \
cd MLOps-project && \
python3 -m venv .venv && \
source .venv/bin/activate && \
pip install --upgrade pip && \
pip install -r requirements.txt && \
dvc init && \
dvc remote add -d storage ./dvc_storage && \
dvc pull && \
dvc repro
```

**Note:** If using Google Drive remote, run `bash setup_gdrive_remote.sh` after `dvc init`.

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

## Docker Deployment

### Building Docker Image

Build the Docker image for batch inference:

```bash
# Build Docker image
docker build -t ml-app:v1 .

# Verify image was created
docker images | grep ml-app
```

### Running Docker Container

The container performs batch inference on CSV files. It expects:
- **Input CSV**: Must contain a `text` column with review texts
- **Output CSV**: Will contain `text`, `prediction`, `label`, and `confidence` columns

#### Basic Usage

```bash
# Create directories for input/output data
mkdir -p /tmp/mlops_data

# Prepare input CSV file (example)
cat > /tmp/mlops_data/input.csv << EOF
text
"This movie was absolutely fantastic! I loved every minute of it."
"The film was terrible and boring. Waste of time."
"I thought it was okay, nothing special."
EOF

# Run container with mounted data directory
docker run --rm \
  -v /tmp/mlops_data:/data \
  -v $(pwd)/models:/app/models \
  ml-app:v1 \
  --input_path /data/input.csv \
  --output_path /data/predictions.csv

# Check results
cat /tmp/mlops_data/predictions.csv
```

#### Using DVC Model (Recommended)

If your model is stored in DVC remote storage:

```bash
# Run container with DVC configuration
docker run --rm \
  -v /tmp/mlops_data:/data \
  -v $(pwd)/.dvc:/app/.dvc \
  -v $(pwd)/dvc.yaml:/app/dvc.yaml \
  -v $(pwd)/dvc.lock:/app/dvc.lock \
  -e DVC_REMOTE=storage \
  ml-app:v1 \
  --input_path /data/input.csv \
  --output_path /data/predictions.csv
```

#### Custom Model Path

If your model is in a different location:

```bash
docker run --rm \
  -v /tmp/mlops_data:/data \
  -v /path/to/your/model:/app/models/custom \
  ml-app:v1 \
  --input_path /data/input.csv \
  --output_path /data/predictions.csv \
  --model_path /app/models/custom
```

### Input/Output Format

**Input CSV format:**
```csv
text
"This is a great movie!"
"The film was disappointing."
```

**Output CSV format:**
```csv
text,prediction,label,confidence
"This is a great movie!",positive,1,0.9876
"The film was disappointing.",negative,0,0.9234
```

- `text`: Original input text
- `prediction`: Sentiment label ("positive" or "negative")
- `label`: Numeric label (0 = negative, 1 = positive)
- `confidence`: Prediction confidence score (0.0 to 1.0)

### Script Details

The `src/predict.py` script:
- Loads the trained model from `/app/models/baseline/` (or custom path)
- Reads input CSV file with text reviews
- Performs batch inference using the DistilBERT model
- Saves predictions to output CSV with sentiment labels and confidence scores
- Supports custom text column name via `--text_column` argument
- Configurable maximum sequence length via `--max_length` argument

### Docker Image Contents

- Python 3.12 slim base image
- All dependencies from `requirements.txt`
- DVC for model versioning
- Source code from `src/` directory
- Model pulled via DVC or mounted as volume
- Entrypoint: `python -m src.predict`

---

## TorchServe Deployment (Online Inference)

### Overview

The model is deployed as an online REST API service using TorchServe, PyTorch's model serving framework. This provides production-ready inference with features like multi-model serving, metrics, and logging.

### Building TorchServe Container

#### Step 1: Export Model to TorchScript

```bash
# Activate virtual environment
source .venv/bin/activate

# Export model to TorchScript format
python torchserve/export_model.py \
  --model-path models/baseline \
  --output-dir torchserve/model-artifacts
```

This creates:
- `torchserve/model-artifacts/model.pt` - TorchScript model
- `torchserve/model-artifacts/tokenizer.json` - Tokenizer
- `torchserve/model-artifacts/config.json` - Model configuration

#### Step 2: Create Model Archive (.mar)

```bash
# Navigate to torchserve directory
cd torchserve

# Create .mar archive
bash create_mar.sh
```

This creates `torchserve/model-store/imdb-sentiment.mar` containing:
- Model weights (TorchScript)
- Custom handler (`handler.py`)
- Tokenizer and configuration files
- Dependencies (`requirements.txt`)

#### Step 3: Build Docker Image

```bash
# Build TorchServe image
docker build -t imdb-sentiment-serve:v1 -f torchserve/Dockerfile .
```

#### Step 4: Run Container

```bash
# Run in background
docker run -d \
  --name imdb-sentiment \
  -p 8080:8080 \
  -p 8081:8081 \
  -p 8082:8082 \
  imdb-sentiment-serve:v1

# Check logs
docker logs imdb-sentiment

# Check health
curl http://localhost:8080/ping
```

### API Usage

#### Inference API (Port 8080)

**Single Prediction:**

```bash
# Using JSON file
curl -X POST http://localhost:8080/predictions/imdb-sentiment \
  -H "Content-Type: application/json" \
  -d @torchserve/sample_request.json

# Using inline JSON
curl -X POST http://localhost:8080/predictions/imdb-sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was fantastic! Highly recommended."}'

# Using plain text
curl -X POST http://localhost:8080/predictions/imdb-sentiment \
  -H "Content-Type: text/plain" \
  -d "This movie was terrible and boring."
```

**Response Format:**

```json
[
  {
    "text": "This movie was fantastic! Highly recommended.",
    "sentiment": "positive",
    "label": 1,
    "confidence": 0.9876,
    "probabilities": {
      "negative": 0.0124,
      "positive": 0.9876
    }
  }
]
```

**Batch Prediction:**

```bash
curl -X POST http://localhost:8080/predictions/imdb-sentiment \
  -H "Content-Type: application/json" \
  -d '[
    {"text": "Great movie!"},
    {"text": "Terrible film."},
    {"text": "It was okay."}
  ]'
```

#### Management API (Port 8081)

```bash
# List registered models
curl http://localhost:8081/models

# Get model details
curl http://localhost:8081/models/imdb-sentiment

# Scale workers
curl -X PUT http://localhost:8081/models/imdb-sentiment?min_worker=2&max_worker=4

# Unregister model
curl -X DELETE http://localhost:8081/models/imdb-sentiment
```

#### Metrics API (Port 8082)

```bash
# Get Prometheus metrics
curl http://localhost:8082/metrics
```

### Configuration

TorchServe configuration is defined in `torchserve/config.properties`:

```properties
# Inference address
inference_address=http://0.0.0.0:8080

# Management address
management_address=http://0.0.0.0:8081

# Metrics address
metrics_address=http://0.0.0.0:8082

# Workers configuration
default_workers_per_model=1
max_workers=4

# Request/response limits
max_response_size=6553500
max_request_size=6553500
```

### Custom Handler

The custom handler (`torchserve/handler.py`) implements:

1. **Preprocessing**: Tokenizes input text using DistilBERT tokenizer
2. **Inference**: Runs TorchScript model on tokenized input
3. **Postprocessing**: Converts logits to sentiment labels and confidence scores

Key features:
- Supports JSON and plain text input
- Batch processing
- Detailed logging
- Probability distribution output

### Files Structure

```
torchserve/
??? export_model.py          # Export HuggingFace model to TorchScript
??? handler.py               # Custom TorchServe handler
??? config.properties        # TorchServe configuration
??? requirements.txt         # Handler dependencies
??? create_mar.sh            # Script to create .mar archive
??? Dockerfile               # TorchServe Docker image
??? sample_request.json      # Example API request
??? model-artifacts/         # Exported model files (generated)
?   ??? model.pt
?   ??? config.json
?   ??? tokenizer.json
??? model-store/             # Model archives (generated)
    ??? imdb-sentiment.mar
```

### Troubleshooting

**Container won't start:**
```bash
# Check logs
docker logs imdb-sentiment

# Run in foreground for debugging
docker run --rm -it -p 8080:8080 -p 8081:8081 imdb-sentiment-serve:v1
```

**Model not loading:**
```bash
# Verify .mar file exists
ls -lh torchserve/model-store/

# Check model registration
curl http://localhost:8081/models
```

**Slow inference:**
```bash
# Increase workers
curl -X PUT http://localhost:8081/models/imdb-sentiment?min_worker=2&max_worker=4

# Check metrics
curl http://localhost:8082/metrics | grep -i imdb
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
|   +-- train.py                   # Training script (standalone)
|   +-- train_dvc.py               # Training script (for DVC pipeline)
|   +-- prepare.py                 # Data preparation (DVC stage)
|   +-- evaluate.py                # Model evaluation (DVC stage)
|   +-- validate.py                # Model validation script
|   +-- upload_to_hub.py           # HF Hub upload script
|
+-- data/
|   +-- raw/                       # Raw dataset (downloaded from HF)
|   +-- processed/                 # DVC tracked: processed dataset
|       +-- processed_train.jsonl
|       +-- processed_test.jsonl
|       +-- processed.dvc          # DVC metadata file
|
+-- models/
|   +-- baseline/                  # DVC tracked: trained model
|       +-- model.safetensors
|       +-- config.json
|       +-- metrics.json
|       +-- baseline.dvc           # DVC metadata file
|
+-- metrics/
|   +-- eval.json                  # DVC tracked: evaluation metrics
|
+-- dvc.yaml                       # DVC pipeline definition
+-- dvc.lock                       # DVC pipeline lock file
+-- .dvc/                          # DVC internal files
+-- .dvcignore                     # DVC ignore patterns
|   +-- setup_dvc.sh               # DVC initialization script
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
|       +-- dvc_prepare.py         # DVC stage: data preparation
|       +-- dvc_train.py           # DVC stage: model training
|       +-- dvc_evaluate.py        # DVC stage: evaluation
|
+-- tests/                         # Test suite
|   +-- __init__.py
|   +-- conftest.py                # Pytest fixtures and configuration
|   +-- test_config.py             # Configuration tests
|   +-- test_data.py               # Data preprocessing tests
|   +-- test_inference.py          # Inference and postprocessing tests
|   +-- test_validation.py         # Data validation tests
|
+-- data/                          # Data directory (DVC tracked)
|   +-- raw/                       # Raw datasets (DVC versioned)
|   +-- processed/                 # Processed datasets (DVC outputs)
|       +-- dataset_info.json      # Dataset metadata (DVC tracked)
|       +-- prepare_metadata.json  # Preparation metadata (DVC tracked)
|
+-- models/                        # Models directory (DVC tracked)
|   +-- distilbert-imdb/           # Trained model (DVC versioned)
|       +-- model.safetensors      # Model weights
|       +-- config.json            # Model config
|       +-- tokenizer.json         # Tokenizer
|       +-- vocab.txt              # Vocabulary
|       +-- metrics_test.json      # Test metrics
|       +-- training_config.yaml   # Training config copy
|   +-- evaluation_summary.json    # Evaluation summary (DVC tracked)
|
+-- .dvc/                          # DVC configuration (versioned in git)
|   +-- config                     # DVC config file
|   +-- .gitignore                 # DVC gitignore
|
+-- dvc.yaml                       # DVC pipeline definition
+-- dvc.lock                        # DVC pipeline lock file (auto-generated)
+-- .dvcignore                     # DVC ignore patterns
+-- *.dvc                           # DVC tracking files (in git, point to remote storage)
|
+-- Dockerfile                      # Docker image definition for inference
+-- .dockerignore                   # Docker build ignore patterns
|
+-- artefacts/                     # Temporary artifacts (not in git/DVC)
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

- **Experiment tracking:** MLflow integration
  - Automatic logging of parameters, metrics, and artifacts
  - `mlflow.transformers.autolog()` for Transformers models
  - Local storage (`mlruns/`) or remote (PostgreSQL + S3/MinIO)
  - MLflow UI for visualization and comparison
  - Each training run creates separate experiment with:
    - All hyperparameters
    - Training/validation/test metrics
    - Model artifacts, configs, logs, dvc.lock

### Model Performance

Baseline results on test set:
- **Accuracy:** 90.88% (target: >= 90%)
- **Precision:** 89.98%
- **Recall:** 92.00%
- **F1-Score:** 90.98%

### Data Version Control (DVC)

**Data and Model Storage:**

- **Processed Dataset:** Versioned via DVC, stored in remote storage (not in Git)
  - **Physical location:** DVC remote storage (configured via `dvc remote`)
    - Default: `./dvc_storage` (local) or configured remote (S3/GCS/Azure)
  - **Metadata tracked in Git:** `data/processed.dvc`
  - **Actual data files:** `data/processed/processed_train.jsonl`, `data/processed/processed_test.jsonl`
  - **After `dvc pull`:** Files downloaded to `data/processed/`

- **Trained Models:** Versioned via DVC, stored in remote storage (not in Git)
  - **Physical location:** DVC remote storage
  - **Metadata tracked in Git:** `models/baseline.dvc`
  - **Actual model files:** `models/baseline/model.safetensors`, `models/baseline/config.json`, etc.
  - **After `dvc pull`:** Model downloaded to `models/baseline/`

**Where files physically are:**
- **In Git:** Only metadata files (`.dvc` files, `dvc.yaml`, `dvc.lock`)
- **In DVC remote:** Actual large files (processed dataset, model weights, ~300MB+)
- **After `dvc pull`:** Files downloaded to local directories (`data/processed/`, `models/baseline/`)

**Complete Reproduction Command:**

```bash
# Clone repository
git clone https://github.com/vanish77/MLOps-project.git
cd MLOps-project

# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Setup DVC and pull data/models
dvc init
dvc remote add -d storage ./dvc_storage
dvc pull

# Reproduce entire pipeline
dvc repro
```

**DVC Pipeline Stages (defined in `dvc.yaml`):**

1. **prepare** - Downloads and preprocesses IMDb dataset
   - Script: `scripts/prepare.py`
   - Outputs: `data/processed/processed_train.jsonl`, `data/processed/processed_test.jsonl`
   - Also saves raw splits: `data/processed/raw_train.jsonl`, `data/processed/raw_test.jsonl`

2. **train** - Trains the sentiment classification model
   - Script: `scripts/train_dvc.py`
   - Depends on: `data/processed/processed_*.jsonl`, `configs/baseline.yaml`
   - Outputs: `models/baseline/` (trained model with weights, config, tokenizer)
   - Metrics: `models/baseline/metrics.json`

3. **evaluate** - Evaluates trained model
   - Script: `scripts/evaluate.py`
   - Depends on: `models/baseline/`
   - Outputs: `metrics/eval.json` (evaluation summary)

**Reproducing Experiments:**

```bash
# Full reproduction from scratch
git clone https://github.com/vanish77/MLOps-project.git
cd MLOps-project
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
dvc init && dvc remote add -d storage ./dvc_storage
dvc pull              # Download data and models
dvc repro             # Run entire pipeline: prepare -> train -> evaluate

# Reproduce specific stage
dvc repro prepare     # Only data preparation
dvc repro train       # Only training stage
dvc repro evaluate    # Only evaluation

# Reproduce from specific stage onwards
dvc repro train evaluate  # Train and evaluate (skip prepare)
```

**Switching Versions:**

```bash
# Switch to specific Git commit
git checkout <commit-hash>

# Pull corresponding DVC data/models
dvc pull

# This will automatically pull the correct versions matching the Git commit
```

**DVC Remote Configuration:**

```bash
# View current remote
dvc remote list

# Add local remote storage (default for development)
dvc remote add -d storage ./dvc_storage

# Google Drive (Recommended for personal projects)
# 1. Install Google Drive Desktop: https://www.google.com/drive/download/
# 2. Run setup script:
bash setup_gdrive_remote.sh
# Or manually:
dvc remote add -d gdrive ~/Google\ Drive/MLOps-DVC-Storage

# Or use cloud storage (for production)
dvc remote add -d storage s3://bucket-name/mlops
dvc remote add -d storage gs://bucket-name/mlops

# Push data to remote
dvc push
```

**Google Drive Setup:**

For easy cloud backup and sharing, you can use Google Drive as DVC remote:

1. **Install Google Drive Desktop** (if not already installed)
2. **Run setup script:**
   ```bash
   bash setup_gdrive_remote.sh
   ```
3. **Verify sync:** Check `~/Google Drive/MLOps-DVC-Storage/` folder
4. **Push data:**
   ```bash
   dvc push
   ```

See `DVC_GOOGLE_DRIVE_SETUP.md` for detailed instructions.

For detailed DVC setup instructions, see [DVC_SETUP.md](DVC_SETUP.md).

### Reproducibility

- Fixed random seed (42)
- All dependencies with versions in `requirements.txt`
- Training configuration saved with model
- Complete training logs
- **DVC versioning** for data and models

### Data and Model Versioning (DVC)

**Data Location:**
- **Processed Dataset**: `data/processed/` (DVC tracked)
  - `processed_train.jsonl` - Preprocessed training data
  - `processed_test.jsonl` - Preprocessed test data
  - Actual files stored in DVC remote storage
  - `.dvc` metadata files tracked in Git

**Model Location:**
- **Trained Model**: `models/baseline/` (DVC tracked)
  - Model weights, config, tokenizer
  - Metrics file (`metrics.json`)
  - Actual files stored in DVC remote storage
  - `.dvc` metadata files tracked in Git

**DVC Pipeline:**
- `dvc.yaml` defines pipeline stages:
  1. **prepare**: Download and preprocess IMDb dataset
  2. **train**: Train sentiment classification model
  3. **evaluate**: Evaluate trained model

**Restore Any Version:**
```bash
# Clone repository
git clone <repo-url> && cd MLOps

# Pull specific version
git checkout <commit-hash>
dvc pull

# Or rebuild from scratch
dvc repro
```

See [DVC_SETUP.md](DVC_SETUP.md) for detailed DVC setup instructions.
- **DVC for data and model versioning**
- **Pipeline definition in `dvc.yaml`**
- **Full reproducibility via `dvc repro`**

---

## Experiment Plan

### Baseline Experiment

1. **Data Preparation (prepare stage)**
   - Dataset: IMDb (50k reviews)
   - Preprocessing: HTML removal, tokenization
   - Split: 80% train, 10% validation, 10% test
   - Validation: Structure, types, quality checks

2. **Model Training (train stage)**
   - Architecture: DistilBERT-base-uncased
   - Hyperparameters:
     - Learning rate: 2e-5
     - Batch size: 16 (train), 32 (eval)
     - Epochs: 1-2
     - Optimizer: AdamW with weight decay
   - Output: Trained model in `models/baseline/`

3. **Evaluation (evaluate stage)**
   - Metrics: Accuracy, Precision, Recall, F1-score
   - Target: Accuracy >= 90%
   - Output: Evaluation summary with metrics

### Future Experiments

- **Experiment 2:** Hyperparameter tuning (different LR, batch sizes)
- **Experiment 3:** Different architectures (BERT, RoBERTa)
- **Experiment 4:** Data augmentation strategies
- **Experiment 5:** Longer training (more epochs)

All experiments are versioned via DVC for easy comparison and rollback.

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

### Using Makefile

Convenient commands for common tasks:

```bash
make install    # Install dependencies
make test       # Run all tests with coverage
make lint       # Check code quality
make format     # Auto-format code
make train      # Run training
make validate   # Validate trained model
make clean      # Clean build artifacts
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.44+
- MLflow 2.10+ (experiment tracking)
- DVC 3.0+ (data versioning)
- See `requirements.txt` for complete list

## Platform Notes

- **CPU:** Works on any platform, training takes ~30-45 minutes
- **Apple Silicon (M1/M2):** Automatically uses MPS acceleration, ~10-15 minutes
- **NVIDIA GPU:** Set `fp16: true` in config for faster training

## License

This project is for educational purposes.

## Author

MLOps course project - Sentiment Analysis on IMDb dataset
