# Sentiment Analysis of IMDb Reviews

## Project Goal

Develop a machine learning service capable of automatically classifying movie reviews from IMDb as positive or negative, enabling automated content analysis for businesses (e.g., media companies, recommendation services, monitoring brand sentiment).

## Target Metrics for Production

- **Average service response time:** ? 200 ms
- **Share of failed requests:** ? 1%
- **Memory/CPU usage:** within specified SLA limits
- **Model quality:** Accuracy ? 90%

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
   - Measure response time and resource usage during inference (target average ? 200 ms).

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
+-- example_inference.py           # Inference usage example
|
+-- configs/
|   +-- baseline.yaml              # Training configuration
|
+-- scripts/
|   +-- train.py                   # Training script
|   +-- validate.py                # Validation script
|   +-- upload_to_hub.py           # HF Hub upload script
|
+-- src/
|   +-- mlops_imdb/
|       +-- __init__.py
|       +-- config.py              # Configuration management
|       +-- data.py                # Data loading and preprocessing
|       +-- model.py               # Model creation
|       +-- train.py               # Training logic
|
+-- artefacts/                     # Created during training
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

- **Neural network architecture:** `src/mlops_imdb/model.py`
  - DistilBERT-based model for binary classification
  - Pretrained weights from Hugging Face

- **Training execution:** `scripts/train.py`
  - CLI interface with argument parsing
  - Config-based training
  - Parameter overrides support

- **Model saving:** `src/mlops_imdb/train.py`
  - Saves using `save_pretrained()` method
  - Hugging Face compatible format
  - Stored in `artefacts/distilbert-imdb/`

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

### Model Performance

Baseline results on test set:
- **Accuracy:** 90.88% (target: ? 90%)
- **Precision:** 89.98%
- **Recall:** 92.00%
- **F1-Score:** 90.98%

### Reproducibility

- Fixed random seed (42)
- All dependencies with versions in `requirements.txt`
- Training configuration saved with model
- Complete training logs

---

## Configuration

Key parameters in `configs/baseline.yaml`:

```yaml
seed: 42

data:
  dataset_name: imdb
  max_length: 256
  val_size: 0.1

model:
  pretrained_name: distilbert-base-uncased
  num_labels: 2

training:
  per_device_train_batch_size: 16
  learning_rate: 2e-5
  num_train_epochs: 1
  fp16: false  # Set to false for Apple Silicon (M1/M2)
```

---

## Upload to Hugging Face Hub (Optional)

To share your trained model:

```bash
# Login to HF
huggingface-cli login

# Upload model
python scripts/upload_to_hub.py \
  --model-path artefacts/distilbert-imdb \
  --repo-name YOUR_USERNAME/distilbert-imdb-sentiment
```

Then use it directly from the Hub:

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", 
                     model="YOUR_USERNAME/distilbert-imdb-sentiment")
result = classifier("This movie was great!")
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.44+
- See `requirements.txt` for complete list

## Platform Notes

- **CPU:** Works on any platform, training takes ~30-45 minutes
- **Apple Silicon (M1/M2):** Automatically uses MPS acceleration, ~10-15 minutes
- **NVIDIA GPU:** Set `fp16: true` in config for faster training

---

## License

This project is for educational purposes.

## Author

MLOps course project - Sentiment Analysis on IMDb dataset
