/*
# Sentiment Analysis of IMDb Reviews

## ?? Project Goal

Develop a machine learning service capable of automatically classifying movie reviews from IMDb as positive or negative, enabling automated content analysis for businesses (e.g., media companies, recommendation services, monitoring brand sentiment).

## ?? Target Metrics for Production

- **Average service response time:** ? 200 ms
- **Share of failed requests:** ? 1 %
- **Memory/CPU usage:** within specified SLA limits
- **Model quality:** Accuracy ? 90%

## ?? Dataset

**IMDb reviews dataset**
- 50,000 labeled reviews (train/test split)
- Balanced: 25,000 positive, 25,000 negative examples
- Public domain, commonly used as a benchmark for sentiment analysis

## ?? Experiment Plan

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

## ?? Quick Start

### Installation
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
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

### Project Structure
```
MLOps/
??? configs/baseline.yaml       # Training configuration
??? scripts/
?   ??? train.py               # Training script
?   ??? validate.py            # Validation script
??? src/mlops_imdb/            # Core modules
?   ??? config.py              # Config management
?   ??? data.py                # Data preprocessing
?   ??? model.py               # Model creation
?   ??? train.py               # Training logic
??? artefacts/                 # Training outputs
    ??? distilbert-imdb/       # Trained model (HF compatible)
    ??? logs/                  # Training logs
```

**For detailed setup instructions, see [SETUP.md](SETUP.md)**

---

*Repository includes training scripts, configs, and model artifacts.*
*/
