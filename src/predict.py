#!/usr/bin/env python3
"""
Prediction script for Docker container.

Loads trained model and performs batch inference on input CSV file.
Saves predictions to output CSV file.

Usage:
    python -m src.predict --input_path /data/input.csv --output_path /data/predictions.csv
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Default model path (can be overridden)
DEFAULT_MODEL_PATH = Path("/app/models/baseline")
MAX_LENGTH = 256


def load_model(model_path: Path):
    """
    Load trained model and tokenizer.

    Args:
        model_path: Path to model directory

    Returns:
        Tuple (model, tokenizer)
    """
    logger.info("Loading model from %s...", model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))  # nosec B615
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))  # nosec B615
        logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        raise


def predict_batch(model, tokenizer, texts: list, max_length: int = MAX_LENGTH):
    """
    Perform batch prediction on texts.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        texts: List of texts to predict
        max_length: Maximum sequence length

    Returns:
        List of predictions with sentiment and confidence
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for text in texts:
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Inference
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prediction = logits.argmax(-1).item()
            confidence = probs[0, prediction].item()

            # Map label to text
            sentiment = "positive" if prediction == 1 else "negative"

            results.append(
                {
                    "text": text,
                    "prediction": sentiment,
                    "label": prediction,
                    "confidence": confidence,
                }
            )

    return results


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Predict sentiment for texts in CSV file")
    parser.add_argument("--input_path", required=True, help="Path to input CSV file")
    parser.add_argument("--output_path", required=True, help="Path to output CSV file")
    parser.add_argument("--model_path", default=None, help="Path to model directory (default: /app/models/baseline)")
    parser.add_argument("--text_column", default="text", help="Name of text column in input CSV (default: 'text')")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help=f"Maximum sequence length (default: {MAX_LENGTH})")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    model_path = Path(args.model_path) if args.model_path else DEFAULT_MODEL_PATH

    logger.info("=" * 60)
    logger.info("IMDb Sentiment Classification - Batch Inference")
    logger.info("=" * 60)
    logger.info("Input file: %s", input_path)
    logger.info("Output file: %s", output_path)
    logger.info("Model path: %s", model_path)
    logger.info("Text column: %s", args.text_column)
    logger.info("=" * 60)

    # Validate input file
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Load model
    try:
        model, tokenizer = load_model(model_path)
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        sys.exit(1)

    # Load input data
    logger.info("Loading input data from %s...", input_path)
    try:
        df = pd.read_csv(input_path)
        logger.info("Loaded %d rows from input file", len(df))
    except Exception as e:
        logger.error("Failed to load input file: %s", e)
        sys.exit(1)

    # Validate text column
    if args.text_column not in df.columns:
        logger.error("Text column '%s' not found in CSV. Available columns: %s", args.text_column, list(df.columns))
        sys.exit(1)

    # Extract texts
    texts = df[args.text_column].astype(str).tolist()
    logger.info("Processing %d texts...", len(texts))

    # Perform predictions
    try:
        results = predict_batch(model, tokenizer, texts, max_length=args.max_length)
        logger.info("Predictions completed successfully")
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        sys.exit(1)

    # Create output DataFrame
    output_df = pd.DataFrame(results)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info("Predictions saved to %s", output_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info("  Total predictions: %d", len(results))
    positive_count = sum(1 for r in results if r["prediction"] == "positive")
    negative_count = len(results) - positive_count
    logger.info("  Positive: %d (%.1f%%)", positive_count, 100 * positive_count / len(results))
    logger.info("  Negative: %d (%.1f%%)", negative_count, 100 * negative_count / len(results))
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    logger.info("  Average confidence: %.2f%%", 100 * avg_confidence)
    logger.info("=" * 60)
    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main()
