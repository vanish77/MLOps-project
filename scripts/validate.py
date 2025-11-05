#!/usr/bin/env python3
"""
Script for validating trained model on examples.

Usage:
    python scripts/validate.py --model-path artefacts/distilbert-imdb
    python scripts/validate.py --model-path artefacts/distilbert-imdb --examples "Great movie!" "Terrible film"
"""
import argparse
import logging
import sys
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_EXAMPLES = [
    "This movie was absolutely amazing! I loved every minute of it.",
    "Terrible film, waste of time. Would not recommend.",
    "Not bad, but could have been better.",
    "One of the best movies I've ever seen!",
    "Boring and predictable. Very disappointed.",
]


def predict(model, tokenizer, texts: List[str], max_length: int = 256):
    """
    Predict sentiment for list of texts.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        texts: List of texts for classification
        max_length: Maximum sequence length
        
    Returns:
        List of predictions and probabilities
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for text in texts:
            # Tokenization
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Inference
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prediction = logits.argmax(-1).item()
            confidence = probs[0, prediction].item()
            
            results.append({
                "text": text,
                "prediction": "Positive" if prediction == 1 else "Negative",
                "confidence": confidence,
                "label": prediction
            })
    
    return results


def main():
    """Main function for model validation."""
    parser = argparse.ArgumentParser(
        description="Validate trained IMDb sentiment model on examples."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--examples",
        nargs="+",
        default=None,
        help="Custom examples to test (if not provided, uses default examples)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Model Validation Script")
    logger.info("=" * 70)
    
    # Load model and tokenizer
    logger.info("Loading model from %s...", args.model_path)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        sys.exit(1)
    
    # Choose examples
    examples = args.examples if args.examples else DEFAULT_EXAMPLES
    logger.info("Testing on %d examples", len(examples))
    logger.info("-" * 70)
    
    # Predictions
    try:
        results = predict(model, tokenizer, examples, args.max_length)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        sys.exit(1)
    
    # Output results
    logger.info("Results:")
    logger.info("=" * 70)
    
    for i, result in enumerate(results, 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"  Text: {result['text']}")
        logger.info(f"  Prediction: {result['prediction']}")
        logger.info(f"  Confidence: {result['confidence']:.2%}")
    
    logger.info("=" * 70)
    logger.info("Validation completed successfully!")
    
    # Brief statistics
    positive_count = sum(1 for r in results if r["label"] == 1)
    negative_count = len(results) - positive_count
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    
    logger.info("\nStatistics:")
    logger.info(f"  Positive predictions: {positive_count}/{len(results)}")
    logger.info(f"  Negative predictions: {negative_count}/{len(results)}")
    logger.info(f"  Average confidence: {avg_confidence:.2%}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
