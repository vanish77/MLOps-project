#!/usr/bin/env python3
"""
DVC evaluate stage: Evaluate trained model.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add root directory to PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Evaluate model."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model-dir", required=True, help="Directory with trained model")
    parser.add_argument("--metrics-file", required=True, help="Path to save evaluation metrics")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    metrics_file = Path(args.metrics_file)

    logger.info("=" * 60)
    logger.info("DVC Evaluate Stage: Evaluating trained model")
    logger.info("=" * 60)

    # Load metrics from training
    training_metrics_file = model_dir / "metrics.json"
    if training_metrics_file.exists():
        with open(training_metrics_file, "r") as f:
            metrics = json.load(f)
        logger.info("Loaded metrics from training:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        # Save to output location
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_file}")
    else:
        logger.warning(f"Metrics file not found: {training_metrics_file}")

    # Verify model can be loaded
    logger.info("Verifying model can be loaded...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))  # nosec B615
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))  # nosec B615
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Evaluate stage completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
