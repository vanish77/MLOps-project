"""
DVC Stage: Evaluate - Model evaluation.

Evaluates trained model and saves metrics.
"""
import json
import logging
import sys
from pathlib import Path

# Add root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Evaluate stage: evaluate the trained model."""
    logger.info("=" * 60)
    logger.info("DVC Stage: EVALUATE - Model Evaluation")
    logger.info("=" * 60)

    # Check if model exists
    model_path = Path("models/distilbert-imdb")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run 'train' stage first!")

    metrics_path = model_path / "metrics_test.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found: {metrics_path}. Model training may have failed.")

    # Load metrics
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Create evaluation summary
    eval_summary = {
        "stage": "evaluate",
        "model_path": str(model_path),
        "metrics": metrics,
        "status": "passed" if metrics.get("eval_accuracy", 0) >= 0.90 else "failed",
    }

    # Save evaluation summary
    output_path = Path("models/evaluation_summary.json")
    with open(output_path, "w") as f:
        json.dump(eval_summary, f, indent=2)

    logger.info("Evaluation Summary:")
    logger.info("  Accuracy: %.4f", metrics.get("eval_accuracy", 0))
    logger.info("  Precision: %.4f", metrics.get("eval_precision", 0))
    logger.info("  Recall: %.4f", metrics.get("eval_recall", 0))
    logger.info("  F1-Score: %.4f", metrics.get("eval_f1", 0))
    logger.info("  Status: %s", eval_summary["status"])
    logger.info("=" * 60)
    logger.info("EVALUATE stage completed successfully!")
    logger.info("Evaluation summary saved to: %s", output_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

