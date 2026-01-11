#!/usr/bin/env python3
"""
Export Hugging Face model to TorchScript format for TorchServe.

This script loads the trained DistilBERT model and exports it to TorchScript format.
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def export_model_to_torchscript(model_path: str, output_dir: str):
    """
    Export Hugging Face model to TorchScript format.

    Args:
        model_path: Path to trained Hugging Face model
        output_dir: Directory to save exported model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s...", model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)  # nosec B615
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # nosec B615

    # Set model to evaluation mode
    model.eval()

    logger.info("Exporting model to TorchScript...")

    # Create dummy input for tracing
    dummy_text = "This is a sample movie review for tracing the model."
    dummy_input = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256,
    )

    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            strict=False,
        )

    # Save traced model
    model_file = output_dir / "model.pt"
    torch.jit.save(traced_model, str(model_file))
    logger.info("Model saved to %s", model_file)

    # Save tokenizer
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Tokenizer saved to %s", output_dir)

    # Save model config
    model.config.save_pretrained(str(output_dir))
    logger.info("Config saved to %s", output_dir)

    logger.info("Export completed successfully!")
    logger.info("Files created:")
    logger.info("  - %s/model.pt (TorchScript model)", output_dir)
    logger.info("  - %s/tokenizer.json (Tokenizer)", output_dir)
    logger.info("  - %s/config.json (Model config)", output_dir)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Export Hugging Face model to TorchScript")
    parser.add_argument(
        "--model-path",
        default="models/baseline",
        help="Path to trained Hugging Face model (default: models/baseline)",
    )
    parser.add_argument(
        "--output-dir",
        default="torchserve/model-artifacts",
        help="Directory to save exported model (default: torchserve/model-artifacts)",
    )

    args = parser.parse_args()

    try:
        export_model_to_torchscript(args.model_path, args.output_dir)
    except Exception as e:
        logger.error("Export failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
