#!/usr/bin/env python3
"""
DVC prepare stage: Download and preprocess IMDb dataset.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add root directory to PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import load_dataset

# Import cleaning function
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mlops_imdb.data import _basic_clean

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Download and preprocess IMDb dataset."""
    parser = argparse.ArgumentParser(description="Prepare IMDb dataset for training")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed data")
    parser.add_argument("--remove-html", action="store_true", default=True, help="Remove HTML tags")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DVC Prepare Stage: Downloading and preprocessing IMDb dataset")
    logger.info("=" * 60)

    # Download raw dataset
    logger.info("Downloading IMDb dataset from Hugging Face...")
    raw_dataset = load_dataset("imdb", cache_dir=None)  # nosec B615

    # Save raw train/test splits
    logger.info("Saving raw dataset splits...")
    raw_dataset["train"].to_json(str(output_dir / "raw_train.jsonl"))
    raw_dataset["test"].to_json(str(output_dir / "raw_test.jsonl"))

    # Preprocess: clean HTML tags
    if args.remove_html:
        logger.info("Cleaning HTML tags from texts...")
        processed_train = raw_dataset["train"].map(
            lambda x: {"text": _basic_clean(x["text"])}, batched=False, desc="Cleaning train"
        )
        processed_test = raw_dataset["test"].map(
            lambda x: {"text": _basic_clean(x["text"])}, batched=False, desc="Cleaning test"
        )
    else:
        processed_train = raw_dataset["train"]
        processed_test = raw_dataset["test"]

    # Save processed dataset
    logger.info("Saving processed dataset...")
    processed_train.to_json(str(output_dir / "processed_train.jsonl"))
    processed_test.to_json(str(output_dir / "processed_test.jsonl"))

    logger.info("=" * 60)
    logger.info("Prepare stage completed successfully!")
    logger.info(f"Processed data saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
