#!/usr/bin/env python3
"""
Script to upload trained model to Hugging Face Hub.

Usage:
    python scripts/upload_to_hub.py --model-path artefacts/distilbert-imdb --repo-name your-username/distilbert-imdb
"""
import argparse
import logging
import sys

from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def upload_model(model_path: str, repo_name: str):
    """
    Upload model to Hugging Face Hub.

    Args:
        model_path: Local path to trained model
        repo_name: Repository name on HF Hub (username/model-name)
    """
    logger.info("=" * 70)
    logger.info("Uploading Model to Hugging Face Hub")
    logger.info("=" * 70)

    # Load model and tokenizer
    logger.info("Loading model from %s...", model_path)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        sys.exit(1)

    # Upload to Hub
    logger.info("Uploading to %s...", repo_name)
    try:
        model.push_to_hub(repo_name, commit_message="Upload IMDb sentiment classification model (DistilBERT)")
        tokenizer.push_to_hub(repo_name, commit_message="Upload tokenizer")
        logger.info("=" * 70)
        logger.info("Upload successful!")
        logger.info("Model available at: https://huggingface.co/%s", repo_name)
        logger.info("=" * 70)
    except Exception as e:
        logger.error("Upload failed: %s", e)
        logger.info("Make sure you are logged in: huggingface-cli login")
        sys.exit(1)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Upload trained model to Hugging Face Hub.")
    parser.add_argument("--model-path", required=True, help="Path to the trained model directory")
    parser.add_argument("--repo-name", required=True, help="Repository name on HF Hub (username/model-name)")

    args = parser.parse_args()
    upload_model(args.model_path, args.repo_name)


if __name__ == "__main__":
    main()
