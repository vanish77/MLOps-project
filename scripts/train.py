#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for IMDb sentiment classification model.

Usage:
    python scripts/train.py --config configs/baseline.yaml
    python scripts/train.py --config configs/baseline.yaml --verbose
    python scripts/train.py --config configs/baseline.yaml -o training.learning_rate=3e-5
"""
import argparse
import logging
import os
import sys
from typing import Dict, List

# Add root directory to PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.mlops_imdb.train import run_training  # noqa: E402


def parse_overrides(pairs: List[str]) -> Dict[str, str]:
    """
    Convert list of key=value pairs to dictionary.

    Args:
        pairs: List of strings like ["training.learning_rate=3e-5", "data.max_length=128"]

    Returns:
        Dictionary of overrides

    Raises:
        ValueError: If format is incorrect
    """
    overrides = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"Invalid override '{p}', expected key=value format")
        k, v = p.split("=", 1)
        overrides[k.strip()] = v.strip()
    return overrides


def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(
        description="Train IMDb sentiment classification model using Hugging Face Transformers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config configs/baseline.yaml
  %(prog)s --config configs/baseline.yaml --verbose
  %(prog)s --config configs/baseline.yaml -o training.learning_rate=3e-5 -o data.max_length=128
        """,
    )
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Override config values, e.g. -o training.learning_rate=3e-5",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose (DEBUG level) logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Parse overrides
    try:
        overrides = parse_overrides(args.override)
    except ValueError as e:
        logging.error("Error parsing overrides: %s", e)
        sys.exit(1)

    # Run training
    try:
        run_training(args.config, overrides)
    except Exception as e:
        logging.exception("Training failed with error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
