"""
DVC Stage: Train - Model training.

Trains the model using prepared dataset.
"""
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# Add root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.mlops_imdb.config import load_config
from src.mlops_imdb.train import run_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Train stage: train the model."""
    logger.info("=" * 60)
    logger.info("DVC Stage: TRAIN - Model Training")
    logger.info("=" * 60)

    # Check if prepared data exists
    prepared_info = Path("data/processed/dataset_info.json")
    if not prepared_info.exists():
        raise FileNotFoundError(f"Prepared data not found: {prepared_info}. Run 'prepare' stage first!")

    # Load config
    config_path = "configs/baseline.yaml"
    cfg = load_config(config_path)

    # Override output directory for DVC
    cfg.raw["training"]["output_dir"] = "models/distilbert-imdb"
    cfg.raw["training"]["logging_dir"] = "models/logs"

    # Save modified config temporarily
    temp_config = Path("configs/baseline_dvc.yaml")

    with open(temp_config, "w") as f:
        yaml.dump(cfg.raw, f, default_flow_style=False)

    logger.info("Starting training with DVC output directory: models/distilbert-imdb")
    logger.info("=" * 60)

    try:
        # Run training
        run_training(str(temp_config), overrides=None)
    finally:
        # Clean up temporary config
        if temp_config.exists():
            temp_config.unlink()

    logger.info("=" * 60)
    logger.info("TRAIN stage completed successfully!")
    logger.info("Model saved to: models/distilbert-imdb")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

