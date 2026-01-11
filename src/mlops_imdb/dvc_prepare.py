"""
DVC Stage: Prepare - Data preprocessing.

Downloads and preprocesses IMDb dataset.
"""
import json
import logging
import os
import sys
from pathlib import Path

# Add root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.mlops_imdb.data import load_and_prepare_dataset
from src.mlops_imdb.validation import run_all_validations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Prepare stage: download and preprocess dataset."""
    logger.info("=" * 60)
    logger.info("DVC Stage: PREPARE - Data Preprocessing")
    logger.info("=" * 60)

    # Input: config or use defaults
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare dataset
    logger.info("Loading IMDb dataset...")
    ds = load_and_prepare_dataset(
        dataset_name="imdb",
        cache_dir=None,
        val_size=0.1,
        remove_html=True,
    )

    # Validate dataset
    logger.info("Validating dataset...")
    stats = run_all_validations(ds, num_labels=2)

    # Save processed dataset info
    dataset_info = {
        "num_train": len(ds["train"]),
        "num_validation": len(ds["validation"]),
        "num_test": len(ds["test"]),
        "statistics": stats,
    }

    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    logger.info("Dataset info saved to %s", info_path)
    logger.info("Train: %d, Validation: %d, Test: %d", len(ds["train"]), len(ds["validation"]), len(ds["test"]))

    # Save metadata for DVC tracking
    metadata_path = output_dir / "prepare_metadata.json"
    metadata = {
        "stage": "prepare",
        "dataset_name": "imdb",
        "output_dir": str(output_dir),
        "statistics": stats,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save dataset stats for DVC plots
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 60)
    logger.info("PREPARE stage completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

