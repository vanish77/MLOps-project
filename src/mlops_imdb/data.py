"""
Module for loading and preprocessing IMDb data.
"""

import logging
import re
from typing import Dict, Optional

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def _basic_clean(text: str) -> str:
    """
    Basic text cleaning from HTML tags and extra whitespace.

    Args:
        text: Original text

    Returns:
        Cleaned text
    """
    # Remove HTML tags
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<.*?>", " ", text)
    # Remove multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_prepare_dataset(
    dataset_name: str = "imdb",
    cache_dir: Optional[str] = None,
    val_size: float = 0.1,
    remove_html: bool = True,
) -> DatasetDict:
    """
    Load IMDb dataset and prepare it for training.

    Args:
        dataset_name: Dataset name on Hugging Face Hub
        cache_dir: Directory for dataset caching
        val_size: Fraction of data for validation from train
        remove_html: Whether to remove HTML tags from text

    Returns:
        DatasetDict with train/validation/test splits
    """
    logger.info("Loading dataset '%s'...", dataset_name)
    raw = load_dataset(dataset_name, cache_dir=cache_dir)

    # IMDb already has train/test. Split train into train/validation.
    logger.info("Splitting train into train/validation with val_size=%.3f", val_size)
    split = raw["train"].train_test_split(test_size=val_size, seed=42, stratify_by_column="label")
    ds = DatasetDict(train=split["train"], validation=split["test"], test=raw["test"])

    if remove_html:
        logger.info("Applying basic HTML cleanup...")
        ds = ds.map(lambda x: {"text": _basic_clean(x["text"])}, batched=False, desc="Cleaning text")

    logger.info("Dataset prepared: train=%d, val=%d, test=%d", len(ds["train"]), len(ds["validation"]), len(ds["test"]))
    return ds


def tokenize_dataset(
    ds: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
) -> DatasetDict:
    """
    Tokenize dataset for model.

    Args:
        ds: Dataset with texts
        tokenizer: Model tokenizer
        max_length: Maximum sequence length

    Returns:
        Tokenized dataset
    """
    logger.info("Tokenizing dataset with max_length=%d", max_length)

    def _tok(batch: Dict[str, list]):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,  # dynamic padding in dataloader
            max_length=max_length,
        )

    # Remove all columns except label after tokenization
    columns_to_remove = [c for c in ds["train"].column_names if c not in {"label"}]
    tokenized = ds.map(_tok, batched=True, remove_columns=columns_to_remove, desc="Tokenizing")

    return tokenized
