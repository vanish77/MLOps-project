"""
Data validation module for IMDb sentiment analysis.
"""
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
from datasets import DatasetDict

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


def validate_dataset_structure(ds: DatasetDict, required_splits: List[str] = None) -> bool:
    """
    Validate dataset structure and required splits.
    
    Args:
        ds: Dataset to validate
        required_splits: List of required split names
        
    Returns:
        True if validation passes
        
    Raises:
        DataValidationError: If validation fails
    """
    if required_splits is None:
        required_splits = ["train", "validation", "test"]
    
    # Check if all required splits exist
    for split in required_splits:
        if split not in ds:
            raise DataValidationError(f"Required split '{split}' not found in dataset")
    
    logger.info("Dataset structure validation passed: splits %s", list(ds.keys()))
    return True


def validate_dataset_schema(ds: DatasetDict, required_columns: List[str] = None) -> bool:
    """
    Validate dataset schema (columns).
    
    Args:
        ds: Dataset to validate
        required_columns: List of required column names
        
    Returns:
        True if validation passes
        
    Raises:
        DataValidationError: If validation fails
    """
    if required_columns is None:
        required_columns = ["text", "label"]
    
    # Check train split as representative
    train_columns = ds["train"].column_names
    
    for col in required_columns:
        if col not in train_columns:
            raise DataValidationError(f"Required column '{col}' not found. Available: {train_columns}")
    
    logger.info("Dataset schema validation passed: columns %s", train_columns)
    return True


def validate_data_types(ds: DatasetDict) -> bool:
    """
    Validate data types in dataset.
    
    Args:
        ds: Dataset to validate
        
    Returns:
        True if validation passes
        
    Raises:
        DataValidationError: If validation fails
    """
    # Check first example from train
    sample = ds["train"][0]
    
    # Text should be string
    if not isinstance(sample["text"], str):
        raise DataValidationError(f"Text should be string, got {type(sample['text'])}")
    
    # Label should be int
    if not isinstance(sample["label"], (int, np.integer)):
        raise DataValidationError(f"Label should be int, got {type(sample['label'])}")
    
    logger.info("Data types validation passed")
    return True


def validate_label_range(ds: DatasetDict, num_labels: int = 2) -> bool:
    """
    Validate that labels are in expected range.
    
    Args:
        ds: Dataset to validate
        num_labels: Expected number of unique labels
        
    Returns:
        True if validation passes
        
    Raises:
        DataValidationError: If validation fails
    """
    for split_name in ["train", "validation", "test"]:
        labels = ds[split_name]["label"]
        unique_labels = set(labels)
        
        # Check label range
        if min(unique_labels) < 0 or max(unique_labels) >= num_labels:
            raise DataValidationError(
                f"Labels in {split_name} out of range [0, {num_labels-1}]. Found: {unique_labels}"
            )
        
        # Check all labels present (for balanced datasets)
        if len(unique_labels) != num_labels:
            logger.warning(
                f"{split_name} split has {len(unique_labels)} unique labels, expected {num_labels}"
            )
    
    logger.info("Label range validation passed")
    return True


def validate_text_quality(ds: DatasetDict, min_text_length: int = 10) -> bool:
    """
    Validate text quality (not empty, reasonable length).
    
    Args:
        ds: Dataset to validate
        min_text_length: Minimum expected text length
        
    Returns:
        True if validation passes
        
    Raises:
        DataValidationError: If validation fails
    """
    for split_name in ["train", "validation", "test"]:
        texts = ds[split_name]["text"]
        
        # Check for empty texts
        empty_count = sum(1 for t in texts if not t or len(t.strip()) == 0)
        if empty_count > 0:
            raise DataValidationError(f"Found {empty_count} empty texts in {split_name}")
        
        # Check minimum length
        too_short = sum(1 for t in texts if len(t) < min_text_length)
        if too_short > len(texts) * 0.01:  # More than 1% too short
            logger.warning(
                f"{split_name} has {too_short} texts shorter than {min_text_length} chars"
            )
    
    logger.info("Text quality validation passed")
    return True


def validate_dataset_balance(ds: DatasetDict, max_imbalance_ratio: float = 2.0) -> bool:
    """
    Validate dataset balance (class distribution).
    
    Args:
        ds: Dataset to validate
        max_imbalance_ratio: Maximum allowed ratio between largest and smallest class
        
    Returns:
        True if validation passes
        
    Raises:
        DataValidationError: If validation fails
    """
    for split_name in ["train", "validation", "test"]:
        labels = ds[split_name]["label"]
        
        # Count class distribution
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique, counts))
        
        # Check imbalance
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > max_imbalance_ratio:
            logger.warning(
                f"{split_name} is imbalanced: {class_dist} (ratio: {imbalance_ratio:.2f})"
            )
        else:
            logger.info(f"{split_name} class distribution: {class_dist}")
    
    return True


def get_dataset_statistics(ds: DatasetDict) -> Dict[str, Any]:
    """
    Get basic statistics about dataset.
    
    Args:
        ds: Dataset to analyze
        
    Returns:
        Dictionary with statistics
    """
    stats = {}
    
    for split_name in ds.keys():
        split_data = ds[split_name]
        
        # Text length statistics
        text_lengths = [len(t) for t in split_data["text"]]
        
        # Label distribution
        labels = split_data["label"]
        unique, counts = np.unique(labels, return_counts=True)
        
        stats[split_name] = {
            "num_examples": len(split_data),
            "text_length_mean": np.mean(text_lengths),
            "text_length_std": np.std(text_lengths),
            "text_length_min": np.min(text_lengths),
            "text_length_max": np.max(text_lengths),
            "label_distribution": dict(zip(unique.tolist(), counts.tolist())),
        }
    
    return stats


def validate_tokenized_data(tokenized_ds: DatasetDict, max_length: int = 512) -> bool:
    """
    Validate tokenized dataset.
    
    Args:
        tokenized_ds: Tokenized dataset
        max_length: Maximum expected token length
        
    Returns:
        True if validation passes
        
    Raises:
        DataValidationError: If validation fails
    """
    # Check required fields
    required_fields = ["input_ids", "attention_mask", "label"]
    
    for split_name in tokenized_ds.keys():
        columns = tokenized_ds[split_name].column_names
        
        for field in required_fields:
            if field not in columns:
                raise DataValidationError(
                    f"Required field '{field}' not found in {split_name}. Available: {columns}"
                )
        
        # Check input_ids shape
        sample = tokenized_ds[split_name][0]
        if len(sample["input_ids"]) > max_length:
            raise DataValidationError(
                f"Token length {len(sample['input_ids'])} exceeds max_length {max_length}"
            )
    
    logger.info("Tokenized data validation passed")
    return True


def run_all_validations(ds: DatasetDict, num_labels: int = 2) -> Tuple[bool, Dict[str, Any]]:
    """
    Run all validations on dataset.
    
    Args:
        ds: Dataset to validate
        num_labels: Expected number of labels
        
    Returns:
        Tuple (validation_passed, statistics)
    """
    logger.info("=" * 60)
    logger.info("Running data validation checks...")
    logger.info("=" * 60)
    
    try:
        # Structure and schema
        validate_dataset_structure(ds)
        validate_dataset_schema(ds)
        
        # Data types and ranges
        validate_data_types(ds)
        validate_label_range(ds, num_labels)
        
        # Quality checks
        validate_text_quality(ds)
        validate_dataset_balance(ds)
        
        # Get statistics
        stats = get_dataset_statistics(ds)
        
        logger.info("=" * 60)
        logger.info("All validation checks passed!")
        logger.info("Dataset statistics:")
        for split_name, split_stats in stats.items():
            logger.info(f"\n{split_name}:")
            for key, value in split_stats.items():
                logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
        
        return True, stats
        
    except DataValidationError as e:
        logger.error("Data validation failed: %s", e)
        raise

