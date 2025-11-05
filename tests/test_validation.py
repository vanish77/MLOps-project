"""
Tests for data validation module.
"""
import pytest
import numpy as np
from datasets import Dataset, DatasetDict

from src.mlops_imdb.validation import (
    DataValidationError,
    validate_dataset_structure,
    validate_dataset_schema,
    validate_data_types,
    validate_label_range,
    validate_text_quality,
    validate_dataset_balance,
    validate_tokenized_data,
    get_dataset_statistics,
)


class TestDatasetStructure:
    """Tests for dataset structure validation."""
    
    def test_valid_structure(self, sample_dataset):
        """Test validation passes for valid dataset structure."""
        assert validate_dataset_structure(sample_dataset)
    
    def test_missing_split(self, sample_dataset):
        """Test validation fails when required split is missing."""
        # Remove test split
        ds = DatasetDict({
            "train": sample_dataset["train"],
            "validation": sample_dataset["validation"]
        })
        
        with pytest.raises(DataValidationError, match="Required split 'test' not found"):
            validate_dataset_structure(ds)
    
    def test_custom_splits(self, sample_dataset):
        """Test validation with custom required splits."""
        assert validate_dataset_structure(
            sample_dataset,
            required_splits=["train", "validation"]
        )


class TestDatasetSchema:
    """Tests for dataset schema validation."""
    
    def test_valid_schema(self, sample_dataset):
        """Test validation passes for valid schema."""
        assert validate_dataset_schema(sample_dataset)
    
    def test_missing_column(self, sample_text_data):
        """Test validation fails when required column is missing."""
        # Dataset with only text, no labels
        ds = DatasetDict({
            "train": Dataset.from_dict({"text": sample_text_data["train"]}),
            "validation": Dataset.from_dict({"text": sample_text_data["test"]}),
            "test": Dataset.from_dict({"text": sample_text_data["test"]})
        })
        
        with pytest.raises(DataValidationError, match="Required column 'label' not found"):
            validate_dataset_schema(ds)


class TestDataTypes:
    """Tests for data type validation."""
    
    def test_valid_types(self, sample_dataset):
        """Test validation passes for valid data types."""
        assert validate_data_types(sample_dataset)
    
    def test_invalid_text_type(self):
        """Test validation fails for invalid text type."""
        ds = DatasetDict({
            "train": Dataset.from_dict({"text": [123, 456], "label": [0, 1]}),
            "validation": Dataset.from_dict({"text": ["text"], "label": [0]}),
            "test": Dataset.from_dict({"text": ["text"], "label": [0]})
        })
        
        with pytest.raises(DataValidationError, match="Text should be string"):
            validate_data_types(ds)
    
    def test_invalid_label_type(self):
        """Test validation fails for invalid label type."""
        ds = DatasetDict({
            "train": Dataset.from_dict({"text": ["text1", "text2"], "label": ["0", "1"]}),
            "validation": Dataset.from_dict({"text": ["text"], "label": [0]}),
            "test": Dataset.from_dict({"text": ["text"], "label": [0]})
        })
        
        with pytest.raises(DataValidationError, match="Label should be int"):
            validate_data_types(ds)


class TestLabelRange:
    """Tests for label range validation."""
    
    def test_valid_labels(self, sample_dataset):
        """Test validation passes for valid label range."""
        assert validate_label_range(sample_dataset, num_labels=2)
    
    def test_out_of_range_labels(self):
        """Test validation fails for out-of-range labels."""
        ds = DatasetDict({
            "train": Dataset.from_dict({"text": ["t1", "t2"], "label": [0, 5]}),
            "validation": Dataset.from_dict({"text": ["t"], "label": [0]}),
            "test": Dataset.from_dict({"text": ["t"], "label": [0]})
        })
        
        with pytest.raises(DataValidationError, match="Labels.*out of range"):
            validate_label_range(ds, num_labels=2)
    
    def test_negative_labels(self):
        """Test validation fails for negative labels."""
        ds = DatasetDict({
            "train": Dataset.from_dict({"text": ["t1", "t2"], "label": [-1, 0]}),
            "validation": Dataset.from_dict({"text": ["t"], "label": [0]}),
            "test": Dataset.from_dict({"text": ["t"], "label": [0]})
        })
        
        with pytest.raises(DataValidationError, match="Labels.*out of range"):
            validate_label_range(ds, num_labels=2)


class TestTextQuality:
    """Tests for text quality validation."""
    
    def test_valid_text(self, sample_dataset):
        """Test validation passes for valid text."""
        assert validate_text_quality(sample_dataset, min_text_length=5)
    
    def test_empty_text(self):
        """Test validation fails for empty text."""
        ds = DatasetDict({
            "train": Dataset.from_dict({"text": ["good text", ""], "label": [0, 1]}),
            "validation": Dataset.from_dict({"text": ["text"], "label": [0]}),
            "test": Dataset.from_dict({"text": ["text"], "label": [0]})
        })
        
        with pytest.raises(DataValidationError, match="empty texts"):
            validate_text_quality(ds)
    
    def test_whitespace_only_text(self):
        """Test validation fails for whitespace-only text."""
        ds = DatasetDict({
            "train": Dataset.from_dict({"text": ["good text", "   "], "label": [0, 1]}),
            "validation": Dataset.from_dict({"text": ["text"], "label": [0]}),
            "test": Dataset.from_dict({"text": ["text"], "label": [0]})
        })
        
        with pytest.raises(DataValidationError, match="empty texts"):
            validate_text_quality(ds)


class TestDatasetBalance:
    """Tests for dataset balance validation."""
    
    def test_balanced_dataset(self, sample_dataset):
        """Test validation passes for balanced dataset."""
        assert validate_dataset_balance(sample_dataset, max_imbalance_ratio=2.0)
    
    def test_imbalanced_dataset(self):
        """Test warning for imbalanced dataset."""
        # 10 positive, 2 negative (ratio 5:1)
        ds = DatasetDict({
            "train": Dataset.from_dict({
                "text": ["text"] * 12,
                "label": [1] * 10 + [0] * 2
            }),
            "validation": Dataset.from_dict({"text": ["t1", "t2"], "label": [0, 1]}),
            "test": Dataset.from_dict({"text": ["t1", "t2"], "label": [0, 1]})
        })
        
        # Should pass but log warning
        assert validate_dataset_balance(ds, max_imbalance_ratio=10.0)


class TestTokenizedData:
    """Tests for tokenized data validation."""
    
    def test_valid_tokenized_data(self, sample_tokenized_data):
        """Test validation passes for valid tokenized data."""
        ds = DatasetDict({
            "train": Dataset.from_dict(sample_tokenized_data),
            "validation": Dataset.from_dict(sample_tokenized_data),
            "test": Dataset.from_dict(sample_tokenized_data)
        })
        
        assert validate_tokenized_data(ds, max_length=512)
    
    def test_missing_input_ids(self, sample_tokenized_data):
        """Test validation fails when input_ids is missing."""
        data = sample_tokenized_data.copy()
        del data["input_ids"]
        
        ds = DatasetDict({
            "train": Dataset.from_dict(data),
            "validation": Dataset.from_dict(data),
            "test": Dataset.from_dict(data)
        })
        
        with pytest.raises(DataValidationError, match="Required field 'input_ids'"):
            validate_tokenized_data(ds)
    
    def test_exceeds_max_length(self):
        """Test validation fails when tokens exceed max length."""
        data = {
            "input_ids": [[1] * 600],  # Too long
            "attention_mask": [[1] * 600],
            "label": [0]
        }
        
        ds = DatasetDict({
            "train": Dataset.from_dict(data),
            "validation": Dataset.from_dict(data),
            "test": Dataset.from_dict(data)
        })
        
        with pytest.raises(DataValidationError, match="Token length.*exceeds max_length"):
            validate_tokenized_data(ds, max_length=512)


class TestDatasetStatistics:
    """Tests for dataset statistics."""
    
    def test_get_statistics(self, sample_dataset):
        """Test statistics calculation."""
        stats = get_dataset_statistics(sample_dataset)
        
        # Check all splits are present
        assert "train" in stats
        assert "validation" in stats
        assert "test" in stats
        
        # Check statistics structure
        for split_stats in stats.values():
            assert "num_examples" in split_stats
            assert "text_length_mean" in split_stats
            assert "text_length_std" in split_stats
            assert "text_length_min" in split_stats
            assert "text_length_max" in split_stats
            assert "label_distribution" in split_stats
        
        # Check label distribution
        assert stats["train"]["label_distribution"] == {0: 2, 1: 2}

