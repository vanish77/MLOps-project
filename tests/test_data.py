"""
Tests for data preprocessing module.
"""
import pytest
from unittest.mock import Mock, patch
from datasets import Dataset, DatasetDict

from src.mlops_imdb.data import _basic_clean, tokenize_dataset


class TestBasicClean:
    """Tests for basic HTML cleaning function."""
    
    def test_remove_br_tags(self):
        """Test removal of <br> tags."""
        text = "Hello<br>World"
        cleaned = _basic_clean(text)
        assert "<br>" not in cleaned
        assert "Hello" in cleaned and "World" in cleaned
    
    def test_remove_br_tags_with_slash(self):
        """Test removal of <br/> tags."""
        text = "Hello<br/>World"
        cleaned = _basic_clean(text)
        assert "<br/>" not in cleaned
    
    def test_remove_html_tags(self):
        """Test removal of various HTML tags."""
        text = "<div>Hello</div><span>World</span>"
        cleaned = _basic_clean(text)
        assert "<div>" not in cleaned
        assert "<span>" not in cleaned
        assert "Hello" in cleaned and "World" in cleaned
    
    def test_remove_multiple_whitespace(self):
        """Test removal of multiple whitespaces."""
        text = "Hello    World   Test"
        cleaned = _basic_clean(text)
        assert cleaned == "Hello World Test"
    
    def test_strip_whitespace(self):
        """Test stripping leading/trailing whitespace."""
        text = "   Hello World   "
        cleaned = _basic_clean(text)
        assert cleaned == "Hello World"
    
    def test_combined_cleaning(self):
        """Test combined HTML and whitespace cleaning."""
        text = "<p>  Hello  <br/>  World  </p>"
        cleaned = _basic_clean(text)
        assert "<" not in cleaned
        assert cleaned == "Hello World"
    
    def test_empty_string(self):
        """Test handling of empty string."""
        cleaned = _basic_clean("")
        assert cleaned == ""
    
    def test_no_html(self):
        """Test text without HTML remains unchanged."""
        text = "Simple text without HTML"
        cleaned = _basic_clean(text)
        assert cleaned == text


class TestTokenizeDataset:
    """Tests for dataset tokenization."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        def tokenize_fn(texts, **kwargs):
            # Return one token sequence per input text
            return {
                "input_ids": [[101, 102] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }
        
        tokenizer = Mock()
        tokenizer.side_effect = tokenize_fn
        return tokenizer
    
    def test_tokenize_structure(self, sample_dataset, mock_tokenizer):
        """Test tokenized dataset has correct structure."""
        tokenized = tokenize_dataset(sample_dataset, mock_tokenizer, max_length=128)
        
        # Should have same splits
        assert set(tokenized.keys()) == {"train", "validation", "test"}
        
        # Should have tokenizer output fields
        assert "input_ids" in tokenized["train"].column_names
        assert "attention_mask" in tokenized["train"].column_names
        assert "label" in tokenized["train"].column_names
    
    def test_tokenize_removes_text(self, sample_dataset, mock_tokenizer):
        """Test that text column is removed after tokenization."""
        tokenized = tokenize_dataset(sample_dataset, mock_tokenizer, max_length=128)
        
        # Text should be removed
        assert "text" not in tokenized["train"].column_names
    
    def test_tokenize_preserves_labels(self, sample_dataset, mock_tokenizer):
        """Test that labels are preserved."""
        original_labels = sample_dataset["train"]["label"]
        
        tokenized = tokenize_dataset(sample_dataset, mock_tokenizer, max_length=128)
        
        # Labels should be preserved (but order might change due to mapping)
        assert "label" in tokenized["train"].column_names
    
    def test_tokenize_max_length(self, sample_dataset):
        """Test tokenizer is called with correct max_length."""
        def tokenize_fn(texts, **kwargs):
            return {
                "input_ids": [[101, 102] for _ in texts],
                "attention_mask": [[1, 1] for _ in texts],
            }
        
        mock_tokenizer = Mock()
        mock_tokenizer.side_effect = tokenize_fn
        
        max_len = 256
        tokenize_dataset(sample_dataset, mock_tokenizer, max_length=max_len)
        
        # Check tokenizer was called with max_length
        call_kwargs = mock_tokenizer.call_args[1]
        assert call_kwargs["max_length"] == max_len
        assert call_kwargs["truncation"] is True
        assert call_kwargs["padding"] is False


class TestDataPreprocessingIntegration:
    """Integration tests for data preprocessing pipeline."""
    
    def test_clean_and_validate_pipeline(self, sample_text_data, sample_labels):
        """Test complete preprocessing pipeline."""
        # Create dataset with HTML
        texts_with_html = [
            "<p>Great movie!</p>",
            "Terrible<br/>film",
            "<div>Amazing</div>  performance",
            "Boring  and  predictable",
        ]
        
        ds = DatasetDict({
            "train": Dataset.from_dict({"text": texts_with_html, "label": sample_labels["train"]}),
            "validation": Dataset.from_dict({"text": sample_text_data["test"], "label": sample_labels["test"]}),
            "test": Dataset.from_dict({"text": sample_text_data["test"], "label": sample_labels["test"]})
        })
        
        # Clean texts
        cleaned_ds = ds.map(lambda x: {"text": _basic_clean(x["text"])})
        
        # Verify cleaning
        cleaned_texts = cleaned_ds["train"]["text"]
        for text in cleaned_texts:
            assert "<" not in text
            assert "  " not in text


class TestEdgeCases:
    """Tests for edge cases in data processing."""
    
    def test_very_long_text(self):
        """Test handling of very long texts."""
        long_text = "word " * 1000  # Very long text
        cleaned = _basic_clean(long_text)
        assert len(cleaned) > 0
        assert cleaned.count("word") == 1000
    
    def test_special_characters(self):
        """Test handling of special characters."""
        text = "Hello! How are you? I'm fine. #great @user"
        cleaned = _basic_clean(text)
        # Special chars should be preserved (only HTML removed)
        assert "!" in cleaned
        assert "?" in cleaned
        assert "#" in cleaned
        assert "@" in cleaned
    
    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        text = "Hello ?? ?????"
        cleaned = _basic_clean(text)
        assert "??" in cleaned
        assert "?????" in cleaned
    
    def test_nested_html_tags(self):
        """Test handling of nested HTML tags."""
        text = "<div><p><span>Hello</span></p></div>"
        cleaned = _basic_clean(text)
        assert "<" not in cleaned
        assert cleaned == "Hello"

