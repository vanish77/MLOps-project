"""
Pytest configuration and fixtures.
"""
import os
import sys
from pathlib import Path

import pytest
import numpy as np
import torch
from datasets import Dataset, DatasetDict

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_text_data():
    """Sample text data for testing."""
    return {
        "train": [
            "This movie was great! I loved it.",
            "Terrible film, waste of time.",
            "Amazing performance by the actors.",
            "Boring and predictable storyline.",
        ],
        "test": [
            "Best movie ever!",
            "Worst movie I've seen.",
        ]
    }


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return {
        "train": [1, 0, 1, 0],  # positive, negative, positive, negative
        "test": [1, 0]
    }


@pytest.fixture
def sample_dataset(sample_text_data, sample_labels):
    """Create a sample DatasetDict for testing."""
    train_ds = Dataset.from_dict({
        "text": sample_text_data["train"],
        "label": sample_labels["train"]
    })
    
    test_ds = Dataset.from_dict({
        "text": sample_text_data["test"],
        "label": sample_labels["test"]
    })
    
    return DatasetDict({
        "train": train_ds,
        "validation": test_ds,  # Use same as test for simplicity
        "test": test_ds
    })


@pytest.fixture
def sample_tokenized_data():
    """Sample tokenized data for testing."""
    return {
        "input_ids": [[101, 2023, 3185, 2001, 2307, 102], [101, 6659, 2143, 102]],
        "attention_mask": [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1]],
        "label": [1, 0]
    }


@pytest.fixture
def sample_logits():
    """Sample model logits for testing."""
    return np.array([
        [0.1, 2.5],   # Strong positive
        [3.0, 0.2],   # Strong negative
        [1.0, 1.1],   # Weak positive
        [0.5, 0.4],   # Weak negative
    ])


@pytest.fixture
def sample_logits_torch():
    """Sample model logits as torch tensor."""
    return torch.tensor([
        [0.1, 2.5],
        [3.0, 0.2],
    ])


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_content = """
seed: 42

data:
  dataset_name: imdb
  val_size: 0.1
  max_length: 128
  remove_html: true

model:
  pretrained_name: distilbert-base-uncased
  num_labels: 2

training:
  output_dir: ./test_output
  per_device_train_batch_size: 2
  learning_rate: 2e-5
  num_train_epochs: 1
  fp16: false
  eval_strategy: epoch
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return str(config_file)


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility in tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

