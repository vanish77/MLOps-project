"""
Tests for configuration module.
"""

from pathlib import Path

import pytest

from src.mlops_imdb.config import _apply_override, _maybe_cast, load_config


class TestConfigLoading:
    """Tests for config loading."""

    def test_load_valid_config(self, temp_config_file):
        """Test loading valid config file."""
        cfg = load_config(temp_config_file)

        assert cfg.seed == 42
        assert cfg.data["dataset_name"] == "imdb"
        assert cfg.model["pretrained_name"] == "distilbert-base-uncased"
        assert cfg.training["per_device_train_batch_size"] == 2

    def test_config_properties(self, temp_config_file):
        """Test config properties."""
        cfg = load_config(temp_config_file)

        assert isinstance(cfg.data, dict)
        assert isinstance(cfg.model, dict)
        assert isinstance(cfg.training, dict)
        assert isinstance(cfg.seed, int)

    def test_load_with_overrides(self, temp_config_file):
        """Test loading config with overrides."""
        overrides = {
            "training.learning_rate": "3e-5",
            "data.max_length": "256",
        }

        cfg = load_config(temp_config_file, overrides)

        assert cfg.training["learning_rate"] == 3e-5
        assert cfg.data["max_length"] == 256

    def test_load_nonexistent_file(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")


class TestOverrides:
    """Tests for config overrides."""

    def test_apply_simple_override(self):
        """Test applying simple override."""
        cfg = {"training": {"lr": 1e-5}}
        _apply_override(cfg, "training.lr", "2e-5")

        assert cfg["training"]["lr"] == 2e-5

    def test_apply_nested_override(self):
        """Test applying nested override."""
        cfg = {}
        _apply_override(cfg, "a.b.c", "value")

        assert cfg["a"]["b"]["c"] == "value"

    def test_apply_override_creates_missing_keys(self):
        """Test that override creates missing keys."""
        cfg = {"existing": "value"}
        _apply_override(cfg, "new.key", "123")

        assert cfg["new"]["key"] == 123


class TestTypeCasting:
    """Tests for type casting in configs."""

    def test_cast_integer(self):
        """Test casting string to integer."""
        assert _maybe_cast("42") == 42
        assert _maybe_cast("0") == 0
        assert _maybe_cast("-10") == -10

    def test_cast_float(self):
        """Test casting string to float."""
        assert _maybe_cast("3.14") == 3.14
        assert _maybe_cast("2e-5") == 2e-5
        assert _maybe_cast("1.0") == 1.0

    def test_cast_boolean(self):
        """Test casting string to boolean."""
        assert _maybe_cast("true") is True
        assert _maybe_cast("True") is True
        assert _maybe_cast("TRUE") is True
        assert _maybe_cast("false") is False
        assert _maybe_cast("False") is False
        assert _maybe_cast("FALSE") is False

    def test_cast_string(self):
        """Test that non-numeric strings remain strings."""
        assert _maybe_cast("hello") == "hello"
        assert _maybe_cast("dataset_name") == "dataset_name"


class TestConfigEdgeCases:
    """Tests for edge cases in configuration."""

    def test_empty_config(self):
        """Test handling of empty config."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        cfg = load_config(temp_path)
        assert cfg.raw == {} or cfg.raw is None

    def test_override_with_dots_in_value(self):
        """Test override with dots in the value."""
        cfg = {"model": {}}
        _apply_override(cfg, "model.name", "distilbert-base-uncased")

        assert cfg["model"]["name"] == "distilbert-base-uncased"
