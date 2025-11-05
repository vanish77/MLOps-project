"""
Tests for inference and prediction postprocessing.
"""

import numpy as np
import pytest
import torch

from src.mlops_imdb.inference import (
    PredictionPostprocessor,
    validate_logits,
    validate_probabilities,
)


class TestPredictionPostprocessor:
    """Tests for PredictionPostprocessor class."""

    @pytest.fixture
    def postprocessor(self):
        """Create postprocessor instance."""
        return PredictionPostprocessor(confidence_threshold=0.7)

    def test_logits_to_probabilities_numpy(self, postprocessor, sample_logits):
        """Test conversion of numpy logits to probabilities."""
        probs = postprocessor.logits_to_probabilities(sample_logits)

        # Check shape
        assert probs.shape == sample_logits.shape

        # Check probabilities sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0)

        # Check range [0, 1]
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_logits_to_probabilities_torch(self, postprocessor, sample_logits_torch):
        """Test conversion of torch logits to probabilities."""
        probs = postprocessor.logits_to_probabilities(sample_logits_torch)

        # Should return numpy array
        assert isinstance(probs, np.ndarray)

        # Check probabilities
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_probabilities_to_label(self, postprocessor):
        """Test conversion of probabilities to label and confidence."""
        # Strong positive (class 1)
        probs = np.array([[0.1, 0.9]])
        label, confidence = postprocessor.probabilities_to_label(probs)

        assert label == 1
        assert np.isclose(confidence, 0.9)

        # Strong negative (class 0)
        probs = np.array([[0.95, 0.05]])
        label, confidence = postprocessor.probabilities_to_label(probs)

        assert label == 0
        assert np.isclose(confidence, 0.95)

    def test_label_to_text(self, postprocessor):
        """Test conversion of numeric labels to text."""
        assert postprocessor.label_to_text(0) == "negative"
        assert postprocessor.label_to_text(1) == "positive"

    def test_label_to_text_invalid(self, postprocessor):
        """Test error handling for invalid labels."""
        with pytest.raises(ValueError, match="Invalid label"):
            postprocessor.label_to_text(2)

        with pytest.raises(ValueError, match="Invalid label"):
            postprocessor.label_to_text(-1)

    def test_process_prediction_positive(self, postprocessor):
        """Test full prediction processing for positive sentiment."""
        logits = np.array([[0.1, 2.5]])  # Strong positive
        result = postprocessor.process_prediction(logits)

        assert result["sentiment"] == "positive"
        assert result["label"] == 1
        assert result["confidence"] > 0.7
        assert "warning" not in result  # High confidence

    def test_process_prediction_negative(self, postprocessor):
        """Test full prediction processing for negative sentiment."""
        logits = np.array([[3.0, 0.2]])  # Strong negative
        result = postprocessor.process_prediction(logits)

        assert result["sentiment"] == "negative"
        assert result["label"] == 0
        assert result["confidence"] > 0.7

    def test_process_prediction_low_confidence(self, postprocessor):
        """Test prediction with low confidence warning."""
        logits = np.array([[0.5, 0.6]])  # Low confidence
        result = postprocessor.process_prediction(logits)

        # Should include warning
        assert "warning" in result
        assert result["confidence"] < 0.7

    def test_process_prediction_with_probabilities(self, postprocessor):
        """Test prediction with probability distribution."""
        logits = np.array([[0.1, 2.5]])
        result = postprocessor.process_prediction(logits, return_probabilities=True)

        assert "probabilities" in result
        assert "negative" in result["probabilities"]
        assert "positive" in result["probabilities"]
        assert np.isclose(result["probabilities"]["negative"] + result["probabilities"]["positive"], 1.0)

    def test_process_batch(self, postprocessor, sample_logits):
        """Test batch prediction processing."""
        results = postprocessor.process_batch(sample_logits)

        # Should return list with same length
        assert len(results) == len(sample_logits)

        # Each result should have required fields
        for result in results:
            assert "sentiment" in result
            assert "confidence" in result
            assert "label" in result

    def test_process_batch_with_probabilities(self, postprocessor, sample_logits):
        """Test batch processing with probabilities."""
        results = postprocessor.process_batch(sample_logits, return_probabilities=True)

        for result in results:
            assert "probabilities" in result


class TestLogitsValidation:
    """Tests for logits validation."""

    def test_valid_logits_numpy(self, sample_logits):
        """Test validation passes for valid numpy logits."""
        assert validate_logits(sample_logits)

    def test_valid_logits_torch(self, sample_logits_torch):
        """Test validation passes for valid torch logits."""
        assert validate_logits(sample_logits_torch)

    def test_wrong_dimensionality(self):
        """Test validation fails for wrong dimensionality."""
        logits_1d = np.array([0.1, 2.5])

        with pytest.raises(ValueError, match="Logits should be 2D"):
            validate_logits(logits_1d)

        logits_3d = np.array([[[0.1, 2.5]]])

        with pytest.raises(ValueError, match="Logits should be 2D"):
            validate_logits(logits_3d)

    def test_shape_mismatch(self, sample_logits):
        """Test validation fails for shape mismatch."""
        with pytest.raises(ValueError, match="Expected shape"):
            validate_logits(sample_logits, expected_shape=(2, 2))

    def test_nan_values(self):
        """Test validation fails for NaN values."""
        logits_with_nan = np.array([[0.1, np.nan]])

        with pytest.raises(ValueError, match="NaN values"):
            validate_logits(logits_with_nan)

    def test_inf_values(self):
        """Test validation fails for Inf values."""
        logits_with_inf = np.array([[0.1, np.inf]])

        with pytest.raises(ValueError, match="Inf values"):
            validate_logits(logits_with_inf)


class TestProbabilitiesValidation:
    """Tests for probabilities validation."""

    def test_valid_probabilities(self):
        """Test validation passes for valid probabilities."""
        probs = np.array([[0.3, 0.7], [0.9, 0.1]])
        assert validate_probabilities(probs)

    def test_out_of_range(self):
        """Test validation fails for out-of-range probabilities."""
        probs_negative = np.array([[-0.1, 1.1]])

        with pytest.raises(ValueError, match="should be in \\[0, 1\\]"):
            validate_probabilities(probs_negative)

        probs_over_one = np.array([[0.5, 1.5]])

        with pytest.raises(ValueError, match="should be in \\[0, 1\\]"):
            validate_probabilities(probs_over_one)

    def test_not_sum_to_one(self):
        """Test validation fails when probabilities don't sum to 1."""
        probs_wrong_sum = np.array([[0.3, 0.5]])  # Sums to 0.8

        with pytest.raises(ValueError, match="should sum to 1"):
            validate_probabilities(probs_wrong_sum)


class TestEdgeCases:
    """Tests for edge cases in inference."""

    def test_equal_logits(self):
        """Test handling of equal logits."""
        postprocessor = PredictionPostprocessor()
        logits = np.array([[1.0, 1.0]])  # Equal logits

        result = postprocessor.process_prediction(logits)

        # Should still work, pick first class by argmax
        assert result["label"] in [0, 1]
        assert 0.4 < result["confidence"] < 0.6  # Close to 0.5

    def test_very_confident_prediction(self):
        """Test handling of very confident predictions."""
        postprocessor = PredictionPostprocessor()
        logits = np.array([[0.0, 10.0]])  # Very confident positive

        result = postprocessor.process_prediction(logits)

        assert result["sentiment"] == "positive"
        assert result["confidence"] > 0.999

    def test_batch_size_one(self):
        """Test batch processing with single example."""
        postprocessor = PredictionPostprocessor()
        logits = np.array([[0.1, 2.5]])

        results = postprocessor.process_batch(logits)

        assert len(results) == 1
        assert results[0]["sentiment"] == "positive"
