"""
Inference and prediction postprocessing module.
"""

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PredictionPostprocessor:
    """Class for postprocessing model predictions for API responses."""

    LABEL_MAP = {0: "negative", 1: "positive"}

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize postprocessor.

        Args:
            confidence_threshold: Minimum confidence for prediction
        """
        self.confidence_threshold = confidence_threshold

    def logits_to_probabilities(self, logits: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Convert model logits to probabilities using softmax.

        Args:
            logits: Raw model outputs [batch_size, num_classes]

        Returns:
            Probability distribution [batch_size, num_classes]
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return probabilities

    def probabilities_to_label(self, probabilities: np.ndarray) -> Tuple[int, float]:
        """
        Convert probabilities to predicted label and confidence.

        Args:
            probabilities: Probability distribution [num_classes]

        Returns:
            Tuple (predicted_label, confidence)
        """
        if probabilities.ndim > 1:
            probabilities = probabilities[0]

        label = int(np.argmax(probabilities))
        confidence = float(probabilities[label])

        return label, confidence

    def label_to_text(self, label: int) -> str:
        """
        Convert numeric label to text representation.

        Args:
            label: Numeric label (0 or 1)

        Returns:
            Text label

        Raises:
            ValueError: If label is invalid
        """
        if label not in self.LABEL_MAP:
            raise ValueError(f"Invalid label {label}. Expected 0 or 1")

        return self.LABEL_MAP[label]

    def process_prediction(
        self, logits: Union[np.ndarray, torch.Tensor], return_probabilities: bool = False
    ) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Complete prediction postprocessing pipeline.

        Args:
            logits: Raw model outputs
            return_probabilities: Whether to include probability distribution

        Returns:
            Dictionary with prediction results
        """
        # Convert to probabilities
        probs = self.logits_to_probabilities(logits)

        # Get predicted label and confidence
        label, confidence = self.probabilities_to_label(probs)

        # Convert to text
        sentiment = self.label_to_text(label)

        # Build response
        result = {"sentiment": sentiment, "confidence": confidence, "label": label}

        # Add probabilities if requested
        if return_probabilities:
            result["probabilities"] = {"negative": float(probs[0, 0]), "positive": float(probs[0, 1])}

        # Add warning if confidence is low
        if confidence < self.confidence_threshold:
            result["warning"] = f"Low confidence prediction ({confidence:.2%})"
            logger.warning("Low confidence prediction: %s", result)

        return result

    def process_batch(
        self, logits: Union[np.ndarray, torch.Tensor], return_probabilities: bool = False
    ) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
        """
        Process batch of predictions.

        Args:
            logits: Raw model outputs [batch_size, num_classes]
            return_probabilities: Whether to include probability distribution

        Returns:
            List of prediction results
        """
        probs = self.logits_to_probabilities(logits)

        results = []
        for i in range(probs.shape[0]):
            single_logits = logits[i : i + 1] if isinstance(logits, np.ndarray) else logits[i : i + 1, :]
            result = self.process_prediction(single_logits, return_probabilities)
            results.append(result)

        return results


def validate_logits(logits: Union[np.ndarray, torch.Tensor], expected_shape: Tuple[int, int] = None) -> bool:
    """
    Validate logits tensor.

    Args:
        logits: Model outputs to validate
        expected_shape: Expected shape (batch_size, num_classes)

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    # Convert to numpy for validation
    if isinstance(logits, torch.Tensor):
        logits_np = logits.detach().cpu().numpy()
    else:
        logits_np = logits

    # Check dimensionality
    if logits_np.ndim != 2:
        raise ValueError(f"Logits should be 2D, got shape {logits_np.shape}")

    # Check expected shape
    if expected_shape is not None:
        if logits_np.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {logits_np.shape}")

    # Check for NaN or Inf
    if np.isnan(logits_np).any():
        raise ValueError("Logits contain NaN values")

    if np.isinf(logits_np).any():
        raise ValueError("Logits contain Inf values")

    return True


def validate_probabilities(probabilities: np.ndarray) -> bool:
    """
    Validate probability distribution.

    Args:
        probabilities: Probability array to validate

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    # Check range [0, 1]
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError(f"Probabilities should be in [0, 1], got range [{probabilities.min()}, {probabilities.max()}]")

    # Check sum to 1 (with tolerance)
    prob_sums = probabilities.sum(axis=-1)
    if not np.allclose(prob_sums, 1.0, atol=1e-6):
        raise ValueError(f"Probabilities should sum to 1, got sums: {prob_sums}")

    return True
